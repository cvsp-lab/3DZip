"""Per-scene caching for evaluation on multi-question benchmarks.

Real deployments (embodied QA, robotic exploration) query the same 3D scene
many times. This module lets an eval loop:

  1. Load each scene's RGB-D tensors from disk exactly once  (``SceneCache``)
  2. Encode each scene's visual features through the vision tower exactly once
     (``enable_scene_feature_cache``)

Bit-equivalent to the stock path when the same scene is queried once; skips
redundant disk reads and vision-tower forwards when a scene is queried
multiple consecutive times. Combine with ``group_by_scene`` to maximize hits.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Dict, List, Optional

import torch

from llava.mm_utils import process_videos


class SceneCache:
    """LRU cache of RGB-D scene tensors keyed by video path.

    Each entry stores the 4 tensors produced by ``process_videos`` — already
    moved to ``device`` in ``dtype``. Bumps ``hits``/``misses`` counters and
    records the wall-clock cost of each disk-loading miss in ``io_times_ms``.
    """

    def __init__(self, capacity: int, device, dtype=torch.bfloat16):
        self.capacity = capacity
        self.device = device
        self.dtype = dtype
        self._store: "OrderedDict[str, Dict[str, torch.Tensor]]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.io_times_ms: List[float] = []

    def get(self, video_path: str, processor, text: Optional[str] = None):
        """Return the RGB-D bundle for ``video_path``, loading on miss."""
        if video_path in self._store:
            self._store.move_to_end(video_path)
            self.hits += 1
            return self._store[video_path]
        self.misses += 1
        t0 = time.time()
        v = process_videos(video_path, processor, mode='uniform',
                           device=self.device, text=text)
        entry = {
            'images':     v['images'].to(self.device, dtype=self.dtype),
            'depths':     v['depths'].to(self.device, dtype=self.dtype),
            'poses':      v['poses'].to(self.device,  dtype=self.dtype),
            'intrinsics': v['intrinsics'].to(self.device, dtype=self.dtype),
        }
        torch.cuda.synchronize()
        self.io_times_ms.append((time.time() - t0) * 1000)
        self._store[video_path] = entry
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)
        return entry


def enable_scene_feature_cache(model, capacity: int = 8):
    """Install an LRU cache on the video tower's ``forward`` output.

    The eval loop must set ``video_tower.scene_key`` (typically the scene
    directory path) right before each ``model.generate`` call — the cache
    keys on that string, not on tensor memory addresses (memory can alias
    across scenes once the RGB-D cache evicts).

    Returns the underlying ``OrderedDict`` cache and installs counters
    ``video_tower.feature_cache_hits`` / ``feature_cache_misses``.
    """
    video_tower = model.get_model().get_video_tower()
    if getattr(video_tower, '_scene_feature_cache_installed', False):
        return video_tower._scene_feature_cache

    orig_forward = video_tower.forward
    cache: "OrderedDict" = OrderedDict()

    def _snapshot(ret):
        if isinstance(ret, tuple):
            return tuple(
                x.detach().clone() if torch.is_tensor(x) else x for x in ret
            )
        return ret.detach().clone()

    def cached_forward(images, depths, poses, intrinsics, *args, **kwargs):
        key = getattr(video_tower, 'scene_key', None)
        if key is None:
            # No scene id set — fall back to a per-tensor key (may alias!).
            key = int(images.data_ptr())

        if key in cache:
            cache.move_to_end(key)
            ret, vox_xyz = cache[key]
            video_tower._voxel_xyz = vox_xyz
            video_tower.feature_cache_hits += 1
            return ret

        # Miss: run the real forward and snapshot the output.
        ret = orig_forward(images, depths, poses, intrinsics, *args, **kwargs)
        vox_xyz = getattr(video_tower, '_voxel_xyz', None)
        vox_xyz_snapshot = (
            [x.detach().clone() for x in vox_xyz]
            if vox_xyz is not None else None
        )
        cache[key] = (_snapshot(ret), vox_xyz_snapshot)
        while len(cache) > capacity:
            cache.popitem(last=False)
        video_tower.feature_cache_misses += 1
        return ret

    video_tower.forward = cached_forward
    video_tower._scene_feature_cache = cache
    video_tower._scene_feature_cache_installed = True
    video_tower.feature_cache_hits = 0
    video_tower.feature_cache_misses = 0
    return cache


def group_by_scene(questions: List[dict]) -> List[dict]:
    """Sort questions so same-scene queries run consecutively.

    Assumes each question dict has a ``video`` field (scene identifier).
    Stable order within a scene is preserved.
    """
    return sorted(questions, key=lambda q: q['video'])
