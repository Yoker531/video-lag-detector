"""卡顿检测核心模块 - 实现帧差法、SSIM、光流法、滑动窗口比率法、组合法"""

import cv2
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
import json


@dataclass
class StutterSegment:
    """卡顿片段数据结构"""
    start_time: float   # 开始时间（秒）
    end_time: float     # 结束时间（秒）
    confidence: float   # 置信度 [0, 1]
    method: str         # 检测方法

    def to_dict(self):
        return asdict(self)


@dataclass
class DetectionResult:
    """检测结果数据结构"""
    video_path: str
    method: str
    fps: float
    total_frames: int
    duration: float
    segments: List[StutterSegment]
    metadata: dict

    def to_dict(self):
        d = asdict(self)
        d['segments'] = [s.to_dict() if isinstance(s, StutterSegment) else s for s in self.segments]
        return d

    def to_json(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class VideoCache:
    """视频帧缓存，避免重复读取。统一使用降采样帧，节省内存。"""

    DOWNSAMPLE_SIZE = (480, 270)  # 降采样分辨率

    def __init__(self, video_path: str):
        self.video_path = video_path
        self._frames = None
        self._fps = None
        self._total_frames = None
        self._duration = None
        self._diffs = None
        self._ssim_values = None
        self._flow_values = None

    def load(self):
        if self._frames is not None:
            return
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频: {self.video_path}")
        self._fps = cap.get(cv2.CAP_PROP_FPS)
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._duration = self._total_frames / self._fps if self._fps > 0 else 0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, self.DOWNSAMPLE_SIZE)
            frames.append(small)
        cap.release()
        self._frames = frames

    @property
    def frames(self):
        self.load()
        return self._frames

    @property
    def fps(self):
        self.load()
        return self._fps

    @property
    def total_frames(self):
        self.load()
        return self._total_frames

    @property
    def duration(self):
        self.load()
        return self._duration

    def get_frame_diffs(self) -> np.ndarray:
        """计算并缓存相邻帧灰度差（向量化）"""
        if self._diffs is not None:
            return self._diffs
        frames = self.frames
        n = len(frames) - 1
        diffs = np.empty(n, dtype=np.float64)
        for i in range(n):
            diffs[i] = np.mean(np.abs(frames[i+1].astype(np.float32) - frames[i].astype(np.float32)))
        self._diffs = diffs
        return diffs

    def get_ssim_values(self) -> np.ndarray:
        """计算并缓存相邻帧SSIM"""
        if self._ssim_values is not None:
            return self._ssim_values
        frames = self.frames
        n = len(frames) - 1
        ssim_vals = np.empty(n, dtype=np.float64)
        for i in range(n):
            ssim_vals[i] = _compute_ssim(frames[i], frames[i+1])
            if (i + 1) % 1000 == 0:
                print(f"    SSIM进度: {i+1}/{n}", flush=True)
        self._ssim_values = ssim_vals
        return ssim_vals

    def get_optical_flow_values(self) -> np.ndarray:
        """计算并缓存相邻帧光流幅度"""
        if self._flow_values is not None:
            return self._flow_values
        frames = self.frames
        n = len(frames) - 1
        flow_vals = np.empty(n, dtype=np.float64)
        for i in range(n):
            flow = cv2.calcOpticalFlowFarneback(
                frames[i], frames[i+1], None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            # 计算运动幅度（向量的模）
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_vals[i] = np.mean(mag)
            if (i + 1) % 1000 == 0:
                print(f"    光流进度: {i+1}/{n}", flush=True)
        self._flow_values = flow_vals
        return flow_vals


def _compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """手写SSIM计算（无需scikit-image）"""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim)


def _merge_segments(segments: List[StutterSegment], max_gap: float = 0.3) -> List[StutterSegment]:
    """合并间距过小的相邻卡顿段"""
    if not segments:
        return []
    segments.sort(key=lambda s: s.start_time)
    merged = [segments[0]]
    for seg in segments[1:]:
        last = merged[-1]
        if seg.start_time - last.end_time <= max_gap:
            merged[-1] = StutterSegment(
                start_time=last.start_time,
                end_time=max(last.end_time, seg.end_time),
                confidence=max(last.confidence, seg.confidence),
                method=last.method
            )
        else:
            merged.append(seg)
    return merged


def _diffs_to_segments(diffs: np.ndarray, threshold: float, min_frames: int,
                       fps: float, merge_gap: float, method: str,
                       above: bool = False) -> Tuple[List[StutterSegment], np.ndarray]:
    """
    将帧差/SSIM序列转换为卡顿段。
    above=False: 差值低于阈值判定为卡顿（帧差法）
    above=True: SSIM高于阈值判定为卡顿（SSIM法）
    返回 (segments, boolean_mask)
    """
    if above:
        mask = diffs > threshold
    else:
        mask = diffs < threshold

    segments = []
    stutter_start = None
    stutter_count = 0

    for i in range(len(mask)):
        if mask[i]:
            if stutter_start is None:
                stutter_start = i
            stutter_count += 1
        else:
            if stutter_count >= min_frames:
                start_t = stutter_start / fps
                end_t = (stutter_start + stutter_count + 1) / fps
                if above:
                    avg = np.mean(diffs[stutter_start:stutter_start+stutter_count])
                    conf = min(1.0, (avg - threshold) / (1.0 - threshold + 1e-6))
                else:
                    avg = np.mean(diffs[stutter_start:stutter_start+stutter_count])
                    conf = min(1.0, 1.0 - avg / (threshold + 1e-6))
                segments.append(StutterSegment(start_t, end_t, round(conf, 3), method))
            stutter_start = None
            stutter_count = 0

    if stutter_count >= min_frames:
        start_t = stutter_start / fps
        end_t = (stutter_start + stutter_count + 1) / fps
        if above:
            avg = np.mean(diffs[stutter_start:stutter_start+stutter_count])
            conf = min(1.0, (avg - threshold) / (1.0 - threshold + 1e-6))
        else:
            avg = np.mean(diffs[stutter_start:stutter_start+stutter_count])
            conf = min(1.0, 1.0 - avg / (threshold + 1e-6))
        segments.append(StutterSegment(start_t, end_t, round(conf, 3), method))

    return _merge_segments(segments, merge_gap), mask


class FrameDiffDetector:
    """基于帧差法的卡顿检测器"""

    def __init__(self, threshold: float = 5.0, min_stutter_frames: int = 30, merge_gap: float = 0.5):
        self.threshold = threshold
        self.min_stutter_frames = min_stutter_frames
        self.merge_gap = merge_gap

    def detect(self, video_path: str, cache: VideoCache = None) -> DetectionResult:
        if cache is None:
            cache = VideoCache(video_path)
        diffs = cache.get_frame_diffs()
        segments, _ = _diffs_to_segments(diffs, self.threshold, self.min_stutter_frames,
                                         cache.fps, self.merge_gap, "frame_diff", above=False)
        return DetectionResult(
            video_path=video_path, method="frame_diff",
            fps=cache.fps, total_frames=cache.total_frames,
            duration=cache.duration, segments=segments,
            metadata={"threshold": self.threshold, "min_stutter_frames": self.min_stutter_frames}
        )


class SSIMDetector:
    """基于SSIM的卡顿检测器"""

    def __init__(self, threshold: float = 0.95, min_stutter_frames: int = 30, merge_gap: float = 0.5):
        self.threshold = threshold
        self.min_stutter_frames = min_stutter_frames
        self.merge_gap = merge_gap

    def detect(self, video_path: str, cache: VideoCache = None) -> DetectionResult:
        if cache is None:
            cache = VideoCache(video_path)
        ssim_vals = cache.get_ssim_values()
        segments, _ = _diffs_to_segments(ssim_vals, self.threshold, self.min_stutter_frames,
                                         cache.fps, self.merge_gap, "ssim", above=True)
        return DetectionResult(
            video_path=video_path, method="ssim",
            fps=cache.fps, total_frames=cache.total_frames,
            duration=cache.duration, segments=segments,
            metadata={"threshold": self.threshold, "min_stutter_frames": self.min_stutter_frames}
        )


class OpticalFlowDetector:
    """基于光流法的卡顿检测器 - 通过运动向量幅度判断画面是否冻结"""

    def __init__(self, threshold: float = 0.5, min_stutter_frames: int = 30, merge_gap: float = 0.5):
        self.threshold = threshold
        self.min_stutter_frames = min_stutter_frames
        self.merge_gap = merge_gap

    def detect(self, video_path: str, cache: VideoCache = None) -> DetectionResult:
        if cache is None:
            cache = VideoCache(video_path)
        flow_vals = cache.get_optical_flow_values()
        # 光流幅度低于阈值 → 画面冻结
        segments, _ = _diffs_to_segments(flow_vals, self.threshold, self.min_stutter_frames,
                                         cache.fps, self.merge_gap, "optical_flow", above=False)
        return DetectionResult(
            video_path=video_path, method="optical_flow",
            fps=cache.fps, total_frames=cache.total_frames,
            duration=cache.duration, segments=segments,
            metadata={"threshold": self.threshold, "min_stutter_frames": self.min_stutter_frames}
        )


class RatioDetector:
    """滑动窗口比率法检测器 - 适合基线本身就有很多静止帧的视频"""

    def __init__(self, low_threshold=1.0, ratio_threshold=0.70,
                 window_sec=1.0, min_duration=1.0, merge_gap=0.5,
                 adaptive=False, adaptive_k=2.0):
        self.low_threshold = low_threshold
        self.ratio_threshold = ratio_threshold
        self.window_sec = window_sec
        self.min_duration = min_duration
        self.merge_gap = merge_gap
        self.adaptive = adaptive
        self.adaptive_k = adaptive_k

    def detect(self, video_path: str, cache: VideoCache = None) -> DetectionResult:
        if cache is None:
            cache = VideoCache(video_path)
        diffs = cache.get_frame_diffs()
        fps = cache.fps
        win = int(fps * self.window_sec)
        step = win // 2  # 半窗口步长

        # 计算每个窗口的 low-diff 比率
        ratios = []
        for i in range(0, len(diffs) - win, step):
            seg = diffs[i:i+win]
            ratio = (seg < self.low_threshold).sum() / len(seg)
            ratios.append((i, i + win, ratio))

        # 自适应阈值：用均值 + K*标准差替代固定阈值
        actual_threshold = self.ratio_threshold
        if self.adaptive and ratios:
            all_ratios = [r for _, _, r in ratios]
            mean_r = np.mean(all_ratios)
            std_r = np.std(all_ratios)
            actual_threshold = mean_r + self.adaptive_k * std_r
            actual_threshold = max(actual_threshold, 0.5)  # 下限0.5

        # 标记卡顿窗口
        stutter_windows = []
        for start_f, end_f, r in ratios:
            if r >= actual_threshold:
                stutter_windows.append((start_f, end_f))

        # 合并重叠/相邻窗口
        if not stutter_windows:
            return DetectionResult(video_path, "ratio", fps, cache.total_frames,
                                   cache.duration, [], {})

        merged_windows = [stutter_windows[0]]
        for s, e in stutter_windows[1:]:
            ls, le = merged_windows[-1]
            if s - le <= step:
                merged_windows[-1] = (ls, max(le, e))
            else:
                merged_windows.append((s, e))

        # 转为 StutterSegment，过滤最小时长
        segments = []
        for s, e in merged_windows:
            dur = (e - s) / fps
            if dur >= self.min_duration:
                avg_ratio = np.mean([r for _, _, r in ratios
                                     if _ >= s and _ <= e]) if ratios else 0
                conf = min(1.0, (avg_ratio - self.ratio_threshold) / (1.0 - self.ratio_threshold + 1e-6))
                segments.append(StutterSegment(s/fps, e/fps, round(conf, 3), "ratio"))

        metadata = {"low_threshold": self.low_threshold,
                    "ratio_threshold": self.ratio_threshold,
                    "window_sec": self.window_sec}
        if self.adaptive:
            metadata["adaptive"] = True
            metadata["adaptive_k"] = self.adaptive_k
            metadata["actual_threshold"] = round(actual_threshold, 4)

        return DetectionResult(video_path, "ratio", fps, cache.total_frames,
                               cache.duration, segments, metadata)


class CombinedDetector:
    """组合检测器：帧差法 + SSIM + 滑动窗口比率法，取并集"""

    def __init__(self, frame_diff_threshold=5.0, ssim_threshold=0.95,
                 min_stutter_frames=30, merge_gap=0.5,
                 ratio_low_threshold=1.0, ratio_threshold=0.70):
        self.fd_thresh = frame_diff_threshold
        self.ssim_thresh = ssim_threshold
        self.min_frames = min_stutter_frames
        self.merge_gap = merge_gap
        self.ratio_low = ratio_low_threshold
        self.ratio_thresh = ratio_threshold

    def detect(self, video_path: str, cache: VideoCache = None) -> DetectionResult:
        if cache is None:
            cache = VideoCache(video_path)

        fd = FrameDiffDetector(self.fd_thresh, self.min_frames, self.merge_gap)
        ss = SSIMDetector(self.ssim_thresh, self.min_frames, self.merge_gap)
        ra = RatioDetector(self.ratio_low, self.ratio_thresh, merge_gap=self.merge_gap)

        result_fd = fd.detect(video_path, cache)
        result_ss = ss.detect(video_path, cache)
        result_ra = ra.detect(video_path, cache)

        all_segs = result_fd.segments + result_ss.segments + result_ra.segments
        all_segs.sort(key=lambda s: s.start_time)
        merged = _merge_segments(all_segs, self.merge_gap)
        for s in merged:
            s.method = "combined"

        return DetectionResult(
            video_path=video_path, method="combined",
            fps=cache.fps, total_frames=cache.total_frames,
            duration=cache.duration, segments=merged,
            metadata={
                "frame_diff_count": len(result_fd.segments),
                "ssim_count": len(result_ss.segments),
                "ratio_count": len(result_ra.segments),
                "combined_count": len(merged)
            }
        )


def analyze_diffs(video_path: str):
    """分析帧差值和SSIM分布，辅助调参"""
    cache = VideoCache(video_path)
    diffs = cache.get_frame_diffs()
    ssim_vals = cache.get_ssim_values()

    print(f"视频: {video_path}")
    print(f"FPS: {cache.fps}, 总帧数: {cache.total_frames}, 时长: {cache.duration:.1f}s")
    print(f"\n--- 帧差值 (灰度均值差) 统计 ---")
    print(f"  min={diffs.min():.2f}, max={diffs.max():.2f}, mean={diffs.mean():.2f}, median={np.median(diffs):.2f}")
    print(f"  p5={np.percentile(diffs,5):.2f}, p10={np.percentile(diffs,10):.2f}, p25={np.percentile(diffs,25):.2f}")
    print(f"  p75={np.percentile(diffs,75):.2f}, p90={np.percentile(diffs,90):.2f}, p95={np.percentile(diffs,95):.2f}")
    print(f"  差值<1.0的帧占比: {(diffs<1.0).sum()/len(diffs)*100:.1f}%")
    print(f"  差值<2.0的帧占比: {(diffs<2.0).sum()/len(diffs)*100:.1f}%")
    print(f"  差值<5.0的帧占比: {(diffs<5.0).sum()/len(diffs)*100:.1f}%")

    print(f"\n--- SSIM 统计 ---")
    print(f"  min={ssim_vals.min():.4f}, max={ssim_vals.max():.4f}, mean={ssim_vals.mean():.4f}, median={np.median(ssim_vals):.4f}")
    print(f"  SSIM>0.99的帧占比: {(ssim_vals>0.99).sum()/len(ssim_vals)*100:.1f}%")
    print(f"  SSIM>0.98的帧占比: {(ssim_vals>0.98).sum()/len(ssim_vals)*100:.1f}%")
    print(f"  SSIM>0.95的帧占比: {(ssim_vals>0.95).sum()/len(ssim_vals)*100:.1f}%")

    # 找出连续低差异帧的位置
    print(f"\n--- 连续低差值段 (diff<2.0, >=30帧) ---")
    _find_streaks(diffs, 2.0, 30, cache.fps, above=False)

    print(f"\n--- 连续高SSIM段 (ssim>0.99, >=30帧) ---")
    _find_streaks(ssim_vals, 0.99, 30, cache.fps, above=True)

    return diffs, ssim_vals


def _find_streaks(values, threshold, min_len, fps, above=False):
    """找出连续满足条件的段"""
    if above:
        mask = values > threshold
    else:
        mask = values < threshold

    start = None
    count = 0
    for i in range(len(mask)):
        if mask[i]:
            if start is None:
                start = i
            count += 1
        else:
            if count >= min_len:
                print(f"  {start/fps:.1f}s - {(start+count)/fps:.1f}s (帧 {start}-{start+count}, {count}帧)")
            start = None
            count = 0
    if count >= min_len:
        print(f"  {start/fps:.1f}s - {(start+count)/fps:.1f}s (帧 {start}-{start+count}, {count}帧)")


def grid_search_ratio(cache: VideoCache, evaluate_fn, gt_segments,
                      diff_thresholds=(0.3, 0.5, 1.0, 2.0, 3.0),
                      ratio_thresholds=(0.55, 0.60, 0.65, 0.70, 0.75),
                      window_sizes=(0.5, 1.0, 2.0)):
    """
    对滑动窗口比率法做网格搜索，返回最优参数和全部结果。
    evaluate_fn: callable(pred_segments, gt_segments) -> dict with 'f1', 'precision', 'recall'
    """
    diffs = cache.get_frame_diffs()
    fps = cache.fps
    best_f1 = -1
    best_params = {}
    best_result = None
    all_results = []

    total = len(diff_thresholds) * len(ratio_thresholds) * len(window_sizes)
    idx = 0

    for dt in diff_thresholds:
        for rt in ratio_thresholds:
            for ws in window_sizes:
                idx += 1
                det = RatioDetector(low_threshold=dt, ratio_threshold=rt,
                                    window_sec=ws, merge_gap=0.5)
                result = det.detect(cache.video_path, cache=cache)
                metrics = evaluate_fn(result.segments, gt_segments)

                row = {"diff_thresh": dt, "ratio_thresh": rt, "window_sec": ws,
                       "segments": len(result.segments), **metrics}
                all_results.append(row)

                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    best_params = {"diff_thresh": dt, "ratio_thresh": rt, "window_sec": ws}
                    best_result = result

                print(f"  [{idx}/{total}] dt={dt}, rt={rt}, ws={ws} → "
                      f"segs={len(result.segments):>2}, P={metrics['precision']:.3f}, "
                      f"R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}", flush=True)

    return best_params, best_f1, best_result, all_results
