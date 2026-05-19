"""离线视频卡顿检测 - 主入口脚本"""

import argparse
import json
import os
import sys
import time

from detector import (FrameDiffDetector, SSIMDetector, RatioDetector,
                      CombinedDetector, OpticalFlowDetector, VideoCache, grid_search_ratio)
from evaluator import load_ground_truth, evaluate, generate_report


def log(msg):
    print(msg, flush=True)


def run_with_cache(method_name, cache, video_path, gt_segments, output_dir, **kwargs):
    """使用缓存运行单个检测方法"""
    if method_name == "frame_diff":
        detector = FrameDiffDetector(kwargs['fd_thresh'], kwargs['min_frames'], kwargs['merge_gap'])
    elif method_name == "ssim":
        detector = SSIMDetector(kwargs['ssim_thresh'], kwargs['min_frames'], kwargs['merge_gap'])
    elif method_name == "ratio":
        detector = RatioDetector(kwargs.get('ratio_low', 1.0), kwargs.get('ratio_thresh', 0.70),
                                 merge_gap=kwargs['merge_gap'])
    elif method_name == "optical_flow":
        detector = OpticalFlowDetector(kwargs['of_thresh'], kwargs['min_frames'], kwargs['merge_gap'])
    else:
        detector = CombinedDetector(kwargs['fd_thresh'], kwargs['ssim_thresh'],
                                    kwargs['min_frames'], kwargs['merge_gap'])

    t0 = time.time()
    result = detector.detect(video_path, cache=cache)
    elapsed = time.time() - t0

    metrics = None
    if gt_segments:
        metrics = evaluate(result.segments, gt_segments)

    result.to_json(os.path.join(output_dir, f"result_{method_name}.json"))
    return result, metrics, elapsed


def main():
    parser = argparse.ArgumentParser(description="离线视频卡顿检测工具")
    parser.add_argument("--video", type=str, required=True, help="视频文件路径")
    parser.add_argument("--gt", type=str, default=None, help="Ground Truth xlsx文件路径")
    parser.add_argument("--method", type=str,
                        choices=["frame_diff", "ssim", "optical_flow", "ratio", "combined", "all"],
                        default="combined", help="检测方法")
    parser.add_argument("--grid-search", action="store_true", help="对滑动窗口比率法做网格搜索找最优参数")
    parser.add_argument("--output", type=str, default="results", help="结果输出目录")
    parser.add_argument("--frame-diff-threshold", type=float, default=5.0, help="帧差法阈值")
    parser.add_argument("--ssim-threshold", type=float, default=0.95, help="SSIM阈值")
    parser.add_argument("--min-frames", type=int, default=30, help="最小卡顿帧数")
    parser.add_argument("--optical-flow-threshold", type=float, default=0.5, help="光流法阈值")
    parser.add_argument("--merge-gap", type=float, default=0.5, help="合并间距(秒)")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        log(f"错误: 视频文件不存在: {args.video}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    kwargs = {
        'fd_thresh': args.frame_diff_threshold,
        'ssim_thresh': args.ssim_threshold,
        'of_thresh': args.optical_flow_threshold,
        'min_frames': args.min_frames,
        'merge_gap': args.merge_gap
    }

    # 加载GT
    gt_segments = None
    if args.gt and os.path.exists(args.gt):
        log(f"[1/4] 加载 Ground Truth...")
        gt_segments = load_ground_truth(args.gt)
        log(f"  GT: {len(gt_segments)} 个卡顿段")

    # 读取视频（只读一次）
    log(f"[2/4] 读取视频: {args.video}")
    t0 = time.time()
    cache = VideoCache(args.video)
    cache.load()
    log(f"  已读取 {cache.total_frames} 帧 ({cache.fps}fps, {cache.duration:.1f}s), 耗时 {time.time()-t0:.1f}s")

    # 预计算帧差值
    log(f"[3/4] 计算特征值...")
    t0 = time.time()
    diffs = cache.get_frame_diffs()
    log(f"  帧差值计算完成 ({len(diffs)} 个值), 耗时 {time.time()-t0:.1f}s")
    log(f"  差值范围: min={diffs.min():.2f}, max={diffs.max():.2f}, mean={diffs.mean():.2f}")

    t0 = time.time()
    ssim_vals = cache.get_ssim_values()
    log(f"  SSIM计算完成 ({len(ssim_vals)} 个值), 耗时 {time.time()-t0:.1f}s")
    log(f"  SSIM范围: min={ssim_vals.min():.4f}, max={ssim_vals.max():.4f}, mean={ssim_vals.mean():.4f}")

    # 如果需要光流法，预计算光流
    need_flow = args.method in ("optical_flow", "all")
    if need_flow:
        t0 = time.time()
        flow_vals = cache.get_optical_flow_values()
        log(f"  光流计算完成 ({len(flow_vals)} 个值), 耗时 {time.time()-t0:.1f}s")
        log(f"  光流范围: min={flow_vals.min():.2f}, max={flow_vals.max():.2f}, mean={flow_vals.mean():.2f}")

    # 网格搜索模式
    grid_search_done = False
    if args.grid_search:
        if not gt_segments:
            log("错误: --grid-search 需要 --gt 参数")
            sys.exit(1)
        log(f"\n--- 网格搜索：滑动窗口比率法参数优化 ---")
        t0 = time.time()
        best_params, best_f1, best_result, all_results = grid_search_ratio(
            cache, evaluate, gt_segments)
        elapsed = time.time() - t0
        log(f"\n搜索完成! 耗时 {elapsed:.1f}s")
        log(f"最优参数: {best_params}")
        log(f"最优 F1: {best_f1:.4f}")
        # 保存最优结果
        best_result.to_json(os.path.join(args.output, "result_ratio.json"))
        import json
        with open(os.path.join(args.output, "grid_search_results.json"), 'w') as f:
            json.dump({"best_params": best_params, "best_f1": best_f1,
                       "all_results": all_results}, f, indent=2, ensure_ascii=False)
        log(f"结果已保存至 {args.output}/")
        grid_search_done = True
        # 如果只跑滑动窗口比率法，网格搜索结果已是最优，无需再跑
        if args.method == "ratio":
            return

    # 运行检测
    if args.method == "all":
        methods = ["frame_diff", "ssim", "optical_flow", "ratio", "combined"]
        # 如果网格搜索已完成滑动窗口比率法，跳过重复运行
        if grid_search_done:
            methods.remove("ratio")
    else:
        methods = [args.method]

    log(f"[4/4] 运行检测 (方法: {', '.join(methods)})")
    all_results = {}

    for m in methods:
        log(f"\n  --- {m} ---")
        result, metrics, elapsed = run_with_cache(m, cache, args.video, gt_segments, args.output, **kwargs)
        all_results[m] = {"result": result, "metrics": metrics, "elapsed": elapsed}

        log(f"  检测段数: {len(result.segments)}, 耗时: {elapsed:.1f}s")
        for i, seg in enumerate(result.segments):
            log(f"    [{i+1}] {seg.start_time:.1f}s - {seg.end_time:.1f}s ({seg.end_time-seg.start_time:.2f}s)")
        if metrics:
            log(f"  TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}")
            log(f"  Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

    # 对比报告
    if len(methods) > 1:
        log(f"\n{'='*60}")
        log("三种方法对比")
        log(f"{'='*60}")
        log(f"{'方法':<12} {'段数':>4} {'耗时(s)':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
        log("-" * 60)

        report_lines = ["# 方法对比报告\n", f"视频: {args.video}", f"参数: {kwargs}\n"]
        report_lines.append("| 方法 | 检测段数 | 耗时(s) | Precision | Recall | F1 |")
        report_lines.append("|------|----------|---------|-----------|--------|-----|")

        for m in methods:
            r = all_results[m]
            n = len(r["result"].segments)
            t = r["elapsed"]
            met = r["metrics"]
            if met:
                line = f"{m:<12} {n:>4} {t:>8.1f} {met['precision']:>10.4f} {met['recall']:>8.4f} {met['f1']:>8.4f}"
                report_lines.append(f"| {m} | {n} | {t:.1f} | {met['precision']:.4f} | {met['recall']:.4f} | {met['f1']:.4f} |")
            else:
                line = f"{m:<12} {n:>4} {t:>8.1f} {'N/A':>10} {'N/A':>8} {'N/A':>8}"
                report_lines.append(f"| {m} | {n} | {t:.1f} | - | - | - |")
            log(line)

        report_path = os.path.join(args.output, "comparison_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        log(f"\n对比报告已保存至: {report_path}")

    # 为每个方法生成单独的评估报告
    if gt_segments:
        for m in methods:
            r = all_results[m]
            report = generate_report(r["result"], gt_segments)
            report_path = os.path.join(args.output, f"report_{m}.md")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            log(f"评估报告: {report_path}")

    log("\n完成!")


if __name__ == "__main__":
    main()
