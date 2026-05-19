"""评估模块 - 计算检测结果与Ground Truth之间的性能指标"""

import zipfile
import xml.etree.ElementTree as ET
from typing import List, Tuple
from detector import StutterSegment, DetectionResult


def parse_time_range(time_str: str) -> Tuple[float, float]:
    """解析时间范围字符串，如 '0:03-0:04' -> (3.0, 4.0)"""
    parts = time_str.strip().split('-')
    start = _parse_time(parts[0])
    end = _parse_time(parts[1])
    return start, end


def _parse_time(time_str: str) -> float:
    """解析时间字符串为秒数，支持 M:SS 和 H:MM:SS 格式"""
    time_str = time_str.strip()
    parts = time_str.split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    else:
        raise ValueError(f"无法解析时间: {time_str}")


def _read_xlsx_column(xlsx_path: str, col_index: int = 0) -> list:
    """用zipfile直接读取xlsx指定列数据，无需openpyxl"""
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(xlsx_path, 'r') as z:
        # 读取共享字符串
        ss_xml = z.read('xl/sharedStrings.xml')
        ss_tree = ET.fromstring(ss_xml)
        strings = []
        for si in ss_tree.findall('.//s:si', ns):
            texts = si.findall('.//s:t', ns)
            strings.append(''.join(t.text or '' for t in texts))

        # 读取sheet数据
        sheet_xml = z.read('xl/worksheets/sheet1.xml')
        tree = ET.fromstring(sheet_xml)
        rows = tree.findall('.//s:sheetData/s:row', ns)

        values = []
        for row in rows:
            cells = row.findall('s:c', ns)
            if len(cells) > col_index:
                c = cells[col_index]
                t = c.attrib.get('t', '')
                v = c.find('s:v', ns)
                if v is not None:
                    if t == 's':
                        values.append(strings[int(v.text)])
                    else:
                        values.append(v.text)
        return values


def load_ground_truth(xlsx_path: str) -> List[StutterSegment]:
    """从xlsx文件加载Ground Truth数据"""
    values = _read_xlsx_column(xlsx_path, 0)
    # 跳过表头
    segments = []
    for time_str in values[1:]:  # 第一行是表头"卡顿时间"
        start, end = parse_time_range(time_str)
        segments.append(StutterSegment(
            start_time=start,
            end_time=end,
            confidence=1.0,
            method="ground_truth"
        ))
    return segments


def compute_iou(seg1: StutterSegment, seg2: StutterSegment) -> float:
    """计算两个时间段的IoU（Intersection over Union）"""
    inter_start = max(seg1.start_time, seg2.start_time)
    inter_end = min(seg1.end_time, seg2.end_time)
    inter = max(0, inter_end - inter_start)

    union_start = min(seg1.start_time, seg2.start_time)
    union_end = max(seg1.end_time, seg2.end_time)
    union = union_end - union_start

    return inter / union if union > 0 else 0.0


def evaluate(pred_segments: List[StutterSegment],
             gt_segments: List[StutterSegment],
             iou_threshold: float = 0.5) -> dict:
    """
    评估检测结果

    返回:
        {
            'tp': int, 'fp': int, 'fn': int,
            'precision': float, 'recall': float, 'f1': float,
            'matches': list of (pred_idx, gt_idx, iou)
        }
    """
    matched_gt = set()
    tp = 0
    fp = 0
    matches = []

    for i, pred in enumerate(pred_segments):
        best_iou = 0
        best_j = -1
        for j, gt in enumerate(gt_segments):
            if j in matched_gt:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_threshold and best_j >= 0:
            tp += 1
            matched_gt.add(best_j)
            matches.append((i, best_j, best_iou))
        else:
            fp += 1

    fn = len(gt_segments) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'matches': matches
    }


def generate_report(detection_result: DetectionResult,
                    gt_segments: List[StutterSegment],
                    iou_threshold: float = 0.5) -> str:
    """生成评估报告（Markdown格式）"""
    metrics = evaluate(detection_result.segments, gt_segments, iou_threshold)

    lines = [
        "# 卡顿检测评估报告\n",
        "## 检测方法",
        f"- **方法:** {detection_result.method}",
        f"- **视频:** {detection_result.video_path}",
        f"- **FPS:** {detection_result.fps}",
        f"- **总帧数:** {detection_result.total_frames}",
        f"- **视频时长:** {detection_result.duration:.1f} 秒",
        f"- **检测参数:** {detection_result.metadata}\n",
        "## 评估指标",
        f"| 指标 | 值 |",
        f"|------|------|",
        f"| TP (真阳性) | {metrics['tp']} |",
        f"| FP (假阳性) | {metrics['fp']} |",
        f"| FN (假阴性) | {metrics['fn']} |",
        f"| **Precision** | **{metrics['precision']:.4f}** |",
        f"| **Recall** | **{metrics['recall']:.4f}** |",
        f"| **F1 Score** | **{metrics['f1']:.4f}** |",
        f"| IoU 阈值 | {iou_threshold} |\n",
        "## 检测到的卡顿段\n",
        "| # | 开始时间 | 结束时间 | 时长(s) | 置信度 | 匹配GT |",
        "|---|----------|----------|---------|--------|--------|",
    ]

    for i, seg in enumerate(detection_result.segments):
        matched = "否"
        for pred_idx, gt_idx, iou in metrics['matches']:
            if pred_idx == i:
                matched = f"是 (IoU={iou:.2f})"
                break
        duration = seg.end_time - seg.start_time
        lines.append(
            f"| {i+1} | {seg.start_time:.1f}s | {seg.end_time:.1f}s | {duration:.2f} | {seg.confidence:.3f} | {matched} |"
        )

    lines.append("\n## Ground Truth 参考\n")
    lines.append("| # | 开始时间 | 结束时间 | 时长(s) | 状态 |")
    lines.append("|---|----------|----------|---------|------|")

    matched_gt_indices = {gt_idx for _, gt_idx, _ in metrics['matches']}
    for i, gt in enumerate(gt_segments):
        status = "已检出" if i in matched_gt_indices else "漏检"
        duration = gt.end_time - gt.start_time
        lines.append(f"| {i+1} | {gt.start_time:.1f}s | {gt.end_time:.1f}s | {duration:.2f} | {status} |")

    return "\n".join(lines)
