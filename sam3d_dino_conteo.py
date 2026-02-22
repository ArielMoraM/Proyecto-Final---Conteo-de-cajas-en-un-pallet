#!/usr/bin/env python3
"""
Pipeline SAM (base usado en notebook SAM3D) + Grounding DINO
para conteo semántico de cajas en frames de video.

Uso ejemplo:
python sam3d_dino_conteo.py \
  --video_path /ruta/video.mp4 \
  --sam_checkpoint /ruta/sam_vit_b.pth \
  --output_dir ./outputs \
  --max_frames 8
"""

import argparse
import glob
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# Requiere instalar segment-anything:
# pip install git+https://github.com/facebookresearch/segment-anything.git
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


@dataclass
class Config:
    video_path: str
    sam_checkpoint: str
    output_dir: str
    model_type: str = "vit_b"
    max_frames: int = 8
    resize_w: int = 640
    resize_h: int = 360
    min_mask_region_area: int = 1000
    max_mask_region_area: int = 50000
    text_prompt: str = "cardboard box . carton box . pallet box ."
    box_threshold: float = 0.20
    text_threshold: float = 0.15
    iou_threshold: float = 0.05
    dino_model_id: str = "IDEA-Research/grounding-dino-base"
    max_mask_area_ratio: float = 0.40
    frame_stride: int = 1
    save_video: bool = False


def extract_frames(video_path: str, frames_dir: str, max_frames: int, resize_w: int, resize_h: int, frame_stride: int) -> List[str]:
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames en video: {total_frames}")

    frame_paths = []
    count = 0
    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % max(1, frame_stride) == 0:
            frame_small = cv2.resize(frame, (resize_w, resize_h))
            save_path = os.path.join(frames_dir, f"frame_{saved:03d}.jpg")
            cv2.imwrite(save_path, frame_small)
            frame_paths.append(save_path)
            print(f"Guardado: {save_path}")
            saved += 1
            if saved >= max_frames:
                break

        frame_idx += 1
        count += 1

    cap.release()
    print(f"Total frames extraídos: {len(frame_paths)}")
    return frame_paths


def bbox_iou(box_a: List[float], box_b: List[float]) -> float:
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_w = max(0.0, x_b - x_a)
    inter_h = max(0.0, y_b - y_a)
    inter = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter + 1e-6
    return float(inter / union)


def dino_detect_boxes(
    image_rgb: np.ndarray,
    processor,
    dino_model,
    device: str,
    prompt: str,
    box_threshold: float,
    text_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    pil_img = Image.fromarray(image_rgb)
    inputs = processor(images=pil_img, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = dino_model(**inputs)

    target_sizes = torch.tensor([pil_img.size[::-1]]).to(device)
    post_fn = processor.post_process_grounded_object_detection

    # Fallback robusto entre versiones de transformers
    call_variants = [
        {"input_ids": inputs.input_ids, "target_sizes": target_sizes, "box_threshold": box_threshold, "text_threshold": text_threshold},
        {"input_ids": inputs.input_ids, "target_sizes": target_sizes, "threshold": box_threshold, "text_threshold": text_threshold},
        {"input_ids": inputs.input_ids, "target_sizes": target_sizes, "threshold": box_threshold},
        {"input_ids": inputs.input_ids, "target_sizes": target_sizes},
        {"target_sizes": target_sizes, "threshold": box_threshold, "text_threshold": text_threshold},
        {"target_sizes": target_sizes, "threshold": box_threshold},
        {"target_sizes": target_sizes},
        {},
    ]

    results = None
    last_error = None
    for kwargs in call_variants:
        try:
            results = post_fn(outputs, **kwargs)[0]
            break
        except TypeError as exc:
            last_error = exc
            continue

    if results is None:
        raise TypeError(f"No fue posible ejecutar post_process_grounded_object_detection con la versión instalada de transformers: {last_error}")

    boxes = results["boxes"].detach().cpu().numpy() if len(results["boxes"]) else np.empty((0, 4))
    scores = results["scores"].detach().cpu().numpy() if len(results["scores"]) else np.array([])
    labels = results["labels"]
    return boxes, scores, labels


def sam_dino_boxes_for_frame(
    image_rgb: np.ndarray,
    mask_generator: SamAutomaticMaskGenerator,
    processor,
    dino_model,
    device: str,
    cfg: Config,
):
    sam_masks = mask_generator.generate(image_rgb)
    sam_total = len(sam_masks)
    frame_area = image_rgb.shape[0] * image_rgb.shape[1]
    max_area_limit = min(cfg.max_mask_region_area, int(frame_area * cfg.max_mask_area_ratio))
    sam_masks = [m for m in sam_masks if m["segmentation"].sum() <= max_area_limit]

    dino_boxes, dino_scores, dino_labels = dino_detect_boxes(
        image_rgb=image_rgb,
        processor=processor,
        dino_model=dino_model,
        device=device,
        prompt=cfg.text_prompt,
        box_threshold=cfg.box_threshold,
        text_threshold=cfg.text_threshold,
    )

    semantic_masks = []
    for mask in sam_masks:
        y, x = np.where(mask["segmentation"])
        if len(x) == 0:
            continue
        sam_box = [x.min(), y.min(), x.max(), y.max()]

        if any(bbox_iou(sam_box, db) >= cfg.iou_threshold for db in dino_boxes):
            semantic_masks.append(mask)

    debug = {"sam_total": sam_total, "sam_filtered": len(sam_masks), "dino_boxes": len(dino_boxes)}
    return semantic_masks, dino_boxes, dino_scores, dino_labels, debug


def save_visualization(
    fp: str,
    img_rgb: np.ndarray,
    semantic_masks,
    dino_boxes: np.ndarray,
    dino_scores: np.ndarray,
    dino_labels,
    output_path: str,
):
    overlay = img_rgb.copy()
    np.random.seed(42)
    colors = np.random.rand(max(1, len(semantic_masks)), 3)

    plt.figure(figsize=(14, 8))

    for i, mask in enumerate(semantic_masks):
        m = mask["segmentation"].astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = (colors[i] * 255).astype(np.uint8).tolist()
        cv2.drawContours(overlay, contours, -1, color=color, thickness=2)

        y, x = np.where(m)
        if len(x) > 0:
            cx, cy = int(np.mean(x)), int(np.mean(y))
            plt.text(
                cx,
                cy,
                f"{i + 1}",
                color="white",
                fontsize=10,
                bbox=dict(facecolor=tuple(c / 255 for c in color), alpha=0.8, edgecolor="none"),
            )

    for j, box in enumerate(dino_boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 2)
        score = dino_scores[j] if j < len(dino_scores) else 0
        label = dino_labels[j] if j < len(dino_labels) else "box"
        cv2.putText(
            overlay,
            f"{label}:{score:.2f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
        )

    viz = cv2.addWeighted(overlay, 0.6, img_rgb, 0.4, 0)
    plt.imshow(viz)
    plt.title(f"{os.path.basename(fp)} | Cajas detectadas: {len(semantic_masks)}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main(cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)
    frames_dir = os.path.join(cfg.output_dir, "frames")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando device: {device}")

    if not os.path.exists(cfg.video_path):
        raise FileNotFoundError(f"No existe video: {cfg.video_path}")
    if not os.path.exists(cfg.sam_checkpoint):
        raise FileNotFoundError(f"No existe checkpoint SAM: {cfg.sam_checkpoint}")

    frame_paths = extract_frames(
        cfg.video_path,
        frames_dir,
        cfg.max_frames,
        cfg.resize_w,
        cfg.resize_h,
        cfg.frame_stride,
    )

    sam = sam_model_registry[cfg.model_type](checkpoint=cfg.sam_checkpoint)
    sam.to(device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=70,
        pred_iou_thresh=0.82,
        stability_score_thresh=0.95,
        min_mask_region_area=cfg.min_mask_region_area,
    )

    processor = AutoProcessor.from_pretrained(cfg.dino_model_id)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(cfg.dino_model_id).to(device)

    conteos_por_frame = []
    resultados = []

    for fp in sorted(glob.glob(os.path.join(frames_dir, "*.jpg"))):
        img_bgr = cv2.imread(fp)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        semantic_masks, dino_boxes, dino_scores, dino_labels, debug = sam_dino_boxes_for_frame(
            img_rgb,
            mask_generator,
            processor,
            dino_model,
            device,
            cfg,
        )
        conteos_por_frame.append(len(semantic_masks))
        resultados.append((fp, img_rgb, semantic_masks, dino_boxes, dino_scores, dino_labels))
        print(f"{os.path.basename(fp)} | SAM_total={debug['sam_total']} | SAM_filtradas={debug['sam_filtered']} | DINO_boxes={debug['dino_boxes']} | Final={len(semantic_masks)}")

    print("\nConteo estimado de cajas por frame:")
    for (fp, *_), c in zip(resultados, conteos_por_frame):
        print(f"{os.path.basename(fp)} -> {c}")

    promedio = round(float(np.mean(conteos_por_frame)), 2) if conteos_por_frame else 0.0
    print(f"Promedio de cajas detectadas: {promedio}")

    if resultados:
        vis_dir = os.path.join(cfg.output_dir, "visualizaciones")
        os.makedirs(vis_dir, exist_ok=True)
        vis_paths = []
        for i, r in enumerate(resultados):
            vis_path = os.path.join(vis_dir, f"vis_{i:03d}.png")
            save_visualization(*r, output_path=vis_path)
            vis_paths.append(vis_path)
        print(f"Visualizaciones guardadas en: {vis_dir}")

        if cfg.save_video and vis_paths:
            first = cv2.imread(vis_paths[0])
            h, w = first.shape[:2]
            video_out = os.path.join(cfg.output_dir, "visualizacion_sam_dino.mp4")
            writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
            for pth in vis_paths:
                frame = cv2.imread(pth)
                writer.write(frame)
            writer.release()
            print(f"Video de visualización guardado en: {video_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conteo de cajas con SAM + Grounding DINO")
    parser.add_argument("--video_path", required=True, help="Ruta al video de entrada")
    parser.add_argument("--sam_checkpoint", required=True, help="Ruta al checkpoint sam_vit_b.pth")
    parser.add_argument("--output_dir", default="./outputs", help="Directorio de salida")
    parser.add_argument("--model_type", default="vit_b", help="Tipo de SAM: vit_b, vit_l, vit_h")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--resize_w", type=int, default=640)
    parser.add_argument("--resize_h", type=int, default=360)
    parser.add_argument("--min_mask_region_area", type=int, default=1000)
    parser.add_argument("--max_mask_region_area", type=int, default=50000)
    parser.add_argument("--max_mask_area_ratio", type=float, default=0.40)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--text_prompt", default="cardboard box . carton box . pallet box .")
    parser.add_argument("--box_threshold", type=float, default=0.20)
    parser.add_argument("--text_threshold", type=float, default=0.15)
    parser.add_argument("--iou_threshold", type=float, default=0.05)
    parser.add_argument("--dino_model_id", default="IDEA-Research/grounding-dino-base")

    args = parser.parse_args()

    cfg = Config(
        video_path=args.video_path,
        sam_checkpoint=args.sam_checkpoint,
        output_dir=args.output_dir,
        model_type=args.model_type,
        max_frames=args.max_frames,
        resize_w=args.resize_w,
        resize_h=args.resize_h,
        min_mask_region_area=args.min_mask_region_area,
        max_mask_region_area=args.max_mask_region_area,
        text_prompt=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        iou_threshold=args.iou_threshold,
        dino_model_id=args.dino_model_id,
        max_mask_area_ratio=args.max_mask_area_ratio,
        frame_stride=args.frame_stride,
        save_video=args.save_video,
    )

    main(cfg)
