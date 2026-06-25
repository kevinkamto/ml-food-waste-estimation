"""
Automatic segmentation of raw food plate images into the ground-truth format:
  black background | pure white plate area | food at original colors

Two-stage algorithm:
  Stage 1 -- SAM ViT-B: center-point + corner-background prompts identify the plate region
  Stage 2 -- HSV + texture within mask: plate surface -> white, food -> original colors
"""

import argparse
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm

_OUT_SIZE = 800
_CLOSE_KERNEL = 30  # morphological closing after SAM mask to fill small gaps
_PLATE_SAT_MAX = 40
_PLATE_VAL_MIN = 150
_PLATE_TEX_MAX = 12

_SAM_MODEL_TYPE = "vit_b"
_SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
_SAM_CHECKPOINT_PATH = Path.home() / ".cache" / "segment_anything" / "sam_vit_b_01ec64.pth"

_sam_predictor: SamPredictor | None = None


def _get_sam_predictor() -> SamPredictor:
    """Lazy-load SAM ViT-B predictor, downloading the checkpoint on first use (~375 MB)."""
    global _sam_predictor
    if _sam_predictor is None:
        if not _SAM_CHECKPOINT_PATH.exists():
            _SAM_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading SAM checkpoint (~375 MB) to {_SAM_CHECKPOINT_PATH} ...")
            urllib.request.urlretrieve(_SAM_CHECKPOINT_URL, str(_SAM_CHECKPOINT_PATH))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[_SAM_MODEL_TYPE](checkpoint=str(_SAM_CHECKPOINT_PATH))
        sam.to(device)
        sam.eval()
        _sam_predictor = SamPredictor(sam)
    return _sam_predictor


def segment_image(input_path: str, output_path: str | None = None) -> Image.Image:
    """
    Segment a raw food plate image into the standard 3-region format.

    Returns an 800x800 RGB PIL Image:
      - background (outside plate) -> black (0, 0, 0)
      - plate surface (inside mask, low saturation / high brightness) -> white (255, 255, 255)
      - food (inside mask, colored or textured) -> original pixel colors

    Args:
        input_path: Path to the raw input image (JPG/PNG).
        output_path: If provided, saves the result to this path.
    """
    bgr = cv2.imread(input_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    bgr_sq = _pad_to_square(bgr)
    bgr_800 = cv2.resize(bgr_sq, (_OUT_SIZE, _OUT_SIZE), interpolation=cv2.INTER_LINEAR)

    plate_mask = _detect_plate_sam(bgr_800)
    result_bgr = _normalize_plate(bgr_800, plate_mask)

    rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    out = Image.fromarray(rgb)
    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        out.save(output_path)
    return out


def _pad_to_square(bgr: np.ndarray) -> np.ndarray:
    """Pad the shorter dimension with black so the image becomes square."""
    h, w = bgr.shape[:2]
    if h == w:
        return bgr
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=bgr.dtype)
    top = (side - h) // 2
    left = (side - w) // 2
    canvas[top : top + h, left : left + w] = bgr
    return canvas


def _detect_plate_sam(bgr: np.ndarray) -> np.ndarray:
    """
    Stage 1: SAM ViT-B with center-point + corner-background prompts.

    Foreground prompt: center of frame (plate is always centered, fixed camera).
    Background prompts: four corners (always the checkered mat background).
    Returns a binary uint8 mask (255 = plate+food, 0 = background), same HxW as bgr.
    """
    h, w = bgr.shape[:2]
    margin = min(h, w) // 10

    point_coords = np.array(
        [
            [w // 2, h // 2],  # center = foreground (plate)
            [margin, margin],  # top-left = background
            [w - margin, margin],  # top-right = background
            [margin, h - margin],  # bottom-left = background
            [w - margin, h - margin],  # bottom-right = background
        ]
    )
    point_labels = np.array([1, 0, 0, 0, 0])

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    predictor = _get_sam_predictor()
    predictor.set_image(rgb)

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    best = masks[int(np.argmax(scores))].astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_CLOSE_KERNEL, _CLOSE_KERNEL))
    return np.asarray(cv2.morphologyEx(best, cv2.MORPH_CLOSE, kernel), dtype=np.uint8)


def _normalize_plate(bgr: np.ndarray, plate_mask: np.ndarray) -> np.ndarray:
    """
    Stage 2: within the plate mask, classify pixels by HSV + local texture.

    Plate surface: S < _PLATE_SAT_MAX AND V > _PLATE_VAL_MIN AND texture_std < _PLATE_TEX_MAX
    Food:          everything else inside the mask -> keep original BGR
    Outside mask:  (0, 0, 0)

    Texture check is required to preserve white/cream foods (rice, porridge -- ~29% of dataset).
    Rice grains have local std ~10-20; smooth plate surface has local std ~3-8.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (7, 7))
    mean_sq = cv2.blur(gray**2, (7, 7))
    texture_std = np.sqrt(np.clip(mean_sq - mean**2, 0, None))

    inside = plate_mask == 255
    plate_surface = (
        inside & (s < _PLATE_SAT_MAX) & (v > _PLATE_VAL_MIN) & (texture_std < _PLATE_TEX_MAX)
    )

    result = np.zeros_like(bgr)
    result[inside] = bgr[inside]
    result[plate_surface] = (255, 255, 255)
    return result


def _batch_segment(input_dir: str, output_dir: str) -> None:
    """Walk input_dir recursively and segment every JPG/PNG into output_dir (flat)."""
    exts = {".jpg", ".jpeg", ".png"}
    paths = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(root, f))

    os.makedirs(output_dir, exist_ok=True)
    for src in tqdm(paths, desc="Segmenting"):
        dst = os.path.join(output_dir, os.path.basename(src))
        try:
            segment_image(src, dst)
        except Exception as exc:
            print(f"WARN: skipping {src} -- {exc}")


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Segment raw food plate images to the standard 3-region format."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", help="Single input image path")
    mode.add_argument(
        "--input_dir", help="Batch mode: directory of raw images (walked recursively)"
    )
    parser.add_argument("--output", help="Single output image path (required with --input)")
    parser.add_argument(
        "--output_dir",
        help="Batch mode: directory to write segmented images (flat)",
    )
    args = parser.parse_args()

    if args.input:
        if not args.output:
            parser.error("--output is required when using --input")
        img = segment_image(args.input, args.output)
        print(f"Saved {args.output}  ({img.size[0]}x{img.size[1]})")
    else:
        if not args.output_dir:
            parser.error("--output_dir is required when using --input_dir")
        _batch_segment(args.input_dir, args.output_dir)


if __name__ == "__main__":
    _cli()
