"""One-off script to render assets/readme/architecture.png, a figure of the
dual-stream EfficientNet-B0 model. Not part of the training/inference pipeline;
kept for reproducing the README asset.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_PATH = Path(__file__).parent / "architecture.png"

BLUE = "#3B6FA0"
GREEN = "#4C8C5A"
ORANGE = "#C77B2C"
GRAY = "#5A5A5A"
LIGHT = "#F2F2F2"


def box(ax, xy, w, h, text, color, fontsize=10, fontcolor="white"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2,
        edgecolor=color,
        facecolor=color,
        alpha=0.92,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=fontcolor, fontweight="bold", wrap=True)
    return patch


def arrow(ax, start, end, color=GRAY):
    a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                         linewidth=1.4, color=color)
    ax.add_patch(a)


fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis("off")

# Inputs
box(ax, (0.3, 6.2), 2.2, 0.9, "Segmented\nBefore Image\n(3, 224, 224)", BLUE)
box(ax, (0.3, 1.0), 2.2, 0.9, "Segmented\nAfter Image\n(3, 224, 224)", BLUE)

# Shared backbone
box(ax, (3.1, 5.9), 2.4, 1.5, "EfficientNet-B0\n(shared weights)", ORANGE, fontsize=10)
box(ax, (3.1, 0.7), 2.4, 1.5, "EfficientNet-B0\n(shared weights)", ORANGE, fontsize=10)
ax.text(4.3, 4.0, "Siamese stream\n(same weights)", ha="center", va="center",
        fontsize=8.5, color=GRAY, style="italic")

arrow(ax, (2.5, 6.65), (3.1, 6.65))
arrow(ax, (2.5, 1.45), (3.1, 1.45))

# Features
box(ax, (6.1, 5.9), 1.9, 0.9, "feat_before\n(1280,)", GREEN, fontsize=9)
box(ax, (6.1, 1.3), 1.9, 0.9, "feat_after\n(1280,)", GREEN, fontsize=9)
arrow(ax, (5.5, 6.65), (6.1, 6.35))
arrow(ax, (5.5, 1.45), (6.1, 1.75))

# area ratio
box(ax, (6.1, 3.55), 1.9, 0.7, "area_ratio\n(scalar)", "#8A5BAE", fontsize=9)
ax.text(5.0, 4.55, "non-black px(after)\n/ non-black px(before)", ha="center", va="center",
        fontsize=7.5, color=GRAY, style="italic")

# Fusion
box(ax, (8.5, 3.0), 1.3, 2.0,
    "Fuse:\nconcat(\nfeat_before,\nfeat_after,\n|diff|,\narea_ratio)\n(3841,)",
    GRAY, fontsize=8)
arrow(ax, (8.0, 6.3), (8.5, 4.6))
arrow(ax, (8.0, 1.7), (8.5, 3.4))
arrow(ax, (8.0, 3.9), (8.5, 3.9))

# Regression head (separate row below, full width)
box(ax, (1.5, -0.65), 8.7, 1.1,
    "Regression Head\nFC(3841->1024) -> FC(1024->512) -> FC(512->1) -> clamp(0, 1)",
    "#7A3B3B", fontsize=10)
arrow(ax, (9.15, 3.0), (8.0, 0.45))

box(ax, (3.5, -2.05), 5.0, 0.85, "consumption_ratio  r  in [0, 1]", "#1F1F1F", fontsize=10.5)
arrow(ax, (5.85, -0.65), (5.85, -1.2))

ax.set_title("Dual-Stream EfficientNet-B0 -- Single-Task Regression Architecture",
              fontsize=13, fontweight="bold", pad=14)
ax.set_xlim(0, 10.2)
ax.set_ylim(-2.5, 8)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150)
print(f"saved {OUT_PATH}")
