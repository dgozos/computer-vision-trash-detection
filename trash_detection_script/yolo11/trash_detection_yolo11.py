"""
This script is a YOLO11-based sidewalk trash detection and disposal classification pipeline for an autonomous ground robot. 
Detects 6 waste categories (cardboard, glass, metal, paper, plastic, trash) and classifies each as recyclable or landfill. 
Trained via transfer learning from COCO pretrained weights on the Raw-Images-AllTrash dataset, achieving 0.917 mAP@50.
"""


# ── 1. Setup & Dependencies ────────────────────────────────────────────────────────────────────────────────────

# pip install ultralytics matplotlib seaborn scikit-learn pillow pandas pyyaml --quiet

import os
import time
import random
import json
from pathlib import Path
from collections import Counter, defaultdict
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image
import yaml

import torch
from ultralytics import YOLO

# Add the Google Drive path to sys.path to import utils.py
sys.path.append('/content/drive/My Drive/')
import trash_detection_script.yolo11.utils as utils   # course utils.py

device = utils.device
MEAN   = utils.MEAN
STD    = utils.STD
print(f'Device : {device}')
print(f'ImageNet norm -- mean={MEAN}, std={STD}')

# ── Configuration ──────────────────────────────────────────
# Update RAW_DATA_DIR to point to your extracted Kaggle dataset
# Expected structure:
#   trash-detection/
#     train/  images/  labels/
#     valid/  images/  labels/
#     test/   images/  labels/

RAW_DATA_DIR   = '/content/drive/My Drive/trash-detection'   # <-- UPDATE THIS
OUTPUT_DIR     = 'runs/trash_det'

RANDOM_SEED    = 42
IMG_SIZE       = 224       # detection standard - FURTHER REDUCED TO 224
BATCH_SIZE     = 1         # lower than cls -- detection uses more VRAM - REDUCED TO 1
EPOCHS         = 50
DEVICE         = 0 if torch.cuda.is_available() else 'cpu'

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

os.makedirs(os.path.join(OUTPUT_DIR, 'plots'), exist_ok=True)
print('Config ready.')

# ── 2. Dataset Exploration & data.yaml Creation ──────────────────────────────────────────

# ── Verify dataset structure ──────────────────────────────

dataset_path = Path(RAW_DATA_DIR)

for split in ['train', 'valid', 'test']:
    img_dir = dataset_path / split / 'images'
    lbl_dir = dataset_path / split / 'labels'

    n_imgs = len(list(img_dir.glob('*'))) if img_dir.exists() else 0
    n_lbls = len(list(lbl_dir.glob('*.txt'))) if lbl_dir.exists() else 0

    status = '✓' if n_imgs > 0 and n_lbls > 0 else '✗'
    print(f'  {status} {split:>5s}:  {n_imgs:>5d} images  |  {n_lbls:>5d} labels')

    if n_imgs != n_lbls:
        print(f'           ⚠ mismatch: {abs(n_imgs - n_lbls)} images without labels')

# ── Discover classes from label files ─────────────────────

label_dir = dataset_path / 'train' / 'labels'
class_ids = set()
total_boxes = 0
boxes_per_class = Counter()

for lbl_file in label_dir.glob('*.txt'):
    for line in lbl_file.read_text().strip().splitlines():
        if line.strip():
            cls_id = int(line.split()[0])
            class_ids.add(cls_id)
            boxes_per_class[cls_id] += 1
            total_boxes += 1

print(f'Class IDs found : {sorted(class_ids)}')
print(f'Total boxes     : {total_boxes:,}')
print(f'\nBoxes per class (training set):')
for cls_id in sorted(class_ids):
    count = boxes_per_class[cls_id]
    pct = count / total_boxes * 100
    bar = '█' * int(pct)
    print(f'  class {cls_id}: {count:>6,} ({pct:5.1f}%)  {bar}')

# ── Create data.yaml programmatically ─────────────────────
# Update CLASS_NAMES to match your dataset's actual class mapping.
# Run the cell above first to see which class IDs exist.

CLASS_NAMES = {
    0: 'cardboard',
    1: 'glass',
    2: 'metal',
    3: 'paper',
    4: 'plastic',
    5: 'trash',
}

data_config = {
    'path':  RAW_DATA_DIR, # Changed to use RAW_DATA_DIR directly
    'train': 'train/images',
    'val':   'valid/images',
    'test':  'test/images',
    'names': {i: CLASS_NAMES[i] for i in sorted(class_ids)},
}

yaml_path = Path(RAW_DATA_DIR) / 'data.yaml' # Ensure yaml_path uses RAW_DATA_DIR directly
with open(yaml_path, 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

print(f'Written: {yaml_path}\n')
print(open(yaml_path).read())

# Store for downstream cells
NUM_CLASSES = len(data_config['names'])
classes = [data_config['names'][i] for i in sorted(data_config['names'].keys())]

# ── Preview: sample images with ground-truth boxes ────────

def plot_image_with_boxes(img_path, lbl_path, class_names, ax):
    """Draw an image with its YOLO-format bounding boxes."""
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    ax.imshow(img)

    colors = plt.cm.Set2(np.linspace(0, 1, len(class_names)))

    if lbl_path.exists():
        for line in lbl_path.read_text().strip().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            cls_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            # Convert YOLO (center x, center y, w, h) normalized -> pixel coords
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            box_w = bw * w
            box_h = bh * h

            color = colors[cls_id % len(colors)]
            rect = patches.Rectangle((x1, y1), box_w, box_h,
                                     linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 4, class_names[cls_id],
                    color='white', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))
    ax.axis('off')


# Pick random training images
train_imgs = list((dataset_path / 'train' / 'images').glob('*'))
sampled = random.sample(train_imgs, min(8, len(train_imgs)))

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Training Samples with Ground-Truth Boxes', fontsize=16, fontweight='bold')

for ax, img_path in zip(axes.flatten(), sampled):
    lbl_path = dataset_path / 'train' / 'labels' / (img_path.stem + '.txt')
    plot_image_with_boxes(img_path, lbl_path, classes, ax)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/ground_truth_samples.png', dpi=150, bbox_inches='tight')
plt.show()


# ── Dataset distribution visualization ────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Bounding Box Distribution by Split', fontsize=16, fontweight='bold')
palette = sns.color_palette('Set2', n_colors=NUM_CLASSES)

for ax, split in zip(axes, ['train', 'valid', 'test']):
    lbl_dir = dataset_path / split / 'labels'
    counts = Counter()
    for lbl_file in lbl_dir.glob('*.txt'):
        for line in lbl_file.read_text().strip().splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1

    names = [classes[i] for i in sorted(counts.keys())]
    values = [counts[i] for i in sorted(counts.keys())]
    total = sum(values)

    bars = ax.barh(names, values, color=palette[:len(names)], edgecolor='white')
    ax.set_title(f'{split.upper()} ({total:,} boxes)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Number of Boxes')
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:,}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/dataset_distribution.png', dpi=150, bbox_inches='tight')
plt.show()


# ── Bounding box size distribution ────────────────────────
# Useful to understand if you have tiny objects the model might miss

widths = []
heights = []
cls_labels = []

for lbl_file in (dataset_path / 'train' / 'labels').glob('*.txt'):
    for line in lbl_file.read_text().strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        cls_labels.append(classes[int(parts[0])])
        widths.append(float(parts[3]))    # normalized width
        heights.append(float(parts[4]))   # normalized height

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Bounding Box Size Distribution (Training Set)', fontsize=16, fontweight='bold')

# Width x Height scatter
axes[0].scatter(widths, heights, alpha=0.15, s=8, c='steelblue')
axes[0].set_xlabel('Normalized Width')
axes[0].set_ylabel('Normalized Height')
axes[0].set_title('Box Width vs Height')
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)
axes[0].grid(alpha=0.3)

# Area distribution per class
areas = [w * h for w, h in zip(widths, heights)]
df_boxes = pd.DataFrame({'class': cls_labels, 'area': areas})
df_boxes.boxplot(column='area', by='class', ax=axes[1], grid=False, rot=30)
axes[1].set_title('Box Area by Class')
axes[1].set_xlabel('')
axes[1].set_ylabel('Normalized Area')
plt.suptitle('')  # remove auto-title from boxplot

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/bbox_size_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 3. Model Architecture Inspection ──────────────────────────────────────────

# ── Load YOLO11n with COCO pretrained weights ─────────────
# .pt  = COCO pretrained weights  --> this is transfer learning
# .yaml = architecture only, random init --> training from scratch

yolo_pretrained = YOLO('yolo11n.pt')   # <-- .pt means COCO pretrained (detection)

total_params_pre = sum(p.numel() for p in yolo_pretrained.model.parameters())
trainable_params = sum(p.numel() for p in yolo_pretrained.model.parameters() if p.requires_grad)

print(f'YOLO11n PRETRAINED (COCO) -- Detection')
print(f'  Total params     : {total_params_pre:,}')
print(f'  Trainable params : {trainable_params:,}')
print(f'  Architecture is identical -- only initialization differs.')

# ── Print full model architecture ─────────────────────────
# Shows backbone (C3k2, SPPF, C2PSA), neck (FPN + PAN), and detection head

yolo_pretrained.info(verbose=True, detailed=True)

# ── (Optional) Compare: from-scratch model ────────────────
# Uncomment to also train from random initialization

# yolo_scratch = YOLO('yolo11n.yaml')   # <-- .yaml = random init
# total_params_scratch = sum(p.numel() for p in yolo_scratch.model.parameters())
# print(f'YOLO11n FROM SCRATCH')
# print(f'  Total params     : {total_params_scratch:,}')
# print(f'  Same architecture, random weights.')

# ── 4. Pre-training Sanity Checks  ──────────────────────────────────────────

# ── Pre-training sanity check ─────────────────────────────
print(f'yaml_path exists : {os.path.exists(yaml_path)}')
print(f'yaml_path        : {yaml_path}')
print()

# Check all image dirs are reachable
for split in ['train', 'valid', 'test']:
    img_dir = dataset_path / split / 'images'
    n = len(list(img_dir.glob('*'))) if img_dir.exists() else 0
    print(f'  {split:>5s}/images : {n} files  {"✓" if n > 0 else "✗ NOT FOUND"}')

required = {
    'yolo_pretrained': 'YOLO model object',
    'yaml_path':       'path to data.yaml',
    'OUTPUT_DIR':      'output directory',
    'EPOCHS':          'epoch count',
    'IMG_SIZE':        'image size',
    'BATCH_SIZE':      'batch size',
    'DEVICE':          'device',
}

all_ok = True
for var, desc in required.items():
    exists = var in dir() or var in locals() or var in globals()
    status = '✓' if exists else '✗ MISSING — re-run its cell'
    if not exists:
        all_ok = False
    print(f'  {status}  {var:>20s}  ({desc})')

print(f'\n{"All variables ready — safe to train." if all_ok else "Re-run missing cells before training."}')

# ── Verify YOLO can read the yaml before training ─────────
import yaml

with open(yaml_path) as f:
    cfg = yaml.safe_load(f)

print('data.yaml contents:')
print(json.dumps(cfg, indent=2))

# Confirm absolute paths resolve
for split in ['train', 'val', 'test']:
    full = Path(cfg['path']) / cfg[split]
    print(f'\n  {split}: {full}')
    print(f'         exists: {full.exists()}')

# ── 5. Training ──────────────────────────────────────────

# ── Train YOLO11n detection (pretrained / transfer learning)

print(f'Fine-tuning for {EPOCHS} epochs...')
print(f'  data.yaml : {yaml_path}')
print(f'  ImgSize   : {IMG_SIZE}')
print(f'  Batch     : {BATCH_SIZE}')
print(f'  Device    : {DEVICE}')
print(f'  Patience  : 10 (early stopping)\n')

start = time.time()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

yolo_pretrained.train(
    data     = str(yaml_path),
    epochs   = EPOCHS,
    imgsz    = IMG_SIZE,
    batch    = BATCH_SIZE,
    device   = DEVICE,
    project  = OUTPUT_DIR,
    name     = 'yolo11n_6cls',
    patience = 10,
    workers  = 0,
    cache    = False,
    save     = True,
    plots    = True,
    verbose  = True,
    exist_ok = True,
)

time_pretrained = time.time() - start
print(f'\nPretrained fine-tuning complete in {time_pretrained/60:.1f} min')

# ── Locate best weights ───────────────────────────────────

TRAIN_DIR    = os.path.join(OUTPUT_DIR, 'yolo11n_6cls')
BEST_WEIGHTS = os.path.join(TRAIN_DIR, 'weights', 'best.pt')
LAST_WEIGHTS = os.path.join(TRAIN_DIR, 'weights', 'last.pt')

# Diagnose what's in the training directory
train_path = Path(TRAIN_DIR)
if not train_path.exists():
    print(f'✗ Training directory not found: {TRAIN_DIR}')
    print(f'  Training likely crashed before saving anything.')
    print(f'  → Re-run the training cell with workers=0, cache=False')
else:
    print(f'✓ Training directory found: {TRAIN_DIR}')
    print(f'\n  Contents:')
    for f in sorted(train_path.rglob('*')):
        if f.is_file():
            print(f'    {f.relative_to(train_path)}  ({f.stat().st_size/1e6:.2f} MB)')

    if os.path.exists(BEST_WEIGHTS):
        print(f'\n✓ Best weights: {BEST_WEIGHTS}')
        print(f'  File size   : {os.path.getsize(BEST_WEIGHTS)/1e6:.1f} MB')
    elif os.path.exists(LAST_WEIGHTS):
        BEST_WEIGHTS = LAST_WEIGHTS
        print(f'\n⚠ best.pt not found, using last.pt: {BEST_WEIGHTS}')
        print(f'  File size   : {os.path.getsize(BEST_WEIGHTS)/1e6:.1f} MB')
    else:
        print(f'\n✗ No weights found — training did not save.')
        print(f'  → Re-run the training cell with workers=0, cache=False')

# ── 6. Restore Session if Kernel Dies ──────────────────────────────────────────

# # ── Restore session after kernel restart ──────────────────

# import os, time, random, json
# from pathlib import Path
# from collections import Counter, defaultdict

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import seaborn as sns
# from PIL import Image
# import yaml
# import torch
# from ultralytics import YOLO

# # ── Restore all variables ──────────────────────────────────
# RAW_DATA_DIR = '/home1/gozosd9005@cgu.edu/Dataset/raw-images-alltrash'
# OUTPUT_DIR   = '/home1/gozosd9005@cgu.edu/runs/detect/runs/trash_det'
# TRAIN_DIR    = os.path.join(OUTPUT_DIR, 'yolo11n_pretrained')
# BEST_WEIGHTS = os.path.join(TRAIN_DIR, 'weights', 'best.pt')

# IMG_SIZE     = 640
# BATCH_SIZE   = 32
# DEVICE       = 0 if torch.cuda.is_available() else 'cpu'
# RANDOM_SEED  = 42

# dataset_path = Path(RAW_DATA_DIR)
# yaml_path    = dataset_path / 'data.yaml'

# CLASS_NAMES  = {0:'cardboard', 1:'glass', 2:'metal', 3:'paper', 4:'plastic', 5:'trash'}
# NUM_CLASSES  = len(CLASS_NAMES)
# classes      = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
# palette      = sns.color_palette('Set2', n_colors=NUM_CLASSES)

# random.seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)
# os.makedirs(os.path.join(OUTPUT_DIR, 'plots'), exist_ok=True)

# # ── Verify weights exist ───────────────────────────────────
# if os.path.exists(BEST_WEIGHTS):
#     print(f'✓ YOLO imported')
#     print(f'✓ Best weights : {BEST_WEIGHTS}')
#     print(f'  File size    : {os.path.getsize(BEST_WEIGHTS)/1e6:.1f} MB')
#     print(f'✓ Session restored — safe to continue from validation.')
# else:
#     print(f'✗ Weights not found at {BEST_WEIGHTS}')

# ── 7. Evaluation & Key Metrics ──────────────────────────────────────────

# ── Run YOLO's built-in validation on test set ────────────
# This computes mAP@50, mAP@50:95, precision, recall per class

model = YOLO(BEST_WEIGHTS)

start = time.time()
val_results = model.val(
    data   = str(yaml_path),
    split  = 'test',
    imgsz  = IMG_SIZE,
    batch  = BATCH_SIZE,
    device = DEVICE,
    workers   = 0,           # <-- fixes kernel crash
    cache     = False,       # <-- prevents cache freeze
    plots  = True,
    save_json = True,
)
time_val = time.time() - start

print(f'\nValidation complete in {time_val:.1f}s')

# ── Key metrics summary ───────────────────────────────────

# Overall metrics
map50    = val_results.box.map50       # mAP @ IoU=0.50
map50_95 = val_results.box.map         # mAP @ IoU=0.50:0.95
mp       = val_results.box.mp          # mean precision
mr       = val_results.box.mr          # mean recall

print(f'╔══════════════════════════════════════╗')
print(f'║       TEST SET — KEY METRICS         ║')
print(f'╠══════════════════════════════════════╣')
print(f'║  mAP@50       : {map50:.4f}              ║')
print(f'║  mAP@50:95    : {map50_95:.4f}              ║')
print(f'║  Precision    : {mp:.4f}              ║')
print(f'║  Recall       : {mr:.4f}              ║')
print(f'╚══════════════════════════════════════╝')

# ── Per-class AP breakdown ────────────────────────────────

per_class_ap50 = val_results.box.ap50       # shape: (num_classes,)
per_class_ap   = val_results.box.ap         # mAP@50:95 per class

metrics_df = pd.DataFrame({
    'Class':       classes,
    'AP@50':       per_class_ap50,
    'AP@50:95':    per_class_ap,
})

# Add overall row
metrics_df.loc[len(metrics_df)] = ['ALL (mean)', map50, map50_95]

print(metrics_df.to_string(index=False, float_format='{:.4f}'.format))

metrics_df.style.format({'AP@50': '{:.4f}', 'AP@50:95': '{:.4f}'}) \
    .background_gradient(subset=['AP@50'], cmap='RdYlGn', vmin=0, vmax=1) \
    .background_gradient(subset=['AP@50:95'], cmap='RdYlGn', vmin=0, vmax=1)

# ── Run inference on test images for detailed analysis ────
# Collect per-image predictions for error analysis downstream

test_img_dir = dataset_path / 'test' / 'images'
test_lbl_dir = dataset_path / 'test' / 'labels'
test_images  = sorted(list(test_img_dir.glob('*')))

all_predictions = []   # list of dicts per image

print(f'Running inference on {len(test_images)} test images...\n')
start = time.time()

for img_path in test_images:
    results = model.predict(source=str(img_path), imgsz=IMG_SIZE,
                            conf=0.25, verbose=False)
    r = results[0]

    # Predicted boxes
    pred_boxes = []
    if r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            pred_boxes.append({
                'class_id':  int(box.cls.item()),
                'class_name': classes[int(box.cls.item())],
                'confidence': float(box.conf.item()),
                'bbox_xyxy': box.xyxy[0].tolist(),
            })

    # Ground-truth boxes
    gt_boxes = []
    lbl_path = test_lbl_dir / (img_path.stem + '.txt')
    if lbl_path.exists():
        for line in lbl_path.read_text().strip().splitlines():
            if line.strip():
                parts = line.split()
                gt_boxes.append({
                    'class_id':   int(parts[0]),
                    'class_name': classes[int(parts[0])],
                })

    all_predictions.append({
        'image_path': str(img_path),
        'pred_boxes': pred_boxes,
        'gt_boxes':   gt_boxes,
        'n_pred':     len(pred_boxes),
        'n_gt':       len(gt_boxes),
    })

time_inf = time.time() - start
print(f'Done: {len(test_images)} images in {time_inf:.1f}s ({len(test_images)/time_inf:.1f} img/s)')

# ── Recyclable vs Landfill Classification ─────────────────
# Simple post-processing — no additional model needed

RECYCLE_MAP = {
    'cardboard': 'recyclable',
    'glass':     'recyclable',
    'metal':     'recyclable',
    'paper':     'recyclable',
    'plastic':   'recyclable',
    'trash':     'landfill',
}

# Apply to all predictions
for pred in all_predictions:
    for box in pred['pred_boxes']:
        box['disposal'] = RECYCLE_MAP[box['class_name']]

# Summary
recycle_count = sum(1 for p in all_predictions for b in p['pred_boxes'] if b['disposal'] == 'recyclable')
landfill_count = sum(1 for p in all_predictions for b in p['pred_boxes'] if b['disposal'] == 'landfill')

print(f'Disposal Classification (test set)\n')
print(f'  Recyclable : {recycle_count}')
print(f'  Landfill   : {landfill_count}')
print(f'  Total      : {recycle_count + landfill_count}')
print(f'\nMapping:')
for cls, disposal in RECYCLE_MAP.items():
    print(f'  {cls:>12s} → {disposal}')

# ── 8. Visualizations ──────────────────────────────────────────

# ── 8a. Per-Class AP Bar Chart ────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5))

x = np.arange(len(classes))
width = 0.35

bars1 = ax.bar(x - width/2, per_class_ap50, width, label='AP@50', color='#2196F3')
bars2 = ax.bar(x + width/2, per_class_ap, width, label='AP@50:95', color='#FF9800')

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f'{h:.2f}', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=30, ha='right')
ax.set_ylim(0, 1.15)
ax.set_ylabel('Average Precision')
ax.set_title('Per-Class Average Precision (Test Set)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=map50, color='#2196F3', linestyle='--', alpha=0.4, label=f'mAP@50={map50:.2f}')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/per_class_ap.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 8b. Confidence Distribution ───────────────────────────

all_confs = []
all_cls_names = []
for pred in all_predictions:
    for box in pred['pred_boxes']:
        all_confs.append(box['confidence'])
        all_cls_names.append(box['class_name'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Prediction Confidence Distribution', fontsize=16, fontweight='bold')

# Overall histogram
axes[0].hist(all_confs, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('Confidence Score')
axes[0].set_ylabel('Number of Predictions')
axes[0].set_title('Overall Confidence Distribution')
axes[0].axvline(x=0.5, color='red', linestyle='--', alpha=0.6, label='0.5 threshold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Per-class mean confidence
conf_by_class = defaultdict(list)
for conf, cls_name in zip(all_confs, all_cls_names):
    conf_by_class[cls_name].append(conf)

cls_mean_conf = {cls: np.mean(confs) for cls, confs in conf_by_class.items()}
cls_names_sorted = [c for c in classes if c in cls_mean_conf]
mean_vals = [cls_mean_conf[c] for c in cls_names_sorted]

bars = axes[1].barh(cls_names_sorted, mean_vals, color=palette[:len(cls_names_sorted)])
for bar, val in zip(bars, mean_vals):
    axes[1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.2f}', va='center', fontsize=10)
axes[1].set_xlabel('Mean Confidence')
axes[1].set_title('Mean Confidence per Class')
axes[1].set_xlim(0, 1.1)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/confidence_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 8c. Training Curves (loss, mAP, LR) ──────────────────

csv_path = os.path.join(TRAIN_DIR, 'results.csv')

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Training Curves', fontsize=16, fontweight='bold')

    # Box loss
    loss_cols = [c for c in df.columns if 'loss' in c.lower()]
    for col in loss_cols:
        axes[0, 0].plot(df['epoch'], df[col], label=col, linewidth=1.5)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Losses')
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(alpha=0.3)

    # mAP
    map_cols = [c for c in df.columns if 'map' in c.lower() or 'mAP' in c]
    for col in map_cols:
        axes[0, 1].plot(df['epoch'], df[col], label=col, linewidth=1.5)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('mAP')
    axes[0, 1].set_title('mAP Metrics')
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(alpha=0.3)

    # Precision & Recall
    pr_cols = [c for c in df.columns if 'precision' in c.lower() or 'recall' in c.lower()]
    for col in pr_cols:
        axes[1, 0].plot(df['epoch'], df[col], label=col, linewidth=1.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision & Recall')
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].grid(alpha=0.3)

    # Learning rate
    lr_cols = [c for c in df.columns if 'lr' in c.lower()]
    for col in lr_cols:
        axes[1, 1].plot(df['epoch'], df[col], label=col, linewidth=1.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print(f'Training CSV not found at {csv_path}')

# ── 8d. Sample Predictions on Test Images ────────────────

sampled_test = random.sample(test_images, min(8, len(test_images)))

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Model Predictions on Test Images', fontsize=16, fontweight='bold')
colors_det = plt.cm.Set2(np.linspace(0, 1, NUM_CLASSES))

for ax, img_path in zip(axes.flatten(), sampled_test):
    img = Image.open(img_path).convert('RGB')
    results = model.predict(source=str(img_path), imgsz=IMG_SIZE,
                            conf=0.25, verbose=False)
    r = results[0]
    ax.imshow(img)

    if r.boxes is not None:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls.item())
            conf   = float(box.conf.item())
            color  = colors_det[cls_id % len(colors_det)]

            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 4, f'{classes[cls_id]} {conf:.2f}',
                    color='white', fontsize=7, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))
    ax.axis('off')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/sample_predictions.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 9. Error Analysis ──────────────────────────────────────────

# ── Detection error breakdown ─────────────────────────────
# Categorize images by: missed detections, false positives, class confusion

missed_detections = []      # images where n_pred < n_gt
false_positives   = []      # images where n_pred > n_gt
class_confusions  = []      # images where predicted class != ground truth class
clean_detections  = []      # images where n_pred == n_gt

for pred in all_predictions:
    n_gt   = pred['n_gt']
    n_pred = pred['n_pred']

    if n_pred < n_gt:
        missed_detections.append(pred)
    elif n_pred > n_gt:
        false_positives.append(pred)
    else:
        clean_detections.append(pred)

    # Check for class mismatches
    gt_classes  = Counter(b['class_name'] for b in pred['gt_boxes'])
    pred_classes = Counter(b['class_name'] for b in pred['pred_boxes'])
    if gt_classes != pred_classes:
        class_confusions.append(pred)

total = len(all_predictions)
print(f'Error Analysis Summary ({total} test images)\n')
print(f'  Missed detections (n_pred < n_gt) : {len(missed_detections):>4d} ({len(missed_detections)/total*100:.1f}%)')
print(f'  False positives   (n_pred > n_gt) : {len(false_positives):>4d} ({len(false_positives)/total*100:.1f}%)')
print(f'  Class confusions                  : {len(class_confusions):>4d} ({len(class_confusions)/total*100:.1f}%)')
print(f'  Clean detections  (n_pred == n_gt): {len(clean_detections):>4d} ({len(clean_detections)/total*100:.1f}%)')

# ── Most common class confusions ──────────────────────────

confusion_pairs = Counter()
for pred in class_confusions:
    gt_set   = [b['class_name'] for b in pred['gt_boxes']]
    pred_set = [b['class_name'] for b in pred['pred_boxes']]
    for gt_cls in set(gt_set):
        for pred_cls in set(pred_set):
            if gt_cls != pred_cls:
                confusion_pairs[(gt_cls, pred_cls)] += 1

print('Top Class Confusion Pairs (GT → Predicted):\n')
for (gt, pred_cls), count in confusion_pairs.most_common(10):
    print(f'  {gt:>12s} → {pred_cls:<12s} : {count} images')

# ── Worst missed detections (most GT boxes missed) ────────

missed_sorted = sorted(missed_detections,
                        key=lambda x: x['n_gt'] - x['n_pred'], reverse=True)

print('Worst Missed Detections:\n')
for i, pred in enumerate(missed_sorted[:10], 1):
    diff = pred['n_gt'] - pred['n_pred']
    gt_cls = [b['class_name'] for b in pred['gt_boxes']]
    print(f'  #{i}  missed {diff} of {pred["n_gt"]} objects  '
          f'GT: {dict(Counter(gt_cls))}  '
          f'file: {Path(pred["image_path"]).name}')

# ── Montage: worst missed detections ──────────────────────

n_show = min(len(missed_sorted), 8)
if n_show > 0:
    cols = min(n_show, 4)
    rows = (n_show + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    fig.suptitle('Worst Missed Detections (GT boxes shown)', fontsize=16, fontweight='bold', y=1.02)
    axes_flat = np.array(axes).flatten() if n_show > 1 else [axes]

    for i, ax in enumerate(axes_flat):
        if i < n_show:
            pred = missed_sorted[i]
            img_path = pred['image_path']
            lbl_path = test_lbl_dir / (Path(img_path).stem + '.txt')
            plot_image_with_boxes(img_path, lbl_path, classes, ax)
            diff = pred['n_gt'] - pred['n_pred']
            ax.set_title(f'Missed {diff}/{pred["n_gt"]} objects',
                         fontsize=10, color='red', fontweight='bold')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/missed_detections.png', dpi=150, bbox_inches='tight')
    plt.show()

# ── Worst false positives (most extra predictions) ────────

fp_sorted = sorted(false_positives,
                    key=lambda x: x['n_pred'] - x['n_gt'], reverse=True)

n_show = min(len(fp_sorted), 8)
if n_show > 0:
    cols = min(n_show, 4)
    rows = (n_show + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    fig.suptitle('Worst False Positives (extra predictions)', fontsize=16, fontweight='bold', y=1.02)
    axes_flat = np.array(axes).flatten() if n_show > 1 else [axes]

    for i, ax in enumerate(axes_flat):
        if i < n_show:
            pred_data = fp_sorted[i]
            img_path = pred_data['image_path']
            img = Image.open(img_path).convert('RGB')
            ax.imshow(img)

            # Draw predicted boxes in red
            for box in pred_data['pred_boxes']:
                x1, y1, x2, y2 = box['bbox_xyxy']
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                         linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1-4, f"{box['class_name']} {box['confidence']:.2f}",
                        color='white', fontsize=7, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.7))

            extra = pred_data['n_pred'] - pred_data['n_gt']
            ax.set_title(f'+{extra} false positives (pred={pred_data["n_pred"]}, gt={pred_data["n_gt"]})',
                         fontsize=9, color='red', fontweight='bold')
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/false_positives.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print('No false positives found!')

# ── Low-confidence detections (uncertain predictions) ─────
# These are borderline cases your robot might get wrong in the field

low_conf_preds = []
for pred in all_predictions:
    for box in pred['pred_boxes']:
        if box['confidence'] < 0.5:
            low_conf_preds.append({
                'image_path': pred['image_path'],
                'class_name': box['class_name'],
                'confidence': box['confidence'],
            })

low_conf_preds.sort(key=lambda x: x['confidence'])

print(f'Predictions below 0.5 confidence: {len(low_conf_preds)}\n')
print('Lowest confidence predictions:')
for i, lc in enumerate(low_conf_preds[:15], 1):
    print(f'  #{i:>2d}  conf={lc["confidence"]:.3f}  '
          f'class={lc["class_name"]:>12s}  '
          f'file={Path(lc["image_path"]).name}')

# ── Per-class detection rate ──────────────────────────────

gt_per_class = Counter()
pred_per_class = Counter()

for pred in all_predictions:
    for b in pred['gt_boxes']:
        gt_per_class[b['class_name']] += 1
    for b in pred['pred_boxes']:
        pred_per_class[b['class_name']] += 1

print(f'{"Class":>12s} {"GT Boxes":>10s} {"Pred Boxes":>12s} {"Ratio":>8s}')
print('-' * 46)
for cls in classes:
    gt = gt_per_class.get(cls, 0)
    pr = pred_per_class.get(cls, 0)
    ratio = pr / gt if gt > 0 else 0
    flag = ' ⚠' if ratio < 0.7 or ratio > 1.3 else ''
    print(f'{cls:>12s} {gt:>10,} {pr:>12,} {ratio:>8.2f}{flag}')

# ── Error analysis summary plot ───────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Error Analysis Summary', fontsize=16, fontweight='bold')

# Error type breakdown
labels = ['Clean', 'Missed\nDetections', 'False\nPositives']
sizes  = [len(clean_detections), len(missed_detections), len(false_positives)]
colors_pie = ['#4CAF50', '#F44336', '#FF9800']
axes[0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_pie,
            startangle=90, textprops={'fontsize': 11})
axes[0].set_title('Detection Outcome Breakdown')

# GT vs Pred box counts per class
x = np.arange(len(classes))
gt_vals   = [gt_per_class.get(c, 0) for c in classes]
pred_vals = [pred_per_class.get(c, 0) for c in classes]

axes[1].bar(x - 0.2, gt_vals, 0.4, label='Ground Truth', color='#2196F3')
axes[1].bar(x + 0.2, pred_vals, 0.4, label='Predictions', color='#FF9800')
axes[1].set_xticks(x)
axes[1].set_xticklabels(classes, rotation=30, ha='right')
axes[1].set_ylabel('Box Count')
axes[1].set_title('GT vs Predicted Box Counts')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/error_analysis_summary.png', dpi=150, bbox_inches='tight')
plt.show()