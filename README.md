# Sidewalk Trash Detection & Disposal Classification

Sidewalk trash detection and recycle/landfill classification for an autonomous ground robot using YOLO11 transfer learning. Detects 6 waste categories and classifies each as recyclable or landfill, achieving **0.917 mAP@50** and **0.835 mAP@50:95** on the test set.

## Overview

This project builds the **perception module** for an autonomous ground robot designed to detect and sort sidewalk trash. The pipeline performs two tasks:

1. **Object Detection** — Localize and classify waste items into 6 categories using YOLO11n
2. **Disposal Classification** — Map each detection to a disposal action (recycle or landfill)

## Architecture

- **Model**: YOLO11n (nano) with transfer learning from COCO pretrained weights
- **Backbone**: C3k2 + SPPF + C2PSA (with attention)
- **Neck**: FPN + PAN
- **Head**: Decoupled detection head (3 scales)
- **Parameters**: 2.59M (trainable)
- **GFLOPs**: 6.4

## Dataset

| Split | Images | Labels |
| ----- | ------ | ------ |
| Train | 5,301  | 5,301  |
| Valid | 328    | 328    |
| Test  | 432    | 432    |

**6 Classes:**

| Class     | Train Boxes   | Disposal    |
| --------- | ------------- | ----------- |
| Cardboard | 270 (5.1%)    | ♻️ Recycle  |
| Glass     | 885 (16.6%)   | ♻️ Recycle  |
| Metal     | 1,053 (19.7%) | ♻️ Recycle  |
| Paper     | 861 (16.1%)   | ♻️ Recycle  |
| Plastic   | 1,217 (22.8%) | ♻️ Recycle  |
| Trash     | 1,053 (19.7%) | 🗑️ Landfill |

## Results

### Overall Metrics (Test Set)

| Metric    | Score  |
| --------- | ------ |
| mAP@50    | 0.9174 |
| mAP@50:95 | 0.8352 |
| Precision | 0.8720 |
| Recall    | 0.9017 |

### Per-Class Average Precision

| Class     | AP@50  | AP@50:95 |
| --------- | ------ | -------- |
| Cardboard | 0.8970 | 0.8343   |
| Glass     | 0.9124 | 0.8195   |
| Metal     | 0.9528 | 0.8869   |
| Paper     | 0.9514 | 0.9047   |
| Plastic   | 0.8901 | 0.7389   |
| Trash     | 0.9009 | 0.8270   |

### Error Analysis

- **Missed detections**: 3 / 432 images (0.7%)
- **False positives**: 36 / 432 images (8.3%)
- **Top confusion pair**: glass → plastic (15 images)

## Project Structure

```
├── README.md
├── .gitignore
├── data.yaml                    # Dataset config (paths + class names)
├── yolo11_trash_detection.ipynb # Main training & evaluation notebook
├── runs/
│   └── project_detect/
│       └── yolo11n_6cls/
│           ├── weights/
│           │   ├── best.pt      # Best model weights
│           │   └── last.pt      # Last epoch weights
│           ├── results.csv      # Training metrics per epoch
│           ├── confusion_matrix.png
│           └── ...
└── plots/
    ├── ground_truth_samples.png
    ├── dataset_distribution.png
    ├── bbox_size_distribution.png
    ├── per_class_ap.png
    ├── confidence_distribution.png
    ├── training_curves.png
    ├── sample_predictions.png
    ├── disposal_classification.png
    ├── disposal_predictions_sample.png
    ├── error_analysis_summary.png
    ├── missed_detections.png
    └── false_positives.png
```

## Training Configuration

| Parameter     | Value                          |
| ------------- | ------------------------------ |
| Base Model    | `yolo11n.pt` (COCO pretrained) |
| Image Size    | 640 × 640                      |
| Batch Size    | 32                             |
| Epochs        | 50 (early stopped at 43)       |
| Optimizer     | AdamW (lr=0.001, momentum=0.9) |
| Patience      | 10                             |
| GPU           | NVIDIA L40S (48 GB)            |
| Training Time | 33.5 minutes                   |

## Setup & Usage

### Requirements

```bash
pip install -r trash_detection_script/requirements.txt
```

### Training

```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=32,
    patience=10,
)
```

### Inference

```python
model = YOLO('runs/project_detect/yolo11n_6cls/weights/best.pt')
results = model.predict(source='path/to/image.jpg', imgsz=640, conf=0.25)
```

### Disposal Classification

```python
DISPOSAL_MAP = {
    'cardboard': 'recycle', 'glass': 'recycle', 'metal': 'recycle',
    'paper': 'recycle', 'plastic': 'recycle', 'trash': 'landfill',
}

for box in results[0].boxes:
    cls_name = model.names[int(box.cls.item())]
    disposal = DISPOSAL_MAP[cls_name]
    print(f'{cls_name} → {disposal}')
```

## Known Limitations

- **Glass ↔ Plastic confusion**: The top confusion pair (15 test images), likely due to visual similarity of transparent materials. Does not affect disposal classification since both map to recycle.
- **Cardboard underrepresentation**: Only 5.1% of training boxes, leading to occasional misclassification as trash or plastic.
- **Single-item images**: The dataset predominantly contains one object per image; multi-object cluttered scenes (realistic sidewalk conditions) are underrepresented.
- **Edge deployment**: Inference benchmarks (0.7ms) are on an L40S GPU and won't reflect embedded/edge hardware performance.

## Next Steps

1. Scale up to YOLO11s/m for higher accuracy
2. Tune confidence threshold (0.35–0.40) to reduce false positives
3. Address cardboard class imbalance via oversampling or `copy_paste` augmentation
4. Benchmark inference on target robot hardware (Jetson, etc.)
5. Integrate `best.pt` into robot perception pipeline (ROS node)
6. Compare against Faster R-CNN baseline

## Acknowledgments

- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
