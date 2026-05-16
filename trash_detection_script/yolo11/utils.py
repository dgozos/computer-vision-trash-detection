"""
=============================================================================
utils.py — Generalized Utilities for Image Classification with PyTorch
=============================================================================

A reusable toolkit for any image classification project that follows
this flat-folder + CSV label structure:

    your_dataset/
    ├── train/              ← all training images in ONE flat folder
    │   ├── image_00001.jpg
    │   ├── image_00002.jpg
    │   └── ...
    ├── test/               ← all test images in ONE flat folder
    │   ├── image_03001.jpg
    │   └── ...
    ├── labels.csv          ← master label file (see format below)
    └── class_mapping.csv   ← class_id ↔ class_name lookup

Required CSV format for labels.csv:
    filename,split,class_id,class_name
    image_00001.jpg,train,0,CAT
    image_00002.jpg,train,1,DOG
    image_03001.jpg,test,0,CAT

Required CSV format for class_mapping.csv:
    class_id,class_name
    0,CAT
    1,DOG

To use with a new dataset, just run your own data prep script that
produces the folder structure + CSVs above. Everything else works
out of the box.

=============================================================================
MODULE CONTENTS
=============================================================================

1. DATASET
   - ImageDataset          Custom Dataset: reads flat folder + CSV labels
   - AugmentedSubset       Wrapper to apply augmentation to a Subset

2. DATA LOADING
   - get_transforms()      Returns train (augmented) and eval transforms
   - load_labels()         Reads labels.csv + class_mapping.csv
   - load_train_dataset()  Creates ImageDataset for the train split
   - load_test_loader()    Creates DataLoader for the test split

3. CROSS-VALIDATION
   - stratified_kfold_split()   Stratified K-fold index generation
   - create_fold_loaders()      Build train/val DataLoaders for one fold

4. TRAINING
   - count_parameters()    Count total/trainable model parameters
   - train_one_epoch()     Single epoch forward+backward pass
   - evaluate()            Evaluate model on a DataLoader (no grad)
   - train_single_fold()   Full training loop for one fold (early stopping)
   - train_with_cv()       Orchestrates 5-fold CV over all folds

5. EVALUATION
   - get_predictions()           Collect predictions + probabilities
   - print_classification_report()  Per-class precision/recall/F1
   - print_train_cv_test_summary()   Gap analysis for overfitting/underfitting

6. PLOTTING
   - plot_cv_training_curves()   Loss & accuracy curves for all folds
   - plot_confusion_matrix()     Heatmap of true vs predicted labels
   - plot_comparison()           Bar chart comparing multiple models
   - plot_roc_curves()           One-vs-Rest ROC curves and per-class AUC

7. VISUALIZATION
   - visualize_feature_maps()    Hooks into layers to plot internal channel activations
   - visualize_stn()             Side-by-side plot of input vs STN-transformed images
   - plot_gradcam()              Generates gradient-weighted class activation heatmaps

8. PROFILING & HELPERS
   - format_number()             Format large numbers with K/M/G suffixes
   - compute_flops()             Analytically estimate FLOPs for a single layer
   - profile_model()             Print a detailed block-by-block FLOP/Param breakdown

=============================================================================
"""

import os
import time
import copy
import math
from pathlib import Path
from collections import defaultdict
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from collections import OrderedDict
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
# Use non-interactive backend only if not in a notebook (Colab, Jupyter).
# In notebooks, the default inline backend handles display automatically.
import sys

if 'ipykernel' not in sys.modules and 'google.colab' not in sys.modules:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

# Device selection: uses GPU if available, otherwise CPU.
# All model training and inference functions in this module use this device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Disable torch.compile (Dynamo/Inductor) — requires 'triton' (Linux only)
try:
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True
except ImportError:
    pass

# ImageNet normalization statistics.
# These are standard for models pretrained on ImageNet, and also work well
# as a default for natural-image datasets (photos of animals, objects, etc.).
# If your dataset has very different pixel distributions (e.g., medical images,
# satellite imagery), consider computing dataset-specific mean/std.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# DataLoader settings: num_workers=0 is safest for Colab (avoids multiprocessing
# crashes). pin_memory only helps when using GPU. These are set automatically.
IN_COLAB = 'google.colab' in sys.modules
NUM_WORKERS = 0 if IN_COLAB else 2
PIN_MEMORY = torch.cuda.is_available()


# ============================================================================
# 1. DATASET CLASSES
# ============================================================================

class ImageDataset(Dataset):
    """
    Generic PyTorch Dataset for flat-folder image classification.

    Unlike torchvision.ImageFolder (which infers labels from subfolder names),
    this class reads labels from a pandas DataFrame. Images are stored in a
    single flat directory with no class subfolders.

    This design is more flexible and mirrors how real-world datasets are often
    distributed (images + a CSV/JSON annotation file).

    Args:
        image_dir (str or Path):
            Path to the flat folder containing image files.
            Example: 'my_dataset/train/'

        labels_df (pd.DataFrame):
            Must contain at least two columns:
              - 'filename':  image file name (e.g., 'image_00001.jpg')
              - 'class_id':  integer class label (e.g., 0, 1, 2, ...)
            Should be pre-filtered to the desired split (train or test).

        transform (torchvision.transforms.Compose, optional):
            Image transform pipeline. If None, returns raw PIL image
            (not recommended for training — at minimum convert to tensor).

    Attributes:
        targets (list[int]):  List of class_id values, used by
                              stratified_kfold_split() for balanced splitting.
        filenames (list[str]): List of filenames in order.

    Example:
        labels_df = pd.read_csv('labels.csv')
        train_df = labels_df[labels_df['split'] == 'train']
        dataset = ImageDataset('data/train/', train_df, transform=my_transform)
        image, label = dataset[0]   # returns (Tensor, int)
    """

    def __init__(self, image_dir, labels_df, transform=None):
        """Initialize with image directory path, labels DataFrame, and optional transform."""
        self.image_dir = Path(image_dir)
        self.labels_df = labels_df.reset_index(drop=True)
        self.transform = transform
        self.targets = self.labels_df['class_id'].tolist()
        self.filenames = self.labels_df['filename'].tolist()

    def __len__(self):
        """Return total number of images in this dataset."""
        return len(self.labels_df)

    def __getitem__(self, idx):
        """Load image at index `idx`, apply transform, return (image, label) tuple."""
        row = self.labels_df.iloc[idx]
        img_path = self.image_dir / row['filename']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = int(row['class_id'])
        return image, label


class AugmentedSubset(Dataset):
    """
    Wrapper that applies a DIFFERENT transform to a torch Subset.

    Problem it solves:
        When doing cross-validation, we create Subsets from a base dataset.
        The base dataset uses eval transforms (no augmentation). But the
        training fold needs augmentation. We can't change the base dataset's
        transform (it would affect the validation fold too).

    Solution:
        This wrapper intercepts __getitem__, re-opens the original image
        from disk, and applies the augmented transform instead.

    Args:
        subset (torch.utils.data.Subset):
            A subset of an ImageDataset.
        transform (torchvision.transforms.Compose):
            The augmented transform to apply (e.g., with flips, rotations).

    Note:
        Re-opening the image from disk is slightly slower than in-memory
        transforms, but it's the cleanest way to apply different transforms
        to different subsets of the same base dataset.
    """

    def __init__(self, subset, transform):
        """Initialize with a Subset and the augmented transform to apply."""
        self.subset = subset
        self.transform = transform

    def __len__(self):
        """Return number of samples in the subset."""
        return len(self.subset)

    def __getitem__(self, idx):
        """Re-open original image from disk and apply augmented transform."""
        original_idx = self.subset.indices[idx]
        base_ds = self.subset.dataset
        row = base_ds.labels_df.iloc[original_idx]
        img_path = base_ds.image_dir / row['filename']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = int(row['class_id'])
        return image, label


# ============================================================================
# 2. DATA LOADING
# ============================================================================

def get_transforms(img_size=150, augment=True):
    """
    Build image transform pipelines for training and evaluation.

    Training transform (augment=True):
        Resize → RandomHorizontalFlip → RandomRotation → ColorJitter
        → ToTensor → Normalize

    Eval transform (augment=False / always returned as second value):
        Resize → ToTensor → Normalize

    Args:
        img_size (int): Target height and width (images are resized to square).
                        Default 150 balances quality vs. training speed.
        augment (bool): Whether the training transform includes augmentation.

    Returns:
        train_tf:  training transform (with or without augmentation)
        eval_tf:   evaluation/test transform (never augmented)

    Customization tips:
        - For high-res datasets, increase img_size (e.g., 224)
        - Add RandomResizedCrop for scale invariance
        - Add RandomErasing for occlusion robustness
        - Adjust ColorJitter ranges based on your domain
    """
    if augment:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


def load_labels(data_dir):
    """
    Load the label CSV and class mapping from the processed dataset directory.

    Expects:
        data_dir/labels.csv          — every image with its label
        data_dir/class_mapping.csv   — class_id ↔ class_name lookup

    If class_mapping.csv is missing, class names are inferred from labels.csv
    (sorted unique values of the 'class_name' column).

    Args:
        data_dir (str): Root directory of the processed dataset.

    Returns:
        labels_df (pd.DataFrame):   Full labels table (all splits).
        class_names (list[str]):     Ordered class names (index = class_id).
        num_classes (int):           Number of unique classes.
    """
    labels_path = Path(data_dir) / 'labels.csv'
    labels_df = pd.read_csv(labels_path)

    mapping_path = Path(data_dir) / 'class_mapping.csv'
    if mapping_path.exists():
        mapping_df = pd.read_csv(mapping_path)
        class_names = mapping_df.sort_values('class_id')['class_name'].tolist()
    else:
        class_names = sorted(labels_df['class_name'].unique())

    num_classes = len(class_names)
    return labels_df, class_names, num_classes


def load_train_dataset(data_dir, labels_df, img_size=150):
    """
    Create an ImageDataset for the training split (used for CV splitting).

    The dataset is created with EVAL transforms (no augmentation).
    Augmentation is applied per-fold via AugmentedSubset in create_fold_loaders().

    Args:
        data_dir (str):           Root of processed dataset.
        labels_df (pd.DataFrame): Full labels table from load_labels().
        img_size (int):           Image resize target.

    Returns:
        dataset (ImageDataset): Full training dataset.
    """
    _, eval_tf = get_transforms(img_size, augment=False)
    train_df = labels_df[labels_df['split'] == 'train'].reset_index(drop=True)
    train_dir = os.path.join(data_dir, 'train')
    dataset = ImageDataset(train_dir, train_df, transform=eval_tf)
    print(f"  Train dataset: {len(dataset)} images")
    return dataset


def load_test_loader(data_dir, labels_df, img_size=150, batch_size=32):
    """
    Create a DataLoader for the test split.

    Uses eval transforms (no augmentation, no shuffling).

    Args:
        data_dir (str):           Root of processed dataset.
        labels_df (pd.DataFrame): Full labels table from load_labels().
        img_size (int):           Image resize target.
        batch_size (int):         Batch size for the DataLoader.

    Returns:
        test_loader (DataLoader): Ready-to-iterate test DataLoader.
    """
    _, eval_tf = get_transforms(img_size, augment=False)
    test_df = labels_df[labels_df['split'] == 'test'].reset_index(drop=True)
    test_dir = os.path.join(data_dir, 'test')
    test_ds = ImageDataset(test_dir, test_df, transform=eval_tf)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    print(f"  Test dataset:  {len(test_ds)} images")
    return test_loader


# ============================================================================
# 3. CROSS-VALIDATION
# ============================================================================

def stratified_kfold_split(dataset, n_folds=5, seed=42):
    """
    Generate stratified K-fold train/validation index splits.

    "Stratified" means each fold preserves the same class proportions as
    the full dataset. This is critical for imbalanced datasets — without it,
    some folds might have zero samples of rare classes.

    Algorithm:
        1. Group sample indices by class label
        2. Shuffle within each class (reproducibly via seed)
        3. Split each class's indices into n_folds roughly equal chunks
        4. For each fold i: chunk i = validation, all other chunks = training

    Args:
        dataset:   Must have a .targets attribute (list of int labels).
                   ImageDataset provides this automatically.
        n_folds:   Number of folds (default 5 → 80/20 train/val per fold).
        seed:      Random seed for reproducibility.

    Returns:
        folds (list[tuple]):  List of (train_indices, val_indices) tuples.
                              Length = n_folds.

    Example:
        folds = stratified_kfold_split(dataset, n_folds=5)
        for train_idx, val_idx in folds:
            train_loader, val_loader = create_fold_loaders(
                dataset, train_idx, val_idx
            )
    """
    rng = np.random.RandomState(seed)
    targets = np.array(dataset.targets)
    classes = np.unique(targets)

    # Group indices by class
    class_indices = {c: np.where(targets == c)[0] for c in classes}

    # Shuffle within each class
    for c in classes:
        rng.shuffle(class_indices[c])

    # Split each class into n_folds chunks
    fold_indices = [[] for _ in range(n_folds)]
    for c in classes:
        splits = np.array_split(class_indices[c], n_folds)
        for fold_id in range(n_folds):
            fold_indices[fold_id].extend(splits[fold_id].tolist())

    # Build (train, val) pairs
    folds = []
    for val_fold in range(n_folds):
        val_idx = fold_indices[val_fold]
        train_idx = []
        for j in range(n_folds):
            if j != val_fold:
                train_idx.extend(fold_indices[j])
        folds.append((train_idx, val_idx))

    return folds


def create_fold_loaders(dataset, train_idx, val_idx,
                        img_size=150, batch_size=32, augment=True):
    """
    Build DataLoaders for one fold of cross-validation.

    Training fold:   wrapped in AugmentedSubset (data augmentation applied).
    Validation fold:  uses base dataset's eval transforms (no augmentation).

    Args:
        dataset:    Base ImageDataset (with eval transforms).
        train_idx:  List of indices for training samples this fold.
        val_idx:    List of indices for validation samples this fold.
        img_size:   Image size (must match the base dataset's transform).
        batch_size: Batch size for both loaders.
        augment:    Whether to apply augmentation to the training fold.

    Returns:
        train_loader (DataLoader): Shuffled, augmented training data.
        val_loader (DataLoader):   Non-shuffled, non-augmented validation data.
    """
    train_tf, eval_tf = get_transforms(img_size, augment)

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_ds = AugmentedSubset(train_subset, train_tf) if augment else train_subset

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    return train_loader, val_loader


# ============================================================================
# 4. TRAINING
# ============================================================================

def count_parameters(model):
    """
    Count model parameters.

    Returns:
        total (int):     Total number of parameters.
        trainable (int): Number of parameters with requires_grad=True.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train_one_epoch(model, loader, criterion, optimizer):
    """
    Run one training epoch: forward pass, loss, backward pass, optimizer step.

    Args:
        model:     PyTorch model (set to train mode internally).
        loader:    Training DataLoader.
        criterion: Loss function (e.g., nn.CrossEntropyLoss()).
        optimizer: Optimizer (e.g., Adam).

    Returns:
        avg_loss (float):  Mean loss over all samples.
        accuracy (float):  Fraction of correct predictions (0.0 to 1.0).
    """
    model.train()
    running_loss = 0.0
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    """
    Evaluate model on a DataLoader (no gradient computation).

    Args:
        model:     PyTorch model (set to eval mode internally).
        loader:    Evaluation DataLoader (validation or test).
        criterion: Loss function.

    Returns:
        avg_loss (float):  Mean loss over all samples.
        accuracy (float):  Fraction of correct predictions (0.0 to 1.0).
    """
    model.eval()
    running_loss = 0.0
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
    return running_loss / total, correct / total


def train_single_fold(model, train_loader, val_loader,
                      epochs=30, lr=0.001, patience=8, verbose=True):
    """
    Full training loop for one fold (or one train/val split).

    Includes:
        - Adam optimizer with weight decay (1e-4)
        - ReduceLROnPlateau scheduler (halves LR after 4 epochs without
          improvement in validation loss)
        - Early stopping (stops training after `patience` epochs without
          improvement in validation accuracy)
        - Best model checkpoint (restores weights from the epoch with
          highest validation accuracy)

    Args:
        model:        PyTorch model (will be moved to device).
        train_loader: Training DataLoader for this fold.
        val_loader:   Validation DataLoader for this fold.
        epochs (int): Maximum number of training epochs.
        lr (float):   Initial learning rate.
        patience (int): Early stopping patience (epochs without improvement).
        verbose (bool): Print per-epoch metrics.

    Returns:
        history (dict):     Keys: 'train_loss', 'train_acc', 'val_loss', 'val_acc'
                            Each is a list of per-epoch values.
        best_val_acc (float): Best validation accuracy achieved.
        best_state (dict):    model.state_dict() from the best epoch.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4
    )

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if verbose:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch:3d}/{epochs} │ "
                  f"train_loss={train_loss:.4f}  acc={train_acc:.3f} │ "
                  f"val_loss={val_loss:.4f}  acc={val_acc:.3f} │ "
                  f"lr={lr_now:.1e} ({elapsed:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"    Early stop at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return history, best_val_acc, best_state


def train_with_cv(model_fn, model_name, dataset, folds, img_size=150,
                  batch_size=32, epochs=30, lr=0.001, patience=8,
                  augment=True, verbose=True):
    """
    Run 5-fold (or K-fold) cross-validation.

    For each fold:
        1. Create train/val DataLoaders from the fold indices
        2. Instantiate a FRESH model (via model_fn)
        3. Train with train_single_fold()
        4. Record validation accuracy

    After all folds, prints mean ± std validation accuracy and returns
    the model state from the fold with the highest validation accuracy.

    Args:
        model_fn (callable): Function that returns a new model instance.
                             Called once per fold to ensure fresh weights.
                             Example: lambda: MyModel(num_classes=20)
        model_name (str):    Display name for printing.
        dataset:             Full training ImageDataset.
        folds (list):        Output of stratified_kfold_split().
        img_size (int):      Image size for transforms.
        batch_size (int):    Batch size.
        epochs (int):        Max epochs per fold.
        lr (float):          Learning rate.
        patience (int):      Early stopping patience.
        augment (bool):      Apply data augmentation to training folds.
        verbose (bool):      Print per-epoch training logs.

    Returns:
        fold_results (list[dict]): Per-fold results. Each dict contains:
            - 'fold': fold number (1-indexed)
            - 'best_val_acc': best validation accuracy for this fold
            - 'final_train_acc': training accuracy at last epoch
            - 'time': wall-clock seconds for this fold
            - 'history': full epoch-by-epoch history dict
        best_model_state (dict): state_dict from the best fold overall.
    """
    total_p, trainable_p = count_parameters(model_fn())

    print(f"\n{'=' * 65}")
    print(f"  {model_name} — {len(folds)}-Fold Cross-Validation")
    print(f"{'=' * 65}")
    print(f"  Parameters: {total_p:,} ({trainable_p:,} trainable)")
    print(f"  Optimizer:  Adam (lr={lr}), Epochs: {epochs}, Patience: {patience}")

    fold_results = []
    best_overall_acc = 0.0
    best_model_state = None

    for fold_id, (train_idx, val_idx) in enumerate(folds):
        print(f"\n  ── Fold {fold_id + 1}/{len(folds)} "
              f"(train={len(train_idx)}, val={len(val_idx)}) ──")

        train_loader, val_loader = create_fold_loaders(
            dataset, train_idx, val_idx, img_size, batch_size, augment
        )

        model = model_fn()  # fresh model each fold
        t0 = time.time()
        history, best_val_acc, state = train_single_fold(
            model, train_loader, val_loader,
            epochs=epochs, lr=lr, patience=patience, verbose=verbose
        )
        fold_time = time.time() - t0

        print(f"    Fold {fold_id + 1} best val acc: {best_val_acc:.4f}  ({fold_time:.1f}s)")

        fold_results.append({
            'fold': fold_id + 1,
            'best_val_acc': best_val_acc,
            'final_train_acc': history['train_acc'][-1],
            'time': fold_time,
            'history': history,
        })

        if best_val_acc > best_overall_acc:
            best_overall_acc = best_val_acc
            best_model_state = copy.deepcopy(state)

    # Print summary
    val_accs = [r['best_val_acc'] for r in fold_results]
    mean_acc = np.mean(val_accs)
    std_acc = np.std(val_accs)
    print(f"\n  CV Result: {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"  Per-fold:  {' | '.join(f'{a:.3f}' for a in val_accs)}")

    return fold_results, best_model_state


# ============================================================================
# 5. EVALUATION
# ============================================================================

@torch.no_grad()
def get_predictions(model, loader):
    """
    Collect predictions, true labels, and class probabilities from a DataLoader.

    Useful for computing detailed metrics, confusion matrices, or
    analyzing prediction confidence after training.

    Args:
        model:  Trained PyTorch model.
        loader: DataLoader to predict on (typically test_loader).

    Returns:
        preds (np.ndarray):   Predicted class indices, shape (N,).
        labels (np.ndarray):  True class indices, shape (N,).
        probs (np.ndarray):   Softmax probabilities, shape (N, num_classes).
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def print_classification_report(preds, labels, class_names):
    """
    Print per-class precision, recall, F1-score and overall accuracy.

    Computes metrics from scratch (no sklearn dependency):
        Precision = TP / (TP + FP)   — of predicted positives, how many correct?
        Recall    = TP / (TP + FN)   — of actual positives, how many found?
        F1        = 2 * P * R / (P + R) — harmonic mean of precision and recall

    Args:
        preds (np.ndarray):      Predicted class indices.
        labels (np.ndarray):     True class indices.
        class_names (list[str]): Ordered class names for display.

    Returns:
        accuracy (float): Overall accuracy (correct / total).
    """
    metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    for p, l in zip(preds, labels):
        if p == l:
            metrics[l]['tp'] += 1
        else:
            metrics[p]['fp'] += 1
            metrics[l]['fn'] += 1

    print(f"\n  {'Class':<30s}  {'Prec':>6s}  {'Recall':>6s}  {'F1':>6s}  {'N':>4s}")
    print(f"  {'-' * 30}  {'-' * 6}  {'-' * 6}  {'-' * 6}  {'-' * 4}")
    for idx, name in enumerate(class_names):
        m = metrics[idx]
        prec = m['tp'] / (m['tp'] + m['fp']) if (m['tp'] + m['fp']) else 0
        rec = m['tp'] / (m['tp'] + m['fn']) if (m['tp'] + m['fn']) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        print(f"  {name:<30s}  {prec:6.3f}  {rec:6.3f}  {f1:6.3f}  {m['tp'] + m['fn']:4d}")

    acc = (preds == labels).mean()
    print(f"\n  Overall accuracy: {acc:.4f} ({(preds == labels).sum()}/{len(labels)})")
    return acc

def print_train_cv_test_summary(fold_results, test_loss, test_acc, model_name):
    """
    Print a side-by-side comparison of Training, CV Validation, and Test metrics.

    Extracts from fold_results:
      - Training loss/acc: average of each fold's BEST-EPOCH training metrics
      - CV Val loss/acc:   average of each fold's BEST-EPOCH validation metrics

    Then compares with the held-out test set results.

    This answers the key question:
      - Train >> CV Val?        → overfitting (model memorizes training data)
      - Train ≈ CV Val ≈ Test?  → good generalization
      - All three are low?       → underfitting (model too simple)

    Args:
        fold_results: list of per-fold dicts from train_with_cv()
        test_loss:    test set loss from evaluate()
        test_acc:     test set accuracy from evaluate()
        model_name:   display name for the header
    """
    # ── Extract per-fold metrics at the best epoch ──
    train_losses = []
    train_accs   = []
    val_losses   = []
    val_accs     = []

    for fold in fold_results:
        history = fold.get('history', {})

        # Per-epoch lists are inside fold['history']
        h_train_loss = history.get('train_loss', [])
        h_train_acc  = history.get('train_acc', [])
        h_val_loss   = history.get('val_loss', [])
        h_val_acc    = history.get('val_acc', [])

        if h_val_acc:
            # Find best epoch (highest val acc)
            best_idx = int(np.argmax(h_val_acc))

            if h_train_loss:
                train_losses.append(h_train_loss[best_idx])
            if h_train_acc:
                train_accs.append(h_train_acc[best_idx])
            if h_val_loss:
                val_losses.append(h_val_loss[best_idx])
            val_accs.append(h_val_acc[best_idx])
        else:
            # Fallback: use top-level keys if history is empty
            if 'best_val_acc' in fold:
                val_accs.append(fold['best_val_acc'])
            if 'final_train_acc' in fold:
                train_accs.append(fold['final_train_acc'])

    # ── Compute averages ──
    avg_train_loss = np.mean(train_losses) if train_losses else float('nan')
    avg_train_acc  = np.mean(train_accs)   if train_accs   else float('nan')
    std_train_acc  = np.std(train_accs)    if train_accs   else float('nan')

    avg_val_loss   = np.mean(val_losses)   if val_losses   else float('nan')
    avg_val_acc    = np.mean(val_accs)     if val_accs     else float('nan')
    std_val_acc    = np.std(val_accs)      if val_accs     else float('nan')


    # ── Print ──
    W = 66
    print(f"\n  {'═'*W}")
    print(f"  {model_name} — Train vs CV Validation vs Test")
    print(f"  {'═'*W}")
    print(f"  {'Metric':<20s} {'Training':>14s} {'CV Validation':>14s} {'Test':>14s}")
    print(f"  {'─'*W}")
    print(f"  {'Loss':<20s} {avg_train_loss:>14.4f} {avg_val_loss:>14.4f} {test_loss:>14.4f}")
    print(f"  {'Accuracy':<20s} {avg_train_acc*100:>13.1f}% {avg_val_acc*100:>13.1f}% {test_acc*100:>13.1f}%")
    print(f"  {'Acc ± std':<20s} {'':>4s}± {std_train_acc*100:.1f}% {'':>4s}± {std_val_acc*100:.1f}% {'':>8s}—")
    print(f"  {'─'*W}")

    # Gap analysis
    train_val_gap = (avg_train_acc - avg_val_acc) * 100
    val_test_gap  = (avg_val_acc - test_acc) * 100
    print(f"  {'Train→Val gap':<20s} {train_val_gap:>+.1f}%")
    print(f"  {'Val→Test gap':<20s} {val_test_gap:>+.1f}%")
    print(f"  {'═'*W}\n")

    return {
        'train_loss': avg_train_loss, 'train_acc': avg_train_acc,
        'val_loss':   avg_val_loss,   'val_acc':   avg_val_acc,
        'test_loss':  test_loss,      'test_acc':  test_acc,
    }

# ============================================================================
# 6. PLOTTING
# ============================================================================

def plot_cv_training_curves(fold_results, model_name, save_path):
    """
    Plot training and validation curves for all CV folds overlaid.

    Left panel:  Loss curves (train solid, val dashed) per fold.
    Right panel: Accuracy curves (train solid, val dashed) per fold.

    Each fold gets a distinct color. Training curves are faded (alpha=0.4)
    to emphasize validation curves.

    Args:
        fold_results (list[dict]): Output of train_with_cv().
        model_name (str):          Title prefix.
        save_path (str):           File path to save the plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = ['#4472C4', '#ED7D31', '#70AD47', '#FFC000', '#9B59B6']

    for r in fold_results:
        fold, h, c = r['fold'], r['history'], colors[(r['fold'] - 1) % len(colors)]
        epochs = range(1, len(h['train_loss']) + 1)
        ax1.plot(epochs, h['train_loss'], '-', color=c, alpha=0.4, lw=1)
        ax1.plot(epochs, h['val_loss'], '--', color=c, alpha=0.9, lw=1.5,
                 label=f"Fold {fold} val")
        ax2.plot(epochs, h['train_acc'], '-', color=c, alpha=0.4, lw=1)
        ax2.plot(epochs, h['val_acc'], '--', color=c, alpha=0.9, lw=1.5,
                 label=f"Fold {fold} val")

    ax1.set_xlabel('Epoch');
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} — Loss ({len(fold_results)}-Fold CV)')
    ax1.legend(fontsize=8);
    ax1.grid(alpha=0.3)
    ax2.set_xlabel('Epoch');
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{model_name} — Accuracy ({len(fold_results)}-Fold CV)')
    ax2.legend(fontsize=8);
    ax2.grid(alpha=0.3)
    plt.tight_layout();
    plt.savefig(save_path, dpi=150);
    plt.show();
    plt.close()
    print(f"  Curves saved: {save_path}")


def plot_confusion_matrix(preds, labels, class_names, model_name, save_path):
    """
    Plot a confusion matrix heatmap.

    Rows = true class, Columns = predicted class.
    Cell values are annotated. Uses 'Blues' colormap.
    Class names are truncated to 12 chars for readability.

    Args:
        preds (np.ndarray):      Predicted class indices.
        labels (np.ndarray):     True class indices.
        class_names (list[str]): Ordered class names.
        model_name (str):        Title prefix.
        save_path (str):         File path to save the plot.
    """
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for p, l in zip(preds, labels):
        cm[l][p] += 1

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues')
    short = [c[:12] for c in class_names]
    ax.set_xticks(range(n));
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    ax.set_xlabel('Predicted');
    ax.set_ylabel('True')
    ax.set_title(f'{model_name} — Confusion Matrix (Test)')
    for i in range(n):
        for j in range(n):
            if cm[i][j] > 0:
                color = 'white' if cm[i][j] > cm.max() / 2 else 'black'
                ax.text(j, i, str(cm[i][j]), ha='center', va='center',
                        fontsize=7, color=color)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout();
    plt.savefig(save_path, dpi=150);
    plt.show();
    plt.close()
    print(f"  Confusion matrix saved: {save_path}")


def plot_comparison(results, save_path):
    """
    Bar chart comparing multiple models side by side.

    Shows two bars per model:
        - CV mean accuracy (with ±std error bars)
        - Test accuracy

    Each bar is annotated with the accuracy value and parameter count.

    Args:
        results (list[dict]): Each dict must contain:
            - 'name':     model display name
            - 'cv_mean':  mean CV accuracy (0-1 scale)
            - 'cv_std':   std of CV accuracy
            - 'test_acc': test accuracy (0-1 scale)
            - 'params':   total parameter count
        save_path (str): File path to save the plot.
    """
    names = [r['name'] for r in results]
    cv_means = [r['cv_mean'] * 100 for r in results]
    cv_stds = [r['cv_std'] * 100 for r in results]
    test_accs = [r['test_acc'] * 100 for r in results]
    colors = ['#4472C4', '#ED7D31', '#70AD47', '#FFC000', '#5B9BD5']

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, cv_means, w, yerr=cv_stds, capsize=4,
           label='CV Mean +/- Std', color=colors[:len(names)], alpha=0.8)
    ax.bar(x + w / 2, test_accs, w, label='Test Acc',
           color=colors[:len(names)], alpha=0.5, edgecolor='black', lw=1)

    for bar, val, r in zip(ax.patches[len(names):], test_accs, results):
        p_str = f"{r['params'] / 1e6:.1f}M" if r['params'] > 1e6 else f"{r['params'] / 1e3:.0f}K"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%\n({p_str})", ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Comparison (5-Fold CV)')
    ax.set_xticks(x);
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    ax.legend();
    ax.set_ylim(0, max(max(cv_means), max(test_accs)) * 1.18)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout();
    plt.savefig(save_path, dpi=150);
    plt.show();
    plt.close()
    print(f"  Comparison chart saved: {save_path}")


def plot_roc_curves(probs, true_labels, class_names, model_name, test_acc, save_path):
    """
    Plot per-class ROC curves + macro-average ROC curve.

    Layout:
      - Left panel:  individual ROC curve for EACH class (thin colored lines)
                     + macro-average (thick black line) + random baseline (dashed)
      - Right panel: bar chart of per-class AUC values, sorted from worst to best
                     (makes it easy to spot which species the model struggles with)

    Args:
        probs:       predicted probabilities, shape (N, num_classes), from softmax
        true_labels: ground truth class indices, shape (N,)
        class_names: list of class name strings
        model_name:  display name for plot title
        test_acc:    test accuracy (shown in title for reference)
        save_path:   where to save the plot

    Returns:
        macro_auc: macro-averaged AUC across all classes
    """
    num_classes = len(class_names)

    # ── Step 1: Binarize labels for one-vs-rest ROC ──
    # Convert [3, 0, 7, ...] → [[0,0,0,1,...], [1,0,0,0,...], [0,0,0,0,...,1,...]]
    # Each row is a one-hot vector for that sample's true class.
    true_bin = label_binarize(true_labels, classes=range(num_classes))

    # ── Step 2: Compute ROC curve for each class ──
    fpr = {}  # false positive rate per class
    tpr = {}  # true positive rate per class
    roc_auc = {}  # AUC per class

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(true_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # ── Step 3: Compute macro-average ROC ──
    # Interpolate all curves to a common set of FPR points, then average
    all_fpr = np.linspace(0, 1, 200)
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes

    macro_auc_val = auc(all_fpr, mean_tpr)

    # ── Step 4: Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Left panel: ROC curves ──
    ax = axes[0]

    # Color cycle for per-class curves
    colors = plt.cm.tab20(np.linspace(0, 1, num_classes))

    # Plot each class
    for i in range(num_classes):
        ax.plot(fpr[i], tpr[i], color=colors[i], alpha=0.4, linewidth=1,
                label=f'{class_names[i]} (AUC={roc_auc[i]:.2f})')

    # Macro-average ROC (thick black line)
    ax.plot(all_fpr, mean_tpr, color='black', linewidth=2.5,
            label=f'Macro-avg (AUC={macro_auc_val:.3f})')

    # Random baseline (diagonal)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Random (AUC=0.5)')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'{model_name}\nROC Curves (Test Acc: {test_acc * 100:.1f}%)', fontsize=12)

    # Legend: if too many classes, put it outside
    if num_classes <= 10:
        ax.legend(fontsize=7, loc='lower right')
    else:
        ax.legend(fontsize=6, loc='lower right', ncol=2)

    ax.grid(True, alpha=0.2)

    # ── Right panel: AUC bar chart (sorted worst → best) ──
    ax2 = axes[1]

    # Sort by AUC ascending (worst at top, best at bottom)
    sorted_indices = sorted(range(num_classes), key=lambda i: roc_auc[i])
    sorted_names = [class_names[i] for i in sorted_indices]
    sorted_aucs = [roc_auc[i] for i in sorted_indices]

    # Color bars by AUC value: red (bad) → green (good)
    bar_colors = plt.cm.RdYlGn([(v - 0.5) / 0.5 for v in sorted_aucs])  # normalize 0.5–1.0 → 0–1

    bars = ax2.barh(range(num_classes), sorted_aucs, color=bar_colors, edgecolor='none')

    # Add AUC value labels on bars
    for j, (bar, auc_val) in enumerate(zip(bars, sorted_aucs)):
        ax2.text(auc_val + 0.005, j, f'{auc_val:.3f}', va='center', fontsize=7)

    ax2.set_yticks(range(num_classes))
    ax2.set_yticklabels(sorted_names, fontsize=7)
    ax2.set_xlabel('AUC', fontsize=11)
    ax2.set_title('Per-Class AUC (sorted)', fontsize=12)
    ax2.set_xlim([min(sorted_aucs) - 0.05, 1.02])
    ax2.axvline(x=macro_auc_val, color='black', linestyle='--', alpha=0.5,
                label=f'Macro avg: {macro_auc_val:.3f}')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2, axis='x')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  ROC curves saved: {save_path}")

    return macro_auc_val

# ============================================================================
# 7. VISUALIZATION
# ============================================================================
def visualize_feature_maps(model, test_loader, class_names, model_name, save_path,
                           target_modules=(nn.Conv2d, nn.MaxPool2d)):
    """
    EFFICIENT FEATURE MAP VISUALIZATION:
    Instead of plotting hundreds of noisy individual channels, this computes the
    mean activation across the channel dimension for each target layer.
    This produces a single, highly interpretable heatmap per layer showing
    where the network is focusing its attention.
    """
    print(f"  Visualizing mean feature activations: {model_name}")
    model.eval()
    feature_maps = OrderedDict()
    hooks = []

    # 1. Hook into target layers
    def make_hook(name):
        def fn(module, inp, out):
            # Grab the output tensor, detach it, and compute the mean across channels (dim=1)
            if isinstance(out, torch.Tensor) and out.dim() == 4:
                mean_activation = out.mean(dim=1, keepdim=True).detach().cpu()
                feature_maps[name] = mean_activation

        return fn

    for name, module in model.named_modules():
        if isinstance(module, target_modules):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # 2. Run a single image through
    images, labels = next(iter(test_loader))
    sample = images[:1].to(device)
    true_label = class_names[labels[0].item()]

    with torch.no_grad():
        _ = model(sample)

    for h in hooks:
        h.remove()

    if not feature_maps:
        print("    No feature maps captured. Check target_modules.")
        return

    # 3. Un-normalize the original input image for display
    inp_img = images[0].cpu().clone()
    for c in range(3):
        inp_img[c] = inp_img[c] * STD[c] + MEAN[c]
    inp_img = inp_img.clamp(0, 1).permute(1, 2, 0).numpy()

    # 4. Plotting (Grid layout based on number of captured layers)
    num_plots = len(feature_maps) + 1  # +1 for original image
    cols = 4
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    # Plot original image
    axes[0].imshow(inp_img)
    axes[0].set_title(f"Input ({true_label})", fontweight='bold')
    axes[0].axis('off')

    # Plot mean activation heatmaps
    for idx, (layer_name, fmap) in enumerate(feature_maps.items(), start=1):
        heatmap = fmap[0, 0].numpy()  # Extract the 2D spatial grid

        # Use 'magma' colormap for clear visual intensity
        im = axes[idx].imshow(heatmap, cmap='magma')

        short_name = layer_name.split('.')[-1] if '.' in layer_name else layer_name
        axes[idx].set_title(f"{short_name}\n({heatmap.shape[0]}x{heatmap.shape[1]})", fontsize=10)
        axes[idx].axis('off')

    # Hide any unused subplots
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')

    fig.suptitle(f"{model_name} — Mean Layer Activations", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved efficient feature maps to: {save_path}")


def visualize_stn(model, test_loader, class_names, save_path, n=6):
    """
    Cleanly visualizes the input and output of a Spatial Transformer Network (STN).
    Duck-types the model to ensure it exposes _last_input, _last_transformed, and _last_theta.
    """
    if not (hasattr(model, '_last_input') and hasattr(model, '_last_transformed') and hasattr(model, '_last_theta')):
        return

    print(f"  Visualizing STN geometric transformations...")
    model.eval()
    images, labels = next(iter(test_loader))
    images = images[:n].to(device)

    with torch.no_grad():
        _ = model(images)

    # Move tracked tensors to CPU for plotting
    inp = model._last_input[:n].cpu()
    transformed = model._last_transformed[:n].cpu()
    thetas = model._last_theta[:n].cpu()

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.5))
    fig.suptitle(f"{model.__class__.__name__} — STN Learned Transformations", fontsize=14, fontweight='bold')

    for i in range(n):
        true_label = class_names[labels[i].item()]

        for data, row_idx in [(inp[i], 0), (transformed[i], 1)]:
            img = data.clone()

            # Un-normalize using global MEAN and STD
            for c in range(3):
                img[c] = img[c] * STD[c] + MEAN[c]

            axes[row_idx, i].imshow(img.clamp(0, 1).permute(1, 2, 0).numpy())
            axes[row_idx, i].axis('off')

            # Labels
            if row_idx == 0:
                axes[row_idx, i].set_title(true_label, fontsize=11)
            if i == 0:
                axes[row_idx, i].text(-0.2, 0.5, "Input" if row_idx == 0 else "STN Output",
                                      transform=axes[row_idx, i].transAxes,
                                      fontsize=12, fontweight='bold', va='center', rotation=90)

        # Affine Matrix Data
        th = thetas[i]
        scale_x, scale_y = th[0, 0], th[1, 1]
        trans_x, trans_y = th[0, 2], th[1, 2]

        # Display the learned affine parameters below the image
        param_text = f"Zoom: ({scale_x:.2f}, {scale_y:.2f})\nPan: ({trans_x:.2f}, {trans_y:.2f})"
        axes[1, i].text(0.5, -0.15, param_text, transform=axes[1, i].transAxes,
                        ha='center', va='top', fontsize=9, color='#444444',
                        bbox=dict(facecolor='#f0f0f0', edgecolor='none', boxstyle='round,pad=0.3'))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for text
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    Saved STN visualization to: {save_path}")


import torch.nn.functional as F


def plot_gradcam(model, test_loader, class_names, save_path, target_layer=None):
    """
    Generates Grad-CAM (Gradient-weighted Class Activation Mapping) visualizations.
    Highlights the specific regions of the image that most strongly influenced
    the model's final prediction.
    """
    print(f"  Visualizing Grad-CAM...")
    model.eval()

    # 1. Automatically find the last Conv2d layer if none is provided
    if target_layer is None:
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module

    if target_layer is None:
        print("    Failed: Could not find a Conv2d layer for Grad-CAM.")
        return

    # 2. Set up hooks to grab activations (forward) and gradients (backward)
    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_full_backward_hook(backward_hook)

    # 3. Get a small batch of images (we'll plot the first 4)
    images, labels = next(iter(test_loader))
    images = images[:4].to(device)
    labels = labels[:4].to(device)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Grad-CAM: What drove the model's prediction?", fontsize=16, fontweight='bold')

    for i in range(len(images)):
        img_tensor = images[i:i + 1]  # Keep batch dimension
        true_label = labels[i].item()

        # ── Forward Pass ──
        model.zero_grad()
        output = model(img_tensor)
        pred_class = output.argmax(dim=1).item()

        # ── Backward Pass ──
        # We only want the gradients for the specific class the model predicted
        target = output[0, pred_class]
        target.backward(retain_graph=True)

        # ── Compute Grad-CAM ──
        # Global Average Pooling on the gradients to get 'weights' for each channel
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)

        # Weighted sum of the feature map activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # Discard negative influences

        # Normalize to [0, 1] for plotting
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Resize the tiny heatmap back up to the original image resolution
        cam = F.interpolate(cam, size=(img_tensor.shape[2], img_tensor.shape[3]),
                            mode='bilinear', align_corners=False)
        cam = cam[0, 0].cpu().detach().numpy()

        # ── Un-normalize Image for Display ──
        orig_img = img_tensor[0].cpu().clone()
        for c in range(3):
            orig_img[c] = orig_img[c] * STD[c] + MEAN[c]
        orig_img = orig_img.clamp(0, 1).permute(1, 2, 0).numpy()

        # ── Create Overlay ──
        # Map the 1D CAM to a 3D color map (jet)
        cmap = plt.get_cmap('jet')
        cam_colored = cmap(cam)[..., :3]

        # Blend the image and the heatmap
        overlay = 0.6 * orig_img + 0.4 * cam_colored
        overlay = np.clip(overlay, 0, 1)

        # ── Plotting ──
        # Top row: Original Image
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f"True: {class_names[true_label]}", fontsize=12)
        axes[0, i].axis('off')

        # Bottom row: Grad-CAM Overlay
        color = 'green' if true_label == pred_class else 'red'
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f"Pred: {class_names[pred_class]}", fontsize=12, color=color)
        axes[1, i].axis('off')

    # Clean up hooks so they don't slow down future standard forward passes
    handle_fw.remove()
    handle_bw.remove()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    Saved Grad-CAM visualization to: {save_path}")
# ============================================================================
# 8. PROFILING & HELPERS
# ============================================================================
def format_number(n):
    """Format large numbers with K/M/G suffixes for readability."""
    if n >= 1e9:
        return f"{n/1e9:.2f}G"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    else:
        return str(int(n))


def compute_flops(module, in_shape, out_shape):
    """
    Estimate FLOPs for a single layer based on its type and shapes.

    Each multiply-accumulate counts as 2 FLOPs.

    Args:
        module:    the nn.Module layer
        in_shape:  input tensor shape [B, C, H, W] or [B, features]
        out_shape: output tensor shape

    Returns:
        int: estimated FLOPs
    """
    if isinstance(module, nn.Conv2d):
        Cout = module.out_channels
        Cin  = module.in_channels // module.groups
        kH, kW = module.kernel_size
        Hout, Wout = out_shape[2], out_shape[3]
        flops = 2 * Cin * kH * kW * Cout * Hout * Wout
        if module.bias is not None:
            flops += Cout * Hout * Wout
        return flops

    elif isinstance(module, nn.Linear):
        flops = 2 * module.in_features * module.out_features
        if module.bias is not None:
            flops += module.out_features
        return flops

    elif isinstance(module, nn.BatchNorm2d):
        return 2 * out_shape[1] * out_shape[2] * out_shape[3]

    elif isinstance(module, nn.BatchNorm1d):
        return 2 * out_shape[1]

    elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.LeakyReLU)):
        return int(torch.tensor(out_shape[1:]).prod().item())

    elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
        k = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
        num_out = int(torch.tensor(out_shape[1:]).prod().item())
        return num_out * (k * k)

    elif isinstance(module, nn.AdaptiveAvgPool2d):
        Hin, Win = in_shape[2], in_shape[3]
        if isinstance(module.output_size, int):
            Hout = Wout = module.output_size
        else:
            Hout, Wout = module.output_size
        C = out_shape[1]
        window = (Hin // Hout) * (Win // Wout)
        return C * Hout * Wout * window

    elif isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Flatten, nn.Identity)):
        return 0

    else:
        return 0


def profile_model(model_fn, model_name, img_size=150):
    """
    Profile a model's parameters and FLOPs, layer by layer, grouped into blocks.

    Passes a dummy image through the model with hooks on every layer to
    capture input/output shapes, then computes params and FLOPs analytically.

    Layers are grouped into logical blocks (e.g., features.0–features.3 = Block 1)
    with subtotals for each block, so you can see at a glance where the cost is.

    Args:
        model_fn:    callable returning a fresh model
        model_name:  display name for the table header
        img_size:    input image size (default 150)

    Prints:
        A table with every layer, grouped by block, showing:
          - Layer name, type, output shape
          - Parameters and FLOPs (with % of total)
          - Block subtotals
          - Cumulative running totals
          - ◄ markers on heavy layers
    """
    model = model_fn()
    model.eval()

    # ── Register hooks on every leaf module ──
    layer_info = []

    def make_hook(name, module):
        def hook(mod, inp, out):
            in_shape = inp[0].shape if isinstance(inp, tuple) else inp.shape
            out_shape = out.shape if isinstance(out, torch.Tensor) else out[0].shape
            params = sum(p.numel() for p in mod.parameters(recurse=False))
            flops = compute_flops(mod, in_shape, out_shape)
            layer_info.append({
                'name': name, 'type': mod.__class__.__name__,
                'in_shape': list(in_shape), 'out_shape': list(out_shape),
                'params': params, 'flops': flops,
            })
        return hook

    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(make_hook(name, module)))

    # ── Forward pass ──
    dummy = torch.zeros(1, 3, img_size, img_size)
    with torch.no_grad():
        _ = model(dummy)
    for h in hooks:
        h.remove()

    # ── Compute totals ──
    total_params = sum(info['params'] for info in layer_info)
    total_flops  = sum(info['flops'] for info in layer_info)

    # ── Group layers into blocks ──
    # Detect block boundaries from layer names:
    #   "features.0", "features.1" → same parent "features"
    #   "block1.conv1", "block1.bn1" → same parent "block1"
    #   "classifier.0", "classifier.1" → same parent "classifier"
    def get_block_name(layer_name):
        """Extract the top-level block name from a dotted layer name."""
        parts = layer_name.split('.')
        if len(parts) >= 2:
            # For nn.Sequential like features.0, features.4 → group by features
            # But we want sub-blocks: features.0–3 = Block 1, features.4–7 = Block 2
            # So group by the first two parts if the parent is a Sequential
            return parts[0]
        return layer_name

    def get_conv_block_name(layer_name, layer_type):
        """
        Smarter grouping: detect Conv blocks within nn.Sequential.
        Each Conv2d starts a new sub-block within 'features'.
        """
        parts = layer_name.split('.')
        if parts[0] == 'features' and len(parts) >= 2:
            # Group features layers into conv blocks
            # A new block starts at each Conv2d
            return 'features'  # will be subdivided below
        return parts[0]

    # Build blocks: split 'features' at each Conv2d boundary
    blocks = []
    current_block_name = None
    current_block_layers = []
    conv_count = 0

    for info in layer_info:
        top_level = info['name'].split('.')[0]

        # Detect if this is a Conv2d inside features → start new block
        is_features = top_level == 'features'
        starts_new_conv_block = is_features and info['type'] == 'Conv2d'

        # Detect if top-level changed (e.g., features → pool → classifier)
        top_changed = (top_level != current_block_name) and not is_features

        if starts_new_conv_block or (top_changed and current_block_layers):
            # Save previous block
            if current_block_layers:
                blocks.append(current_block_layers)
            current_block_layers = [info]
            if starts_new_conv_block:
                conv_count += 1
                current_block_name = 'features'
            else:
                current_block_name = top_level
        else:
            if current_block_name is None:
                current_block_name = top_level
            current_block_layers.append(info)

    # Don't forget the last block
    if current_block_layers:
        blocks.append(current_block_layers)

    # ── Print header ──
    W = 120
    print(f"\n{'═'*W}")
    print(f"  MODEL PROFILE: {model_name}")
    print(f"  Input: (1, 3, {img_size}, {img_size})  |  "
          f"Total Params: {total_params:,}  |  Total FLOPs: {format_number(total_flops)}")
    print(f"{'═'*W}")

    print(f"  {'#':<4s} {'Layer Name':<35s} {'Type':<16s} "
          f"{'Output Shape':<20s} {'Params':>10s} {'%':>6s}  "
          f"{'FLOPs':>12s} {'%':>6s}  {'Cumul Params':>12s} {'Cumul FLOPs':>12s}")
    print(f"  {'─'*(W-2)}")

    cumul_params = 0
    cumul_flops  = 0
    layer_num    = 0

    for block_idx, block_layers in enumerate(blocks):
        block_params = sum(l['params'] for l in block_layers)
        block_flops  = sum(l['flops'] for l in block_layers)

        # Determine block label from the first layer's name
        first_name = block_layers[0]['name']
        first_type = block_layers[0]['type']
        top_level  = first_name.split('.')[0]

        # Create a readable block label
        if top_level == 'features' and first_type == 'Conv2d':
            block_label = f"CONV BLOCK {block_idx + 1}"
        elif 'block' in top_level:
            block_label = f"RESIDUAL {top_level.upper()}"
        elif top_level == 'classifier':
            block_label = "CLASSIFIER"
        elif top_level in ('pool', 'gap'):
            block_label = "POOLING"
        elif top_level.startswith('drop'):
            block_label = "DROPOUT"
        else:
            block_label = top_level.upper()

        # Print block header
        block_pct_p = (block_params / total_params * 100) if total_params > 0 else 0
        block_pct_f = (block_flops / total_flops * 100) if total_flops > 0 else 0
        print(f"  ┌─ {block_label} "
              f"(params: {format_number(block_params)} = {block_pct_p:.1f}%, "
              f"FLOPs: {format_number(block_flops)} = {block_pct_f:.1f}%)")

        # Print each layer in the block
        for li, info in enumerate(block_layers):
            layer_num += 1
            cumul_params += info['params']
            cumul_flops  += info['flops']

            shape_str = '×'.join(str(s) for s in info['out_shape'][1:])
            param_pct = (info['params'] / total_params * 100) if total_params > 0 else 0
            flop_pct  = (info['flops'] / total_flops * 100) if total_flops > 0 else 0

            # Mark heavy layers
            marker = ''
            if param_pct > 10:
                marker = ' ◄◄ HEAVY PARAMS'
            elif flop_pct > 10:
                marker = ' ◄◄ HEAVY COMPUTE'

            # Tree connector
            is_last = (li == len(block_layers) - 1)
            connector = '  └──' if is_last else '  ├──'

            # Truncate long names — show the last meaningful part
            name_str = info['name']
            if len(name_str) > 33:
                name_str = '...' + name_str[-30:]

            print(f"  {connector} {layer_num:<3d} {name_str:<30s} {info['type']:<16s} "
                  f"{shape_str:<20s} {info['params']:>10,} {param_pct:>5.1f}%  "
                  f"{format_number(info['flops']):>12s} {flop_pct:>5.1f}%  "
                  f"{format_number(cumul_params):>12s} {format_number(cumul_flops):>12s}"
                  f"{marker}")

        print(f"  │")

    # ── Total ──
    print(f"  {'─'*(W-2)}")
    print(f"  {'TOTAL':<56s} "
          f"{total_params:>10,} {100:>5.1f}%  "
          f"{format_number(total_flops):>12s} {100:>5.1f}%")
    print(f"{'═'*W}\n")

    return layer_info