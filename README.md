# Multilabel Movie Genre Classification Using Movie Posters
A computer vision project. Multilabelling a dataset of movie poster to determine its genre using deep learning and movie poster images. The project investigates the effectiveness of transfer learning, augmentation strategies, and hybrid loss functions for multilabel visual classification.

---

# Overview

Movie posters often represent multiple genres simultaneously. This project aims to develop a model capable of predicting several genres from a single poster image.

The implementation is based on EfficientNetV2-M with additional experiments involving:

* progressive fine-tuning,
* augmentation strategies,
* hybrid multilabel loss functions, and
* genre co-occurrence learning.

---

# Model Architecture

## Backbone

* EfficientNetV2-M pretrained on ImageNet

## Classification Head

* Multi-layer perceptron (MLP)
* Batch normalization
* ReLU activation
* Dropout regularization

## Output

* 14 sigmoid outputs for multilabel genre prediction

---

# Training Pipeline

```text
Movie Posters
      ↓
Image Preprocessing & Augmentation
      ↓
EfficientNetV2-M Feature Extraction
      ↓
MLP Classification Head
      ↓
Sigmoid Activation
      ↓
Multilabel Genre Prediction
```

---

# Ablation Study

The notebook includes a structured ablation study to evaluate the contribution of several components within the training pipeline.

The experiments examine:

* augmentation strategies,
* progressive fine-tuning,
* hybrid loss functions,
* co-occurrence loss integration, and
* classification head configurations.

---

# Dataset

* Approximately 300 movie posters
* 14 genre labels
* Multilabel annotations

## Data Split

| Split      | Size |
| ---------- | ---- |
| Train      | 200  |
| Validation | 50   |
| Test       | 50   |

---

# Image Preprocessing

Image preprocessing and augmentation are implemented using Albumentations.

Applied transformations include:

* resizing,
* horizontal flipping,
* shift-scale-rotate,
* brightness and contrast adjustment, and
* normalization.

---

# Loss Functions

The training framework utilizes a hybrid multilabel loss function combining:

* SigmoidF1 Loss
* Focal Loss

The objective is to improve macro F1 performance while addressing class imbalance in genre distributions.

---

# Evaluation Metrics

Model performance is evaluated using:

* Macro F1 Score
* Jaccard Score
* Exact Match Accuracy
* Hamming Loss

---

# Technologies

* PyTorch
* TIMM
* Albumentations
* Scikit-learn
* NumPy
* Pandas
* Matplotlib

---

# Repository Structure

```bash
project/
│
├── dataset/
├── outputs/
├── splits/
│
├── ablation_v2.ipynb
├── architecture_diagram.png
└── README.md
```

---

# Installation

```bash
pip install torch torchvision timm albumentations \
scikit-learn pandas matplotlib openpyxl \
iterative-stratification
```

---

# Running the Project

```bash
jupyter notebook ablation_v2.ipynb
```

---

# Author

* angoota1
* anggota2
* Danella Zefanya Siahaan - 22/492877/TK/53953
