# Brain Tumor Classification using VGG16

This project classifies brain tumor images into four categories (glioma, meningioma, no_tumor, pituitary) using a Convolutional Neural Network (CNN) built on top of a pre-trained VGG16 model. The model utilizes transfer learning to achieve high accuracy.

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training and Validation](#training-and-validation)
5. [Results](#results)
6. [Prediction](#prediction)
7. [Installation](#installation)
8. [Usage](#usage)

## Overview
This project leverages the pre-trained VGG16 model for feature extraction, freezing its convolutional layers, and adding custom dense layers for classification. The model is designed to classify brain tumor images into multiple categories with high accuracy.

## Dataset
**Source**: Kaggle - Brain Tumor Dataset

## Model Architecture
The model is built using the following key components:
- **VGG16 as Base**: Pre-trained weights on ImageNet.
- **Global Average Pooling**: Used to reduce the dimensionality of the feature maps.
- **Custom Dense Layers**: Added for classification of the tumor types.
- **Dropout**: Included to prevent overfitting during training.
- **Softmax Output Layer**: For multi-class classification (glioma, meningioma, no_tumor, pituitary).

## Training and Validation
- **Batch Size**: 32
- **Image Size**: 224x224 pixels
- **Epochs**: 20
- **Optimizer**: Adam with a learning rate of 1e-4
- **Loss Function**: Categorical Cross-Entropy
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

### Model Performance:
- **Training Accuracy**: 96%
- **Validation Accuracy**: 96%

## Results
### Classification Report:

| **Class**    | **Precision** | **Recall** | **F1-Score** | **Support** |
|--------------|---------------|------------|--------------|-------------|
| glioma       | 1.00          | 0.91       | 0.95         | 291         |
| meningioma   | 0.94          | 0.93       | 0.94         | 306         |
| no_tumor     | 0.99          | 1.00       | 0.99         | 405         |
| pituitary    | 0.92          | 1.00       | 0.96         | 300         |

- **Accuracy**: 0.96
- **Macro Average**: 0.96
- **Weighted Average**: 0.96

## Prediction
To make predictions on new images:

1. **Image Format**: Ensure images are resized to 224x224 pixels.
2. **Image Folder Structure**: Place images in the `predict_images/` folder structured as follows:
    ```
    /predict_images/
      /glioma
      /meningioma
      /no_tumor
      /pituitary
    ```
3. **Prediction Execution**: Run the prediction script to classify new images.
    ```bash
    python predict.py
    ```
4. **View the predictions output** on the console.

## Installation
Clone the repository:
```bash
git clone https://github.com/yourusername/brain-tumor-classification.git

