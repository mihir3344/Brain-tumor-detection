# Brain Tumor Classification using VGG16

This project classifies brain tumor images into four categories (glioma, meningioma, no_tumor, pituitary) using a Convolutional Neural Network (CNN) built on top of a pre-trained VGG16 model. The model utilizes transfer learning to achieve high accuracy.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training and Validation](#training-and-validation)
- [Results](#results)
- [Prediction](#prediction)
- [Installation](#installation)
- [Usage](#usage)

## Overview
This project leverages the pre-trained VGG16 model for feature extraction, freezing its convolutional layers, and adding custom dense layers for classification. The model is designed to classify brain tumor images into multiple categories with high accuracy.

## Dataset
**Source:** Kaggle - Brain Tumor Dataset

### Dataset Structure:
/kaggle/input/brain-tumor/  
    /Training  
        /glioma  
        /meningioma  
        /no_tumor  
        /pituitary  
    /Testing  
        /glioma  
        /meningioma  
        /no_tumor  
        /pituitary  

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

The model achieved the following results:
- **Training Accuracy**: 96%  
- **Validation Accuracy**: 96%

## Results
### Classification Report:
precision    recall  f1-score   support

glioma       1.00      0.91      0.95       291  
meningioma   0.94      0.93      0.94       306  
no_tumor     0.99      1.00      0.99       405  
pituitary    0.92      1.00      0.96       300  

accuracy                           0.96      1302  
macro avg       0.96      0.96      0.96      1302  
weighted avg    0.96      0.96      0.96      1302  

## Prediction
To make predictions on new images:  
- **Image Format**: Ensure images are resized to 224x224 pixels.  
- **Image Folder Structure**: Place images in the `predict_images/` folder structured as follows:  
/predict_images/  
    /glioma  
    /meningioma  
    /no_tumor  
    /pituitary  

- **Prediction Execution**: Run the prediction script to classify new images.

## Installation
Clone the repository:  
`git clone https://github.com/yourusername/brain-tumor-classification.git`

Install dependencies:  
`pip install -r requirements.txt`

## Usage
Place the image(s) you want to classify in the `predict_images/` folder.  
Run the prediction script:  
`python predict.py`  
View the predictions output on the console.
