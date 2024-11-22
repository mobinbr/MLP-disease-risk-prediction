# MLP-disease-risk-prediction

This repository provides a comprehensive framework for disease prediction using machine learning and deep learning. The project demonstrates two main approaches:

1. **Neural Network Implementation in NumPy**: A lightweight implementation of a feed-forward neural network without using a deep learning library.
2. **Neural Network Development with PyTorch**: A modern and scalable deep learning implementation.

---

## Notebooks and Scripts Overview

### 1. `disease_prediction.py`

This script provides a **custom implementation of a feed-forward neural network** for binary classification, demonstrating the fundamentals of deep learning from scratch. Key components include:

#### **Data Preprocessing**
- **Class: `DataPreprocessor`**
  - Handles resampling and splitting of the dataset into training, validation, and test sets.
  - Applies normalization to ensure features are on the same scale.
  - Uses `train_test_split` from scikit-learn for creating the training and validation datasets.

#### **Neural Network Implementation**
- **Class: `MLP` (Multilayer Perceptron)**
  - Defines the architecture with one hidden layer, weights initialized using a normal distribution, and biases initialized to zero.
  - **Activation Function**: Implements a sigmoid activation function for non-linearity.
  - **Forward Pass**: Propagates input through the network using matrix multiplication and activation functions.
  - **Backward Pass**: Computes gradients using the chain rule to update weights and biases. Includes:
    - Error term for output layer (`dz2`) and hidden layer (`dz1`).
    - Weight and bias updates using stochastic gradient descent (SGD).

#### **Training and Validation**
- **Class: `Trainer`**
  - Trains the model using mini-batch gradient descent.
  - Monitors loss and accuracy on both training and validation datasets at each epoch.
  - Provides visualization of training progress through loss and accuracy plots.

#### **Testing**
- **Class: `Tester`**
  - Evaluates the model on a test dataset using Binary Cross-Entropy loss and accuracy as metrics.
  - Outputs test results and example predictions for manual evaluation.

### 2. `disease_prediction_pytorch.ipynb`

This notebook demonstrates the PyTorch implementation of a similar disease prediction task:

#### **Builds a neural network using PyTorch.**

#### **Includes advanced deep learning features such as dropout and modular design.**
