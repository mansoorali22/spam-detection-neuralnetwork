# Spam Detection using Word2Vec and Neural Networks

## Overview

This project focuses on building a **spam detection system** using a **two-layer deep neural network** and **custom-trained Word2Vec embeddings**. The goal is to classify SMS messages as either spam or not spam by learning vector representations of words and feeding them into a neural network for prediction.

The project is divided into two parts:

1. **Word2Vec Model Implementation** – Implements word embeddings using the skip-gram model with negative sampling, without relying on any external libraries.
2. **Neural Network for Classification** – A feed-forward neural network is constructed to perform binary classification using the learned embeddings.

## Dataset

The dataset used is the [Spam or Not Spam Dataset](https://www.kaggle.com/datasets/ozlerhakan/spam-or-not-spam-dataset). It consists of SMS messages labeled as either spam (`1`) or not spam (`0`). The dataset is publicly available on Kaggle and contains the following columns:

- `label`: The target label (1 for spam, 0 for not spam)
- `text`: The SMS message content

## Project Structure

This project is submitted as two separate Jupyter Notebooks:

### 1. Word2Vec Notebook

**Filename:** `spam_detection_word2vec.ipynb`

**Tasks Covered:**
- **Data Preprocessing:**
  - Lowercasing, punctuation removal, tokenization
  - Handling class imbalance using undersampling or oversampling
- **Vocabulary Construction:**
  - Mapping each word to a unique index
- **Word2Vec Training:**
  - Implemented skip-gram with negative sampling
  - Embedding size set to 10
  - Trained using stochastic gradient descent
  - Loss computed using logistic regression with sigmoid activation

### 2. Neural Network Notebook

**Filename:** `spam_detection_nn.ipynb`

**Tasks Covered:**
- **Input Construction:**
  - For each message, 12 words are selected (padded or truncated)
  - Each word is represented using the 10-dimensional embedding from Word2Vec
- **Model Architecture:**
  - **Input Layer:** 12 words × 10-dimensional embeddings (120 input features)
  - **Hidden Layer:** 2 nodes, each of size 8 (using activation functions)
  - **Output Layer:** 1 node (binary output using sigmoid)
- **Training Procedure:**
  - Implemented forward pass and backward pass from scratch
  - Used binary cross-entropy loss and stochastic gradient descent
- **Evaluation:**
  - Metrics computed: Accuracy, Precision, Recall, F1 Score
  - Confusion matrix generated and visualized using matplotlib

## Evaluation Metrics

After training, the model's performance is evaluated using the following metrics:
- **Accuracy:** Measures overall correctness
- **Precision:** True positives over predicted positives
- **Recall:** True positives over actual positives
- **F1 Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** Provides a visual breakdown of correct and incorrect classifications

## Tools and Libraries Used

To comply with restrictions, **no predefined libraries like Gensim, TensorFlow, PyTorch, or Scikit-learn were used** for Word2Vec or neural network implementation. Only basic Python libraries such as:
- `NumPy` – For numerical operations
- `Matplotlib` – For visualization (confusion matrix)

## Conclusion

This project demonstrates how foundational concepts like Word2Vec embeddings and neural networks can be implemented from scratch to solve a real-world text classification problem. It offers a comprehensive learning experience in both natural language processing and deep learning fundamentals.
