# Neural Dependency Parser

A robust transition-based dependency parser implementation using both Perceptron and Multi-Layer Perceptron (MLP) approaches. This project demonstrates modern NLP techniques for syntactic parsing with comparison between traditional and neural methods.

## 🎯 Key Features

- Transition-based arc-eager and arc-standard dependency parsing
- Dual implementation with Perceptron and MLP models
- Feature extraction utilizing word embeddings
- Performance evaluation on Universal Dependencies treebanks

## 🔍 Technical Implementation

- **Parser Architecture**: Arc-eager/Arc-standard transition system with stack-buffer configurations
- **Feature Engineering**: 
  - Dense word embeddings fastText
  - POS tag embeddings
  - Dependency relation embeddings
  - Stack/Buffer positional features

- **Neural Network Details**:
  - 1-layer MLP with ReLU activation
  - Adam optimizer
  - Batch training
  - NLL loss

## 📈 Performance Metrics

| Model | UAS | LAS |
|-------|-----|-----|
| Perceptron | 71.8% | 67.4% |
| MLP | 79.3% | 75.9% |

## 🛠️ Tech Stack

- Python 3.8+
- PyTorch
- NumPy
- sklearn

## 🔗 Related Research

Based on techniques from:
- "A Fast and Accurate Dependency Parser using Neural Networks" (Chen & Manning, 2014)
- "Speech and Language Processing : An Introduction to Natural Language Processing" (Daniel Jurafsky and James Martin, 2008)
