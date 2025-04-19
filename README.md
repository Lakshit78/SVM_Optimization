# SVM Letter Recognition Classifier

A machine learning project that optimizes Support Vector Machine (SVM) models for the UCI Letter Recognition dataset. This implementation evaluates multiple train-test splits to identify the most effective SVM configuration for letter classification.

## Overview

This project:
- Loads the Letter Recognition dataset (26 classes, A-Z)
- Creates 10 different train-test splits
- Optimizes SVM hyperparameters for each split using RandomizedSearchCV
- Compares results across splits to find the optimal configuration
- Generates visualizations of the convergence process
- Saves detailed analytics about the experiment

## Features

- **Multiple Training Samples**: Creates 10 different train-test splits for robust model evaluation
- **Hyperparameter Optimization**: Uses RandomizedSearchCV to find optimal SVM settings
- **Performance Visualization**: Generates convergence graphs to visualize the optimization process
- **Results Analysis**: Creates a comprehensive CSV report of model performance

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
```

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/svm-letter-recognition.git
cd svm-letter-recognition
pip install -r requirements.txt
```

## Usage

Run the main script:

```bash
python letter_recognition_svm.py
```

This will:
1. Download the Letter Recognition dataset from OpenML
2. Perform hyperparameter optimization on multiple splits
3. Save results to CSV and generate visualization files

## Output Files

- `svm_optimization_results.csv`: Table of results for each sample
- `convergence_graph.png`: Visualization of the optimization process for the best performing model
- `data_analytics.txt`: Summary of dataset characteristics and best model configuration

## Performance

The model typically achieves accuracy between 85-95% on the letter recognition task, depending on the specific train-test split and optimized parameters.

## Customization

You can modify the optimization parameters by editing the `param_grid` dictionary:

```python
param_grid = {
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],  # Add more kernels
    'C': [0.1, 1, 10, 100],                         # Adjust regularization values
    'gamma': ['scale', 'auto', 0.1, 0.01]           # Modify gamma values
}
```

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.
