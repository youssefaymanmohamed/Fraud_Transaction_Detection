# Credit Card Fraud Detection

This repository contains code for a credit card fraud detection system. The project includes data preprocessing, visualization, and the implementation of various machine learning models to detect fraudulent credit card transactions.

## Table of Contents
- [Import Libraries](#import-libraries)
- [Data Collection and Statistical Analysis](#data-collection-and-statistical-analysis)
- [Data Cleaning](#data-cleaning)
- [Data Visualization](#data-visualization)
- [Data Preprocessing and Encoding](#data-preprocessing-encoding)
- [Machine Learning Models](#machine-learning-models)
  - [Logistic Regression](#logistic-regression)
  - [Decision Tree](#decision-tree)
  - [Random Forest](#random-forest)
  - [AdaBoost](#adaboost)
  - [K-Neighbors Classifier](#k-neighbors-classifier)
- [Conclusion](#conclusion)

## Import Libraries

This section imports the necessary libraries for data analysis and machine learning, including NumPy, Pandas, Matplotlib, Seaborn, and various scikit-learn modules.

## Data Collection and Statistical Analysis

Data is loaded from the "creditcard.csv" file, and a brief overview of the dataset is provided, including descriptive statistics and data types.

## Data Cleaning

This section checks for null values and handles duplicate rows, ensuring the data is clean and ready for analysis.

## Data Visualization

Data visualization is a crucial step in understanding the dataset. It includes a correlation heatmap and scatter plots to visualize relationships between features and the target variable.

## Data Preprocessing and Encoding

Data preprocessing involves scaling the features using the RobustScaler and handling class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).

## Machine Learning Models

### Logistic Regression

A logistic regression model is trained and evaluated on the dataset, providing train and test scores along with a confusion matrix and classification report.

### Decision Tree

A decision tree classifier is implemented with specified hyperparameters. Model performance is evaluated with train and test scores, a confusion matrix, and a classification report.

### Random Forest

The random forest classifier is employed with hyperparameters. Training and testing scores, a confusion matrix, and a classification report are presented.

### AdaBoost

An AdaBoost classifier is trained and evaluated, providing train and test scores, a confusion matrix, and a classification report. K-Fold cross-validation results are also shown.

### K-Neighbors Classifier

The K-Neighbors Classifier is trained and evaluated with train and test scores, a confusion matrix, and a classification report.

## Conclusion

This README provides an overview of the credit card fraud detection project, including data analysis, data preprocessing, and the implementation and evaluation of various machine learning models. The code and comments in the repository cover each step comprehensively, allowing you to understand and reproduce the analysis.

For more details, please refer to the Jupyter Notebook or Python script in this repository.

---

**Note**: Please make sure to replace placeholders with specific information regarding your project. Additionally, consider adding more information about the dataset source, installation instructions, and potential improvements or future work.
