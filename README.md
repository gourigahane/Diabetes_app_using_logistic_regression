# 🔍 Logistic Regression using Supervised Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-success?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

</div>

---

## 📌 Overview

This project demonstrates the implementation of **Logistic Regression**, a widely used **Supervised Machine Learning** algorithm for classification tasks.

The notebook covers the complete machine learning workflow including data preprocessing, model training, prediction, evaluation, and visualization using Python and Scikit-learn.

The primary goal of this project is to understand how classification models work and how machine learning can be applied to solve real-world prediction problems.

---

## 🤖 About Logistic Regression

Logistic Regression is a supervised learning algorithm used to classify data into categories.  
Unlike Linear Regression, it predicts probabilities instead of continuous values.

### Logistic Function

\[
P(Y=1)=\frac{1}{1+e^{-z}}
\]

Where:

- **P(Y=1)** → Probability of belonging to a class
- **e** → Euler’s constant
- **z** → Linear combination of input features

---

## ⚙️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Pandas | Data Manipulation |
| NumPy | Numerical Computing |
| Matplotlib | Data Visualization |
| Seaborn | Statistical Visualization |
| Scikit-learn | Machine Learning |

---

## 📂 Project Workflow

- Import required libraries
- Load and explore the dataset
- Perform data preprocessing
- Split dataset into training and testing sets
- Train Logistic Regression model
- Make predictions
- Evaluate model performance
- Visualize results

---

## 💻 Implementation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

model = LogisticRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)
