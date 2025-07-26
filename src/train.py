# src/train.py
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os
import torch
import torch.nn as nn
import numpy as np

print("Starting model training...")

# 1. Load the California Housing dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure 'models' directory exists
os.makedirs('models', exist_ok=True)

# 2. Train a scikit-learn Linear Regression model
print("Training scikit-learn Linear Regression model...")
sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)

# Save the scikit-learn model
sklearn_model_path = os.path.join('models', 'sklearn_linear_regression_model.joblib')
joblib.dump(sklearn_model, sklearn_model_path)
print(f"Scikit-learn model saved to {sklearn_model_path}")

# Extract parameters
sklearn_coef = sklearn_model.coef_
sklearn_intercept = sklearn_model.intercept_

print(f"Scikit-learn Coefficients: {sklearn_coef}")
print(f"Scikit-learn Intercept: {sklearn_intercept}")

# 3. Create a single-layer PyTorch neural network and set weights
class SimpleLinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(SimpleLinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

input_dim = X_train.shape[1]
pytorch_model = SimpleLinearRegression(input_dim)

# Manually set weights and bias from scikit-learn model
with torch.no_grad():
    pytorch_model.linear.weight.copy_(torch.tensor(sklearn_coef).float().view(1, -1))
    pytorch_model.linear.bias.copy_(torch.tensor(sklearn_intercept).float())

# Save the PyTorch model (state_dict)
pytorch_model_path = os.path.join('models', 'pytorch_linear_regression_model.pth')
torch.save(pytorch_model.state_dict(), pytorch_model_path)
print(f"PyTorch model (state_dict) saved to {pytorch_model_path}")

print("Model training and saving complete.")