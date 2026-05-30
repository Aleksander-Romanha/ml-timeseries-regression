# Time Series Forecasting: Linear Models & Extreme Learning Machines (ELM)

This repository contains a machine learning project focused on forecasting the future values of a chaotic time series. It compares the performance of classical Linear Predictors (Ordinary Least Squares) against Non-Linear mappings using Extreme Learning Machines (ELM) with Ridge Regression.

## 📌 Project Overview

The objective of this project is to model and predict future values of the **Mackey-Glass time series**, a well-known benchmark associated with a continuous, non-linear, and chaotic dynamical system.

The prediction task consists of estimating the future value of the series $y(n) = x(n+L)$ (where $L$ is the prediction horizon) given a vector of $K$ past observations.

### Key Highlights:
- **Linear Prediction:** Implementing a baseline predictor using Ordinary Least Squares (OLS) and performing grid search to find the optimal number of past samples ($K$).
- **Extreme Learning Machines (ELM):** Projecting input features into a higher-dimensional space using Hyperbolic Tangent ($\tanh$) mappings.
- **Regularization:** Utilizing **Ridge Regression** (L2 Regularization) to compute the output layer weights of the ELM, preventing overfitting in the expanded feature space.
- **Hyperparameter Tuning:** Comprehensive grid search to find the optimal network architecture (number of hidden nodes $V$) and regularization strength ($\lambda$).

## 🛠️ Technologies Used
- **Python 3**
- **NumPy** (Linear algebra, matrix operations)
- **Pandas** (Data manipulation)
- **Scikit-Learn** (Baseline models, metric evaluation)
- **Matplotlib** (Data visualization)

## 📊 Dataset
The dataset (`mackeyglass.csv`) contains generated values from the Mackey-Glass equation. To preserve the temporal dynamics and prevent data leakage, the dataset is partitioned chronologically into Training, Validation, and independent Test sets (strict holdout).

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/time-series-forecasting-elm.git
   cd time-series-forecasting-elm
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   # source venv/bin/activate    # On macOS/Linux
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Open the Jupyter Notebook:
   ```bash
   jupyter notebook efc1_linear_regression_elm.ipynb
   ```

## 📈 Results
- The **Linear Predictor** establishes a strong baseline, demonstrating the linear correlations in the immediate past samples.
- The **Extreme Learning Machine (ELM)** successfully captures the non-linear chaotic dynamics of the series. By optimizing the hidden layer size and applying L2 regularization, the ELM achieves a superior Root Mean Squared Error (RMSE) on the unseen test set compared to the linear baseline.
