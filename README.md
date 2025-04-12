# Customer Churn Prediction

This project predicts whether a customer will churn using machine learning models. It uses the `Churn_Modelling.csv` dataset and compares the performance of different classifiers.

## Files

- `Churn_Modelling.csv` - Dataset with customer info and churn labels.
- `churn-model-hyperparamter-tuning-notebook.ipynb` - Jupyter Notebook with all code for training, testing, and evaluating models.

## Models Used

- Logistic Regression
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

## Results

- **XGBoost** performed the best based on F1 Score.
- **Random Forest** and **Gradient Boosting** also had strong results.
- Simpler models like **Logistic Regression** had lower performance but are easier to interpret.

## How to Use

1. Install the required libraries (e.g., `scikit-learn`, `xgboost`, `pandas`, etc.).
2. Open the Jupyter Notebook.
3. Run the cells to see data processing, model training, and evaluation.

