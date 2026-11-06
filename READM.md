# Customer Churn Prediction & Analytics Dashboard

## Project Overview
An end-to-end Machine Learning project that predicts whether a telecom customer is likely to leave a service. The project analyzes customer demographics, subscription details, and billing information to identify potential churn risks.

## Business Problem
Customer churn affects business revenue and customer retention. This project helps identify customers who may leave so businesses can take proactive retention measures.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Jupyter Notebook

## Project Workflow
1. Data loading and cleaning
2. Exploratory Data Analysis (EDA)
3. Categorical feature encoding
4. Feature scaling
5. Train-test split
6. Logistic Regression and Decision Tree model training
7. Model evaluation and comparison
8. Interactive prediction dashboard using Streamlit

## Model Performance

Selected Model: Balanced Logistic Regression

| Metric | Result |
|---|---:|
| Accuracy | 74% |
| Churn Precision | 51% |
| Churn Recall | 78% |
| Churn F1-score | 61% |

The balanced model prioritizes identifying more actual churn customers.

## Dashboard Features
- Customer profile input
- Subscription and billing details
- Churn prediction
- Estimated churn probability
- Customer retention insights

## Project Structure
```text
Customer_Churn/
├── data/
├── notebooks/
│   └── churn_analysis.ipynb
├── app.py
├── churn_model.pkl
├── scaler.pkl
├── model_columns.pkl
├── requirements.txt
└── README.md
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Dataset
IBM Telco Customer Churn sample dataset.

## Project Objective
To apply data analysis and machine learning techniques to customer retention problems through an interactive dashboard.