# Predictive Analytics for Telco Customer Churn

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/144GQEmf4GCDoauPvRseTRNFpF9kYDZOB?usp=sharing)

---

## 1. Project Overview

This project presents a comprehensive analysis of customer churn for a telecommunications company. The primary goal is to identify the key factors driving customer churn and to develop a predictive model that can identify at-risk customers. By leveraging machine learning, the project aims to provide actionable insights that can help the company develop targeted retention strategies, reduce customer attrition, and improve overall profitability.

---

## 2. Business Problem

Customer churn is a critical issue in the highly competitive telecommunications industry. Acquiring new customers is significantly more expensive than retaining existing ones, making churn reduction a top priority. This project addresses the need for a data-driven approach to understand *why* customers leave and *who* is most likely to leave, enabling the business to shift from a reactive to a proactive retention model.

---

## 3. Dataset

The analysis is based on the **Telco Customer Churn** dataset, publicly available on Kaggle. The dataset contains 7,043 customer records with 21 attributes, including:

*   **Customer Demographics:** Gender, Senior Citizen status, Partner, and Dependents.
*   **Services Subscribed:** Phone Service, Multiple Lines, Internet Service, Online Security, Online Backup, Device Protection, Tech Support, and Streaming services.
*   **Account Information:** Tenure, Contract type, Payment Method, Paperless Billing, Monthly Charges, and Total Charges.

> **Source:** [Kaggle: Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### Data Exploration

The initial exploration of the dataset revealed important patterns:

<table>
  <tr>
    <td align="center" width="50%">
      <img alt="churn_distribution" src="https://github.com/user-attachments/assets/2caa3467-b9b0-458b-b624-f90026ee022b" width="100%" />
      <br/>
      <em>Distribution of Customer Churn - showing class imbalance between churned and retained customers</em>
    </td>
    <td align="center" width="50%">
      <img alt="tenure_distribution" src="https://github.com/user-attachments/assets/9a796b72-640a-4c42-aa43-08ae1c1c5302" width="100%" />
      <br/>
      <em>Distribution of Tenure - revealing a U-shaped distribution with peaks at low and high tenure values</em>
    </td>
  </tr>
</table>
---

## 4. Methodology

The project followed a structured data analysis workflow:

1.  **Exploratory Data Analysis (EDA):** The dataset was analyzed to uncover patterns, identify anomalies, and understand the relationships between different variables and churn. This included visualizing distributions and correlations.

2.  **Data Preprocessing & Feature Engineering:** The data was cleaned and prepared for modeling. This involved handling missing values, encoding categorical variables (One-Hot and Label Encoding), and scaling numerical features to prevent bias in the models.

3.  **Model Development:** Three different machine learning models were trained and evaluated:
    *   **Logistic Regression (with Lasso):** Chosen for its high interpretability and ability to serve as a strong baseline.
    *   **XGBoost Classifier:** A powerful gradient boosting model chosen for its high predictive accuracy.
    *   **Random Forest Classifier:** An ensemble model chosen for its robustness and ability to rank feature importance.

4.  **Model Evaluation:** Models were evaluated using a variety of metrics, including Accuracy, **Recall**, and **ROC AUC**. Recall was prioritized as the key metric to ensure the model effectively identifies the maximum number of actual churners.

### Analysis Visualizations

<table>
  <tr>
    <td align="center" width="50%">
      <img alt="correlation_matrix" src="https://github.com/user-attachments/assets/0ca27e7c-6314-41f1-b618-4123369a0e30" width="100%" />
      <br/>
      <em>Correlation Matrix showing relationships between features and churn probability</em>
    </td>
    <td align="center" width="50%">
      <img alt="feature_importance" src="https://github.com/user-attachments/assets/23fd5b13-d505-4422-9ad5-fe3fbe2a7372" width="100%" />
      <br/>
      <em>Top 15 Feature Importances for Churn Prediction (Random Forest) - Contract_Month-to-month is the most important predictor</em>
    </td>
  </tr>

  <tr>
    <td align="center" width="50%">
      <img alt="monthly_charges_by_churn" src="https://github.com/user-attachments/assets/6565c3f3-f32d-4c93-865e-1686d05b2122" width="100%" />
      <br/>
      <em>Monthly Charges by Churn Status - Churned customers tend to have higher monthly charges</em>
    </td>
    <td align="center" width="50%">
      <img alt="churn_by_contract" src="https://github.com/user-attachments/assets/57eb9713-ff99-47ef-aaf5-989260ae24f8" width="100%" />
      <br/>
      <em>Churn Rate by Contract Type - Month-to-month contracts show significantly higher churn rates</em>
    </td>
  </tr>
</table>

<p align="center">
  <img alt="turnure_vs_monthly_charges" src="https://github.com/user-attachments/assets/fcf01b01-4602-407c-b11f-d786aeb37cb1" width="70%" />
  <br/>
  <em>Tenure vs. Monthly Charges Analysis - Relationship between tenure, charges, and churn patterns</em>
</p>

---

## 5. Key Findings

The analysis revealed several key drivers of customer churn:

*   **Contract Type is Critical:** Customers on **month-to-month contracts** are significantly more likely to churn than those on one-year or two-year contracts.
*   **Lack of Support Drives Churn:** The absence of services like **Tech Support**, **Online Security**, and **Online Backup** is strongly correlated with a higher churn rate.
*   **Tenure Matters:** New customers with low tenure are more likely to leave, while long-term customers are more loyal.
*   **Payment Method Impact:** Customers using **Electronic check** as a payment method show a higher propensity to churn.

Based on the evaluation, the **Tuned Logistic Regression model** was selected as the final model due to its superior **Recall score of 79.41%**, aligning best with the business goal of identifying the most at-risk customers.

### Model Performance

<!-- ROC Curves: 3 images in one row -->
<table>
  <tr>
    <td align="center" width="33%">
      <img alt="roc_randomforest" src="https://github.com/user-attachments/assets/65b7ba06-f465-4b58-b11f-049dc8a665df" width="100%" />
      <br/>
      <em>ROC Curve - Tuned Random Forest Classifier (AUC = 0.82)</em>
    </td>
    <td align="center" width="33%">
      <img alt="roc_xgboost" src="https://github.com/user-attachments/assets/34404ce7-eba2-4042-bcd9-72da34959dbb" width="100%" />
      <br/>
      <em>ROC Curve - Tuned XGBoost Classifier (AUC = 0.82)</em>
    </td>
    <td align="center" width="33%">
      <img alt="roc_logistic" src="https://github.com/user-attachments/assets/81f651f0-5471-4329-ab8f-5681c9bdb3a2" width="100%" />
      <br/>
      <em>ROC Curve - Tuned Logistic/Lasso Regression (AUC = 0.82)</em>
    </td>
  </tr>
</table>

<br/>

<!-- Confusion Matrices: 3 images in one row -->
<table>
  <tr>
    <td align="center" width="33%">
      <img alt="confusion_matrix_randomforest" src="https://github.com/user-attachments/assets/bd12af33-0893-4939-b581-5d7cc292943d" width="100%" />
      <br/>
      <em>Confusion Matrix - Tuned Random Forest Classifier</em>
    </td>
    <td align="center" width="33%">
      <img alt="confusion_matrix_xgboost" src="https://github.com/user-attachments/assets/8e1ab92d-a8af-4dfb-9311-a953cfed457f" width="100%" />
      <br/>
      <em>Confusion Matrix - Tuned XGBoost Classifier</em>
    </td>
    <td align="center" width="33%">
      <img alt="confusion_matrix_logistic" src="https://github.com/user-attachments/assets/76e2d3f7-db68-4ed2-8298-dcf7b78bebe6" width="100%" />
      <br/>
      <em>Confusion Matrix - Tuned Logistic/Lasso Regression</em>
    </td>
  </tr>
</table>

---

## 6. Business Recommendations

Based on the findings, the following actionable recommendations are proposed:

1.  **Incentivize Long-Term Contracts:** Develop targeted campaigns to encourage month-to-month customers to switch to annual or two-year contracts.
2.  **Promote Value-Added Services:** Increase the adoption of Tech Support, Online Security, and Online Backup by bundling them or offering promotional trials.
3.  **Focus on Early-Stage Customers:** Implement a proactive onboarding program for new customers to address their needs and improve their initial experience.
4.  **Review Payment Processes:** Investigate why customers using electronic checks are more likely to churn and encourage the use of more stable, automated payment methods.

---

## 7. Tools and Libraries

This project was implemented in Python 3 and relied on the following key libraries:

*   **Data Manipulation:** `pandas`, `numpy`
*   **Data Visualization:** `matplotlib`, `seaborn`
*   **Machine Learning:** `scikit-learn`, `xgboost`
*   **Dataset Access:** `opendatasets`

---

## 8. Acknowledgments

*   **Dataset:** The Telco Customer Churn dataset was provided by IBM and is available on Kaggle.
*   **AI Assistance:** The structure and content of this analysis were developed with assistance from Gemini, an AI model by Google.

---

## 9. Repository Links

*   **GitHub Repository:** [https://github.com/nyakabawurr-boop/Telco-Customer-Churn-Analysis](https://github.com/nyakabawurr-boop/Telco-Customer-Churn-Analysis)
*   **Kaggle Dataset:** [https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
*   **Google Colab Notebook:** [https://colab.research.google.com/drive/144GQEmf4GCDoauPvRseTRNFpF9kYDZOB](https://colab.research.google.com/drive/144GQEmf4GCDoauPvRseTRNFpF9kYDZOB?usp=sharing)

---

**Course:** ISOM 835 - Predictive Analytics Modelling  
**Project:** Telco Customer Churn Analysis
