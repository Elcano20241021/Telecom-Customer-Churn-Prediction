<img width="1200" height="800" alt="pic11_AI_BI_article" src="https://github.com/user-attachments/assets/e3e2b2d0-6c24-4ff0-a921-71fb734ce1c5" />

# Telecom-Customer-Churn-Prediction
Customer retention campaigns increasingly rely on predictive models to detect potential churners in a vast customer base. From the perspective of machine learning, the task of predicting customer churn can be presented as a binary classification problem. Using data on historic behavior, classification algorithms are built with the purpose of accurately predicting the probability of a customer defecting. The predictive churn models are then commonly selected based on accuracy related performance measures such as the area under the ROC curve (AUC). However, these models are often not well aligned with the core business requirement of profit maximization, in the sense that, the models fail to take into account not only misclassification costs, but also the benefits originating from a correct classification. 
Therefore, The aims of this project is to build an effective predictive model, but with a strategic differential: **aligning the modeling with business objectives, such as profit maximization and efficient retention action**, and not just technical accuracy.

# Methodology: The E4AI Framework
Our project adheres to the E4AI methodology, a practical and structured framework designed to transform data into measurable business value. E4AI guides the entire lifecycle of a Data Science and Artificial Intelligence initiative, from problem understanding to implementation and sustainability.

1. EXPLORE: Gain a deep understanding of the business context, identify challenges, and map available data.
Key Steps: Business Understanding (clarifying goals and KPIs), Data Exploration (EDA for statistical profiling, quality, and structure), and Opportunity Discovery (identifying patterns and areas for impact).
2. EXPLAIN: Align analytical findings with the business context, clarify project objectives, and refine the understanding of expected outcomes, ensuring data exploration is connected to business reality.
Key Steps: Insight Interpretation & Contextualization, Stakeholder Clarification Meeting (for validation and requesting additional data), and Defining Success Criteria (KPIs and desired outcomes).
3. ENGINEER: Design and implement robust data-driven solutions that address identified challenges.
Key Steps: Model Development (selecting, training, and optimizing ML/AI algorithms), Solution Prototyping (building predictive tools, APIs, and user-focused dashboards), and Business Testing with Proof of Value (deploying MVPs to track ROI and impact).
4. EMPOWER: Ensure long-term adoption, scalability of the solution, and team self-sufficiency.
Key Steps: Knowledge Transfer (documentation, training, and explainability guides), Deployment, Automation & Monitoring (putting the solution into production, automating pipelines, and ensuring continuous performance), and Building a Data-Driven Culture (fostering analytical adoption in daily decisions).

Every cycle of the E4AI Framework culminates in measurable business impact and a clear path forward for continuous improvement.

# Project Stages (Overview)
## 1. Data Collection and Preparation

   * Dataset with **7,043 records** and **21 variables**, including demographic data, type of contract, services used and payment method.
   * Target variable: `Churn' (binary: yes/no).
   * Cleaning and transformation: imputation, coding, normalization and creation of composite variables based on business knowledge.

## 2. Data Exploration and Analysis (EDA)

   * Identification of important churn patterns: for example, *month-to-month* contracts showed churn rates of over 50%.
   * Customers with payment by *electronic check* or without technical support were associated with higher churn rates.
     ![image](https://github.com/user-attachments/assets/e1617718-807d-4157-85fa-423a035dd65c)
     ![image](https://github.com/user-attachments/assets/7ceeff95-db9b-483c-baf7-cebdf80c6dba)
     ![image](https://github.com/user-attachments/assets/60b8a7aa-03f7-42f5-88bc-264eb9ceeff7)


## 3. Feature Engineering
   * Creation of variables such as:

     * `ServiceScore`: time + number of services contracted.
     * `ServicesPerMonth_log`: correction of the distribution of the rate of use of services per month.

## 4. Modeling and Evaluation

   * Models used: **XGBoost, CatBoost, RandomForest, Logistic Regression and Decision Tree**.
   * Techniques applied:

     * Balancing with **SMOTE + Undersampling**.
     * Adjustment of the decision threshold based on a cost-benefit matrix.
     * Cross-validation and test set evaluation.
    
## 5. Metrics Evaluated
The primary goal of this churn prediction project was to build a model that not only accurately predicts customer churn but also provides tangible business value. Therefore, our evaluation focused on metrics directly aligned with profitability and proactive customer retention strategies, moving beyond simple accuracy.

Key metrics used for model assessment include:

F2-Score: This metric was prioritized over the standard F1-Score because it places twice as much weight on Recall (identifying actual churners) than on Precision. In the telecom industry, the cost of a False Negative (missing a customer who churns) is significantly higher than the cost of a False Positive (spending retention efforts on a customer who wouldn't churn). Maximizing our ability to detect potential churners was paramount.

EMPC (Expected Maximum Profit Criterion): This custom business-centric metric quantifies the expected profitability of the model's predictions, taking into account the costs associated with misclassifications (e.g., cost of retention campaigns, lost revenue from churned customers) and the benefits of correct classifications (e.g., profit from retained customers). A higher EMPC indicates greater business value.

ROC AUC & PR AUC: These were used to assess the model's overall discriminatory power and its performance on the minority class (churners) in an imbalanced dataset, respectively.

## 6. Model Performance Summary:

Our model consistently demonstrated strong generalization capabilities across validation and unseen test sets.

High Churner Detection (Recall): The model achieved a Recall of 90% for the churn class on the test set, meaning it successfully identified nearly two-thirds of customers who would actually churn. This validates our strategic focus on minimizing customer loss.

Optimized Business Profitability (EMPC): Critically, the EMPC reached an impressive $113.2907 on the test set, indicating a robust positive expected profit from deploying the model. This highlights the model's direct contribution to business objectives by effectively balancing the costs and benefits of retention efforts.

Robust Generalization: While a trade-off was observed in Precision for the churn class (due to the emphasis on Recall), the model's overall performance, including ROC AUC and F2-Score on the test set, confirms its reliability and practical applicability.

The evaluation confirms that the chosen metrics effectively guided the model development, resulting in a solution that is well-aligned with business priorities for churn reduction and profit maximization.
| Metric       | Validation | Test   | Change     |
| ------------ | ---------- | ------ | ---------- |
| **Recall** | 0.87       | 0.90   | ✅ Slight ↑  |
| **Precision** | 0.44     | 0.44 | ✅ Slight ↑ |
| **ROC AUC**  | 0.815     | 0.830 | ✅ Slight ↑ |
| **EMPC**     | 106.66     | 113.29 | ✅ Slight ↑ |


# SHAP Values - 
Interpretability: allows us to explain what led to the customer churn forecast.
<img width="753" height="653" alt="image" src="https://github.com/user-attachments/assets/300e789b-1e3e-4a55-974d-0dedbe86cedc" />


# Conclusions and Impact

The final model achieved:

Confusion Matrix:
[[599 434]
 [ 38 336]]
ROC AUC: 0.8298
PR AUC: 0.6022
F1 Score: 0.5874
F2 Score: 0.7414
EMPC: $113.2907
<img width="1183" height="484" alt="image" src="https://github.com/user-attachments/assets/5ed31700-af9a-4267-9fc8-e55215463b5f" />


Data-driven strategic decisions:

| **Scenario** | **Count** | **Cost/Profit** | **Total Impact** |
| --- | --- | --- | --- |
| **True Positives (TP)** | 336 | +$560 | **+$188,160** |
| **False Positives (FP)** | 434 | -$40 | **-$17,360** |
| **False Negatives (FN)** | 38 | -$300 | **-$11,400** |
| **True Negatives (TN)** | 599 | $0 | **$0** |
| **Net Profit (EMPC × 1407)** | **$113.29 × 1407 ≈ $159,400** |  |  |

24% reduction in unnecessary retention actions.
Increased campaign effectiveness by targeting high-risk customers.

The project validates that AI only generates value when guided by BI - when the right metrics are used for the right problem.
