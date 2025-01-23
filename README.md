# Predicting Employee Attrition in an Organization

## Problem Statement
**“Predicting Employee Attrition in an Organization”**  

This project aims to predict employee attrition and identify factors influencing employee turnover using data analytics and machine learning. It provides actionable insights to help HR departments proactively address employee retention challenges.

---

## Objectives
1. Identify major determinants of employee attrition.  
2. Analyze demographic, job, compensation, and engagement traits.  
3. Build and evaluate predictive models to determine employee attrition.  
4. Address dataset issues like missing values, class imbalance, and feature scaling.  
5. Provide actionable insights for HR strategies based on feature importance analysis.  

---

## Dataset Overview
The dataset contains 1470 rows and 35 columns, with features categorized into:  
- **Demographics**: Age, Gender, Marital Status  
- **Work Details**: Job Role, Department, Years at Company, etc.  
- **Compensation**: Monthly Income, Stock Option Level, etc.  
- **Satisfaction & Performance**: Job Satisfaction, Performance Rating, etc.  
- **Work-Life Balance & Training**: Work-Life Balance, Training Times Last Year  
- **Attrition**: Binary target variable (Yes/No)  

---

## Data Preprocessing
1. **Removed Redundant Columns**: Dropped "EmployeeCount," "EmployeeNumber," "StandardHours," and "Over18."  
2. **Checked Missing Values**: None were found.  
3. **Categorical Encoding**: Used `StringIndexer` and `OneHotEncoder` for categorical features.  
4. **Feature Engineering**:  
   - Created `Tenure` = `YearsAtCompany - YearsInCurrentRole`.  
   - Added `IncomeToJobLevelRatio` = `MonthlyIncome / JobLevel`.  
5. **Feature Selection**: Retained features with correlation ≥ 0.03.  
6. **Feature Scaling**: Applied MinMaxScaler to normalize features.  
7. **Stratified Data Split**: Ensured class balance for training and testing sets.

---

## Tools and Techniques Used
| **Criteria**               | **Details**                                  |
|-----------------------------|----------------------------------------------|
| **Hardware Configuration** | Windows 11, AMD Ryzen 5, 8GB RAM, GTX 1650  |
| **Software Configuration** | Google Colab, TPU v2-8 runtime               |
| **Big Data Tools**          | PySpark 3.5.3                               |
| **Libraries**               | NumPy, Pandas, sklearn, pyspark.ml, pyspark.sql |
| **Visualization Tools**     | Matplotlib, Seaborn                         |

---

## Implementation & Methodology
1. **Spark Session Initialization**: Created a PySpark session for processing the dataset.  
2. **Data Loading**: Imported the dataset using `spark.read.csv()`.  
3. **EDA**: Performed statistical summaries and visualizations.  
   - Used histograms, box plots, and correlation heatmaps.  
4. **Feature Engineering**: Encoded categorical features, created derived features, and normalized numerical features.  
5. **Model Building**: Trained and tested Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting models.  
6. **Model Evaluation**: Used metrics like accuracy, precision, recall, F1-score, and ROC-AUC to evaluate models.  
7. **Hyperparameter Tuning**: Employed `CrossValidator` for optimizing model parameters.  
8. **Visualization**: Plotted confusion matrices, ROC-AUC curves, and model comparisons.

---

## Comparative Analysis of PySpark Models
### Before Hyperparameter Tuning
| **Model**            | **Accuracy** | **Precision** | **Recall** | **F1-Score** | **AUC** |
|-----------------------|--------------|---------------|------------|--------------|---------|
| Logistic Regression   | 89.79        | 88.89         | 89.79      | 88.35        | 0.79    |
| Decision Tree         | 83.26        | 80.04         | 83.26      | 81.27        | 0.35    |
| Random Forest         | 86.12        | 82.85         | 86.12      | 81.38        | 0.77    |
| Gradient Boosting     | 84.89        | 81.48         | 84.89      | 82.41        | 0.74    |

### After Hyperparameter Tuning
| **Model**            | **Accuracy** | **Precision** | **Recall** | **F1-Score** | **AUC** |
|-----------------------|--------------|---------------|------------|--------------|---------|
| Logistic Regression   | 88.16        | 86.86         | 88.16      | 85.57        | 0.79    |
| Decision Tree         | 83.67        | 81.68         | 83.67      | 82.50        | 0.35    |
| Random Forest         | 87.75        | 86.21         | 87.75      | 84.89        | 0.80    |

---

## Key Findings
- **Logistic Regression**: Stable accuracy, high F1-score, and strong AUC (0.79).  
- **Random Forest**: Significant improvement post-tuning, best overall AUC (0.80).  
- **Gradient Boosting**: Moderate performance with balanced precision and recall.  
- **Decision Tree**: Limited performance improvement after tuning, low AUC (0.35).  

---

## Conclusion
This project demonstrates the importance of data preprocessing, feature engineering, and hyperparameter tuning in building effective predictive models for employee attrition. The Random Forest model, post-tuning, provides the best balance of precision, recall, and overall performance.

---

## Future Scope
- Expand the dataset for industry-specific insights.  
- Incorporate time-series analysis for temporal trends in attrition.  
- Use advanced techniques like neural networks for improved prediction accuracy.
