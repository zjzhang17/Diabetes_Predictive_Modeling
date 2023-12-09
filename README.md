# Diabetes_Predictive_Modeling

Xavier Zuo, Bedilu Jebena, Rishipal R. Bansode, Jason Zhang, and Kajari Bhaumik

# Introduction 
Millions of people worldwide are affected by the global health issue of diabetes, emphasizing the critical need to manage diabetic health indicators effectively to prevent complications and enhance overall well-being. To contribute to the comprehension and forecasting of outcomes associated with diabetes, our team has decided to investigate the use of machine learning. This involves forecasting diabetic health indicators based on a dataset enriched with a variety of independent variables.

Our project's main objective is to successfully use machine learning to forecast diabetic health markers. This includes identifying patients at risk early, optimizing treatment regimens, and lowering complications related to diabetes. The overarching goal is to create machine learning models that, given a set of independent factors, can precisely predict diabetes health markers. These health indicators might cover a range of risk factors and diabetes control strategies.

As we delve into this endeavor, it's crucial to acknowledge the diverse nature of diabetes. Type 1 diabetes, usually diagnosed in children and young adults, is an autoimmune condition where the body's immune system mistakenly attacks and destroys the insulin-producing beta cells in the pancreas. While the exact cause is not well understood, it is believed to involve a combination of genetic and environmental factors.

On the other hand, Type 2 diabetes, more common in adults but not exclusive to them, is characterized by insulin resistance. Cells in the body do not respond effectively to insulin, and the pancreas may struggle to produce sufficient insulin. Risk factors for Type 2 diabetes encompass genetics, age, obesity, physical inactivity, and certain ethnicities. Understanding these distinctions is crucial as we work towards creating machine learning models capable of precisely predicting diabetes health markers, contributing to a comprehensive approach in addressing this prevalent global health concern.

# Data Source
The source of the data is Kaggle, which contained a subset of the 2015 Behavioral Risk Factor Surveillance System survey conducted by the Center for Disease Control. Our data sample included a subset of self-reported responses from some 253,680 de-identified subjects. The focus of our predictive modeling efforts is to predict the binary target, which is diabetes status, based on 21 features. Some of these topical features include BP, cholesterol, BMI, smoker status, history of stroke, cardiovascular disease or heart attack, physical activity, and various general and mental health parameters as well as demographics such as sex, age, education, and income. Links to data sources and model codes are provided in the references.

# Methodology
Based on exploratory analysis, the binary target variable of diabetes status is imbalanced with about 86% of the response being of the negative class and 14% identified as diabetics (Figure 1). All other features were continuous variables with no missing values. As the data is mostly already cleaned, additional preprocessing such as standard scaling and dummy variable creation was unnecessary. Nevertheless, some of these steps were taken for specific models as outlined below. Additional techniques such as SMOTE were applied during model development but were not utilized in the final models as they either didn’t improve or even reduced test performance.

The general process for most of the models is outlined below unless otherwise specified in the subsections for each specific model. A stratified train test split was performed with 20% of the data reserved for the test and the remaining 80% utilized for model selection as a training validation set. GridSearchCV was performed with 5-fold cross-validation to determine the optimal hyperparameters for each model. Upon selection of the models with the best hyperparameters, they were evaluated on the test data holdout via ROC-AUC, which serves as our primary performance metric. Additional metrics were performed to obtain the Average Precision, F1 Score, Recall, Precision, and Accuracy. The latter four threshold-based metrics were obtained afterward by shifting our threshold to 0.2 for most models to calculate optimal performance in these secondary metrics. Results were recorded in Table 1.

# Baseline Model Comparison for Diabetes Classification
To classify diabetes cases, we evaluated six baseline machine learning models on two metrics: Recall and Receiver Operating Characteristic (ROC) AUC.
The models include K-Nearest Neighbors (KNN), Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost. Each model's performance was assessed using 5-fold cross-validation.

When comparing the Recall and ROC AUC scores, we see the discrepancy. Models like Logistic Regression had lower recall but higher ROC AUC scores, which tells us they are better at ranking the positive class higher than the negative class rather than detecting all positive cases. On the other hand, the Decision Tree had a higher recall but lower ROC AUC, meaning it could identify more actual cases at the expense of making more false-positive errors.

Gradient Boosting and XGBoost have moderate recall but have the best ROC AUC score which makes them potentially the best models for the classification when considering both metrics (Figures 2 and 3).

# Logistic Regression
After proper preprocessing steps were taken, the Logistic Regression Machine Learning Model was employed to predict the presence of diabetes mellitus. Utilizing Scikit-learn's 'LogisticRegression' function, we incorporated it into a pipeline and conducted hyperparameter tuning using GridSearchCV with 5-fold cross-validation.

To optimize model performance, hyperparameter tuning identified the following optimal values: {'logisticregression__C': 1, 'logisticregression__solver': 'lbfgs'}. These choices were crucial for achieving the best mean cross-validation score of 0.828. The employed 5-fold cross-validation approach, with ROC-AUC as the scoring metric, offered insights into the model's robustness.
While the Logistic Regression model exhibited strong performance in predicting class 0, as evidenced by Precision (0.88), Recall (0.98), and F1 Score (0.92), challenges were noted in predicting class 1 (diabetes).

Given the class imbalance in our dataset, metrics beyond accuracy and recall were considered. Techniques such as SMOTE and threshold adjustment were implemented to handle class imbalance.

Further analysis revealed an Average Precision of 0.414, providing clear insight into the model's ability to discriminate between the two classes.

In conclusion, our Logistic Regression model shows promising results, particularly in predicting class 0. However, challenges related to class 1 prediction warrant further investigation and potential model refinement.



