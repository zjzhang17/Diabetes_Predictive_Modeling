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

# Decision Tree
The decision tree model was one of the first predictive models employed on this dataset. A proper pipeline with a standard scaler was applied. GridSearchCV was applied to find optimal max features and max depth which were found to be 8 and 16 respectively. 

During development, feature engineering was performed to convert several continuous variables, such as Physical Health and Mental Health to dummy variables. BMI was log transformed and SMOTE oversampling technique was applied. The use of these techniques either individually or in combination with each other did not increase model test performance in the primary metric of ROC-AUC. 

In the case of using SMOTE, it resulted in a significant decrease in several secondary metrics such as F1 Score. We speculate that oversampling caused overfitting resulting in increased cross-validated training scores but decreased test scores. The results from this model allowed us to adopt the general methodology outlined above for the rest of our models.

# Random Forest
Following the observation of decision tree model outcomes, the Random Forest classifier was applied in this machine learning pipeline. The code systematically addresses pivotal stages, encompassing data preparation, hyperparameter tuning via grid search cross-validation, model assessment on a test set, and the visualization of key performance metrics like precision, recall, and the ROC curve.

Upon the requisite importation of libraries and the definition of features along with the target variable, the code proceeds to instantiate a Random Forest classifier. Hyperparameter tuning is systematically conducted through grid search cross-validation, utilizing a 3-fold validation strategy to pinpoint optimal values for relevant parameters.

The top-performing model, determined through the grid search process, is subsequently scrutinized on the test set. The code then calculates and presents an array of metrics, including precision, recall, and the area under the ROC curve, offering a comprehensive evaluation of the classifier's performance.

# GBM, Random Forest, and XGBoost Recursive Feature Elimination:
Recursive Feature Elimination, or RFE for short, is a feature selection algorithm. It works by recursively removing attributes and building a model on those attributes that remain. It uses the model accuracy to identify which attributes (and combination of attributes) contribute the most to predicting the target attribute. We performed Recursive Feature Elimination using XGBoost, Random Forest, and GBM. 

XGBoost: The top features identified by RFE include 'HighBP' (High Blood Pressure), 'GenHlth' (General Health), and 'HighChol' (High Cholesterol), suggesting that these are significant predictors for the model.

Random Forest: This model identifies 'BMI', 'Age', and 'PhysHlth' (Physical Health) as the top features.

GBM: The top features for the GBM model are 'GenHlth', 'HighBP', and 'BMI', which are also important features in XGBoost and Random Forest.

The top features identified by RFE include 'HighBP' (High Blood Pressure), 'GenHlth' (General Health), and 'HighChol' (High Cholesterol), suggesting that these are significant predictors for the model and we are going to use these features to build our models (XGBoost and GBM).

# XGBoost
XGBoost has proven to be essential for our classification tasks, optimized via GridSearchCV to find the optimal set of hyper-parameters such as n_estimators, learning_rate, and max_depth. Regularization parameters alpha and lambda were also fine-tuned to lower overfitting.

The model was evaluated using precision, recall, and F1-score at various thresholds, with the confusion matrices offering a clear view of its binary classification efficacy. At a 0.5 threshold, we got a high recall for the negative class but a very low recall for class 1. Adjusting the threshold to 0.25 significantly improved recall for class 1 and since we are trying to predict who will truly have Diabetes, the 0.25 threshold proves to me the more important metric. 


# Gradient Boosting Classifier
The GBM model served as a key predictive tool for our dataset. We initiated the process with a well-structured pipeline. We applied GridSearchCV, determining the best set of hyper-parameters as 'max_depth= 3', ‘learning_rate= 0.1’, etc.

The model was evaluated using precision, recall, and F1-score at various thresholds like XGBoost and we got the same results for recall, AP, etc that we got in XGBoost. 

We tried SMOTE to handle the imbalance data but the results were not that great, so we focused on the traditional machine learning strategies.

# Voter Classifier
Stacking voter classifier from Sklearn was performed using four of the preceding models, logistic regression, decision tree, random forest, and XGBoost with the previously determined hyperparameters. SVC could not be utilized as the decision function code, which was needed to evaluate the ROC-AUC score for SVM models, was not supported by the voter classifier model from Sklearn. The neural network was also not incorporated due to hardware and out-of-memory technical difficulties with using our preferred Python environment.

# SVC Linear
The LinearSVC model was initiated by loading a diabetes dataset and preparing it for classification tasks, splitting it into training and testing sets, and standardizing the features. We then implemented a Support Vector Classifier (SVC) using scikit-learn's LinearSVC. A pipeline was constructed for the classifier, and a grid search was conducted to optimize hyperparameters, specifically focusing on the regularization parameter 'C.' The best-performing SVC model was selected based on the mean cross-validation score. The model's performance was assessed on the test set using metrics such as ROC AUC. The threshold was set to -0.5. Subsequently, the code computed and visualized evaluation metrics, including a confusion matrix, ROC curve, and Precision-Recall curve. These metrics provided insights into the model's ability to classify instances, highlighting areas of strength, such as high accuracy, and areas for improvement, such as the need for better recall in identifying positive instances.

Results: The SVC model, as implemented in the provided code, demonstrated a reasonably high level of accuracy, achieving approximately 82.1% ROC AUC on the test set. The grid search for hyperparameter tuning identified the best regularization parameter 'C' as 1668.1, contributing to the model's optimal performance. However, a closer examination of the confusion matrix and evaluation metrics revealed certain challenges. The model exhibited a relatively low recall of 60.4%, indicating its difficulty in correctly identifying positive instances. On the positive side, precision was comparatively higher at 37.1%, suggesting that when the model predicted positive instances, it was correct around 57% of the time. The trade-off between precision and recall was reflected in the F1 score of 45.9%, emphasizing the need for a balanced approach. The ROC curve and Precision-Recall curve further illustrated the model's discriminative ability and precision-recall trade-off. In summary, while the SVC model demonstrated robust overall accuracy, addressing the challenges in correctly identifying positive instances could enhance its performance in specific areas.

