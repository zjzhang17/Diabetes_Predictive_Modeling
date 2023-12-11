# Diabetes_Predictive_Modeling

By: Xavier Zuo, Bedilu Jebena, Rishipal R. Bansode, Jason Zhang, and Kajari Bhaumik

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

# Neural Network
Scikit learn multilayer preceptor (MLP) was used for neural networks on the diabetes dataset. Since this was a classification problem, an MPLClassifier was used. Imported libraries included Matplotlib, Pandas, Numpy, Scikit-learn, and Seaborn. A function called evaluateBinaryClassification was created to calculate metrics like F-score, accuracy, recall, and precision by comparing the actual and projected values. After that, a dataset was loaded and divided into testing and training sets. The training set was used to train a Multilayer Perceptron (MLP) model, while the test set was used to assess the model's performance. The algorithm also performed a grid search using cross-validation to adjust the hyperparameters of the MLP model. The top hyperparameters were printed together with the scores they earned. The test set was used to assess the final trained MLP model using the chosen hyperparameters. The code continued with further assessments, which included a confusion matrix, ROC curve, and precision-recall curve display, as well as the computation of the ROC AUC score and the number of affirmative predictions. All things considered, the code demonstrated a thorough method for creating, fine-tuning, and evaluating an MLP model for binary classification using diabetes data.
 
Results: The output from the aforementioned code revealed the performance metrics of the Multilayer Perceptron (MLP) model on the test set. The model achieved an accuracy of 0.865382, indicating that approximately 87% of the predictions were correct. However, the recall, or the ability to identify actual positive instances, was low at 0.140985, indicating that only 14% of positive instances were correctly identified. On the other hand, the precision was relatively higher at 0.568225, implying that around 57% of instances predicted as positive were indeed positive. The F-score, representing the harmonic mean of precision and recall, was 0.225916, indicating a trade-off between precision and recall. The precision-recall curve's average precision (AP) was 0.42, suggesting a moderate ability to discriminate positive instances while maintaining precision. In summary, the model demonstrated high overall accuracy but faced challenges in correctly identifying positive instances, and further exploration or fine-tuning may be considered to enhance specific aspects of its performance.

# Results and Performance Metric
The above models were evaluated using the metrics listed in the subsequent table. The primary metric was the Area Under the Curve of the Receiver Operator Curve. This metric has the advantage of being threshold-independent and allows for easy comparison across different models including those that do not utilize a standard threshold range like SVC. Overall, all the models perform approximately similar to each other with the neural network slightly edging out gradient-boosted tree classification models in terms of ROC-AUC with a score of 0.830.

## Table 1: Model Performance on Test Data
<img width="629" alt="Screenshot 2023-12-09 at 2 35 37 PM" src="https://github.com/zjzhang17/Diabetes_Predictive_Modeling/assets/116914452/de018fe5-5ca9-4022-8034-35eae4d53ed4">

*F1 Score, Recall, Precision, and Accuracy were reported using a 0.2 optimized threshold except for
**SVC which used -0.5 for decision function and neural network which used default 0.5

As the harmonic mean between recall and precision, the F1 score was used as our secondary metric for optimization. All of the models achieved suboptimal F1, Recall, Precision, and Accuracy at the default threshold setting of 0.5. Upon finding our optimal hyperparameter setting per ROC-AUC, these additional metrics were calculated using the 0.2 thresholds, which can be taken as the approximate optimal threshold values for the Voter Classifier and several tree models.

Overall, the F1 scores of all models are very closely clustered with Voter Classifier taking a small lead. Note that the neural network was unable to be properly modeled with an altered threshold and used the default threshold instead. This could be due to general hardware and memory usage issues with a resource-intensive model like a deep learning neural network. Its results were in line with the results of other models using the default setting.

# Discussion and Future Work
Collectively, our models produced similar favorable results with ROC-AUC above 0.8 for each model. F1 Score, the harmonic mean between recall and precision, was less impressive as no model was able to break 0.5. This result is unsurprising as diabetes is a highly complex condition with myriads of clinical interactions and nuances. Considering we only used 21 self-reported variable features to try to predict the individual’s disease status, being able to score above 50% in one of the metrics of recall and precision is already good. In a similar case using BRFSS data from 2014, Zidian Xie et al. (Xie, 2019) built models that had sensitivities of around 50%-51% so our models performed very favorably.

In addition, our results highlight the basic tradeoff between recall and precision, between valuing the cost of false negatives versus the cost of false positives, in trying to accurately classify our group of individuals. In the literature for similar clinical scenarios, the focus is often on recall as there is greater clinical value in correctly identifying potential individuals with the disease and minimizing false negatives versus correctly identifying the larger population of healthy individuals.

To further develop and improve on our models, it may be important to find additional high-quality data and even more importantly, properly recorded time series panel data on individuals who progressed from healthy status to diabetic positive status later on. Diabetes is a progressive chronic condition so taking only snapshots of individuals will inevitably miss out on crucial pieces of information. Additionally, building models from time series data will allow us to evaluate model performance by backtesting for forecast errors, which can be more interpretable and be  effectively communicated to clinicians and health policy stakeholders. Unfortunately due to HIPAA and privacy rules, such detailed data would be difficult to acquire in the public domain.

# Conclusion
Using a total of eight machine learning models, Decision Tree, Logistic Regression, Random Forest, XGBoost, GBM, Linear Support Vector Machine, Neural Network, and Voter Classifier, we predicted at a high level the diabetes status of individuals using a dataset with 250,000 subjects and only 21 features. We obtained ROC-AUC scores of above 0.8 for all of our models with the neural network being the best model by a small margin. We also reported secondary metrics using threshold shifting to obtain optimal scores for F1 Score, Recall, and Precision that compared favorably with literature results.

# References
Data Source
https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?resource=download 

Data Model Codes on Google Collab
https://drive.google.com/drive/folders/1wpzaHK-72NQfatfZKzKYNn2FsmBS_V-l?usp=drive_link

Xie, Z., Nikolayeva, O., Luo, J., Li, D. (2019, September 19) Building Risk Prediction Models for Type 2 Diabetes Using Machine Learning Techniques. Preventing Chronic Disease. https://www.cdc.gov/pcd/issues/2019/19_0109.htm#References 

# Appendix
## Team members built the following models and made significant contributions to the presentation and the report:

Bedilu - Logistic Regression

Jason - Random Forest

Kajari - Baseline Models, RFE, XGBoost, GBM

Rishipal - SVC, Neural Network

Xavier - Decision Tree, Voter Classifier

## Figure 1 - Distribution of Binary Target
<img width="694" alt="Screenshot 2023-12-09 at 2 40 44 PM" src="https://github.com/zjzhang17/Diabetes_Predictive_Modeling/assets/116914452/0d6d70c8-043a-4d39-a3af-3273956bd5e8">

## Figure 2
<img width="644" alt="Screenshot 2023-12-09 at 2 41 36 PM" src="https://github.com/zjzhang17/Diabetes_Predictive_Modeling/assets/116914452/dd120f12-5d53-4ea9-bb9a-a0a59e787a11">

## Figure 3
<img width="650" alt="Screenshot 2023-12-09 at 2 45 46 PM" src="https://github.com/zjzhang17/Diabetes_Predictive_Modeling/assets/116914452/f362b359-2c8f-4d01-ba7d-a483fd52140a">





