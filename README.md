# Wine Quality Prediction

## Background
Portugal is one of the top 10 global wine exporters, having exported $1 billion of wine in 2024. This project aims to predict the quality of wine based on its chemical properties. To ensure that the consumer receives high quality wine, we use predictive machine learning models to derive the perceived quality expected from the wine, without having to open the bottle on delivery. These models could also be used during the winemaking process to assist in wine certification and quality assessment. Our algorithm and analysis are based on the article *"Modeling wine preferences by data mining from physicochemical properties"* authored by Cortez, Cerdeira, Almeida, Matos, and Reis (2009).

## Data Set Information
The data set was obtained from the UC Irvine Machine Learning Repository, but is also accesible through Cortez's website. The data was procured from May 2004 through February 2007 using samples of vinho verde wine, produced in the Minho region of Portugal, tested at the official vinho verde certification entity - The Viticulture Commission of the Vinho Verde Region (CVRCC). The chemical properties were recorded by a computerized system that manages laboratory analysis.

The wine quality was evaluated by a minimum of three sensory assessors, which graded the wine from 0 to 10, with 0 being the lowest quality and 10 being the highest quality. The final score was given as the median of these grades.

The data set was also split between red and white wine. While the red wine had 1599 instances, white wine had 4898 instances. These datasets had no missing values.

In addition to quality, the dataset contains 11 features that represent the chemical features within wine. These features include:

- Fixed acidity - Measured in units of g(tartaric acid)/dm<sup>3</sup>
- Volatile acidity - Measured in units of g(acetic acid)/dm<sup>3</sup>
- Citric acid - Measured in units of g/dm<sup>3</sup>
- Residual sugar - Measured in units of g/dm<sup>3</sup>
- Chlorides - Measured in units of g(sodium chloride)/dm<sup>3</sup>
- Free sulfur dioxide - Measured in units of mg/dm<sup>3</sup>
- Total sulfur dioxide - Measured in units of mg/dm<sup>3</sup>
- Density - Measured in units of g/cm<sup>3</sup>
- pH - Measured on a scale from 0.0 to 14.0
- Sulphates - Measured in units of g(potassium sulphate)/dm<sup>3</sup>
- Alcohol - Measured by vol. %

All 11 features are continuous variables. Our target variable, quality, is categorical and measured on a scale of 0 to 10.

## Exploratory Data Analysis

We performed Exploratory Data Analysis on the red and white wine separately. As seen in Figure 1, most of the features within the red wine dataset have a right-skewed distribution. To address this, we used feature scaling during the data pre-processing. Additionally, our target variable is imbalanced, with responses ranging between 3 and 8 with an uneven number of responses per category. Therefore, we performed stratified sampling in the train-test split for red wine.

![Figure 1: Feature Distribution for Red Wine](./outputs/EDA/red%20wine%20feature%20distributions.png)

*Figure 1: Feature Distribution for Red Wine*

We also observed high correlation between some red wine features through the feature correlation matrix displayed in Figure 2. (e.g. positive correlations between citric acid and fixed acidity/volatile acidity, negative correlations between pH and fixed acidity) This made us consider employing feature selection techniques to reduce multicollinearity between red wine predictors. Additionally, the three most correlated features to quality were alcohol, volatile acidity, and sulphates.

![Figure 2: Correlation Matrix of Red Wine Features](./outputs/EDA/red%20wine%20correlation%20matrix.png)

*Figure 2: Correlation Matrix of Red Wine Features*

As seen in Figure 3, most of the features within the white wine dataset also have a right-skewed distribution. We performed feature scaling in data pre-processing for the white wine features as well. Here, our target variable is again imbalanced, with responses ranging between 3 and 9 with an uneven number of responses per category. Therefore, we performed stratified sampling in the train-test split for white wine.

![Figure 3: Feature Distribution for White Wine](./outputs/EDA/white%20wine%20feature%20distributions.png)

*Figure 3: Feature Distribution for White Wine*

We also observed high correlation between some white wine features through the feature correlation matrix displayed in Figure 4. (e.g. negataive correlations between alcohol and density/residual sugar) This made us consider employing feature selection techniques to reduce multicollinearity between white wine predictors. Additionally, the three most correlated features to quality were alcohol, density, and chlorides.

![Figure 4: Correlation Matrix of White Wine Features](./outputs/EDA/white%20wine%20correlation%20matrix.png)

*Figure 4: Correlation Matrix of White Wine Features*

## Approach

Since our target variable quality has a range of 0-10, we tackle wine quality prediction as both a regression and classification problem by treating the target as a continuous or categorical feature, respectively. This allows us to compare the performance of regression vs classification models on the problem set. We examine the following models:

- Linear Regression
- (Multiclass) Logistic Regression
- Support Vector Machine
  - Support Vector Regressor
  - Support Vector Classifier

Linear Regression and Support Vector Regressors were two of the models utilized by the original authors of the wine dataset; therefore, we wanted to reuse them for our comparison. For classification models, we chose Logistic Regression to establish a baseline classifier and Support Vector Classifier to compare Support Vector Machine performance between regression and classification tasks.

For both red and white wine data, we performed a train-test split with an 80%-20% allocation of training and testing sets. We stratified this split based on the target variable of quality to ensure both training and testing sets had the same distribution of target values. In our model pipelines, we applied scaling on the continuous features, Standard or Robust Scaling primarily. We utilized model-based feature selection with Sequential Feature Selector, utilizing LinearSVR and LinearSVC for regression and classification models, respectively. We also performed hyperparameter tuning to get the best training performance for all models, both with and without feature selection. The scoring method in tuning for regressors was negative mean squared error, while the scoring method for classifiers was macro precision.

Lastly, we evaluated all models using macro precision to assess model performance across all quality values, as well as to account for class imbalance. For regression models, we rounded the output of their predictions to the nearest integer value so the quality remained in the range 0-10 and could therefore be treated as classes for macro precision scoring.

## Results

### Red Wine

![Figure 5: Model Accuracy test results for Red Wine, using all and select features](./outputs/evaluation/red%20wine%20accuracy.png)

*Figure 5: Model Accuracy Test Results for Red Wine*

| Model                    | All Features | Select Features |
|------------------------- |--------------|-----------------|
| Linear Regression        | 60%          | 60%             |
| Support Vector Regressor | 63%          | 60%             |
| Logistic Regression      | 58%          | 57%             |
| Support Vector Classifier| 59%          | 57%             |

**Table 1: Accuracy Test Results for Red Wine**

![Figure 6: Model Macro Precision test results for Red Wine, using all and select features](./outputs/evaluation/red%20wine%20macro%20precision.png)

*Figure 6: Model Macro Precision Test Results for Red Wine*

| Model                    | All Features | Select Features |
|------------------------- |--------------|-----------------|
| Linear Regression        | 0.499767     | 0.333330        |
| Support Vector Regressor | 0.332123     | 0.310890        |
| Logistic Regression      | 0.312506     | 0.298259        |
| Support Vector Classifier| 0.326928     | 0.286577        |

**Table 2: Macro Precision Test Results for Red Wine**

As seen in Tables 1 and 2 as well as Figure 5, regression models perform better on the red wine test set than classification models. Although the macro precision of SVR is close to the classifiers' scories, linear regression is the highest-scoring model by far. This suggests that there is more of a linear relationship between the chemical properties of red wine and its perceived quality. We also see that model-based feature selection with LinearSVR doesn't improve performance and can drastically hurt linear regression's macro precision.

### White Wine

![Figure 7: Model Accuracy test results for White Wine, using all and select features](./outputs/evaluation/white%20wine%20accuracy.png)

*Figure 7: Model Accuracy Test Results for White Wine*

| Model                    | All Features | Select Features |
|------------------------- |--------------|-----------------|
| Linear Regression        | 51%          | 51%             |
| Support Vector Regressor | 57%          | 57%             |
| Logistic Regression      | 53%          | 51%             |
| Support Vector Classifier| 54%          | 52%             |

**Table 3: Accuracy Test Results for Red Wine**

![Figure 8: Model Macro Precision test results for White Wine, using all and select features](./outputs/evaluation/white%20wine%20macro%20precision.png)

*Figure 8: Model Macro Precision Test Results for White Wine*

| Model                    | All Features | Select Features |
|------------------------- |--------------|-----------------|
| Linear Regression        | 0.259773     | 0.275552        |
| Support Vector Regressor | 0.349101     | 0.330964        |
| Logistic Regression      | 0.220944     | 0.217569        |
| Support Vector Classifier| 0.374901     | 0.275090        |

**Table 4: Macro Precision Test Results for White Wine**

As seen in Tables 3 and 4 as well as Figure 6, SVM models perform the best for wine quality prediction. With both SVM models scoring the highest, this also lends to the idea that there is more of a nonlinear relationship between the chemical properties of white wine and its perceived quality. Model-based feature selection with LinearSVC doesn't alter the accuracy or macro precision score versus using all features much for the white wine data, except for with SVC which is hurt by a 10% drop in macro precision.

## Conclusion
Through our model experimentation and parameter tuning, we conclude that the RBF/Gaussian kernel is optimal for SVR while the polynomial kernel is optimal for SVC. Linear Regression performed the best for red wine quality prediction with a macro precision of approximately 0.50 while SVC performed the best for white wine quality prediction with a macro precision of approximately 0.37. As previously discussed, the results suggest that red wine has a linear relationship between chemical properties and quality while white wine has a nonlinear relationship between chemical properties and quality. Finally, model-based feature selection did not significantly help achieve better performance for any of the models we experimented with.

## Limitations and Future Work
Both red and white wine datasets are relatively small in size, which hurts their generalizability. Expanding these datasets in the future would strengthen the predictive power and generalization strength of these models. Additional data would also help address the class imbalance issue in both red and white wine. The datasets we used were highly imbalanced with median quality ratings (5-6) being the most common and ratings of 0, 1, 2, and 10 not showing up at all.

We are interested in further comparing additional models on wine quality prediction, such as Random Forest or Neural Networks. Other nonlinear models in particular may yield better results on predicting the quality of white wine.

Additionally, we would like to experiment with other methods of feature selection. The Sequential Feature Selector using LinearSVR/LinearSVC ended up not notably improving performance. Therefore, we would be interested in trying other feature selection methods, either more statistics-based methods or using other models for the model-based feature selection.

## Works Cited
- Wine Exports by Country. World's Top Exports. <https://www.worldstopexports.com/wine-exports-country/>  
- Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, 47(4), 547-553. <https://doi.org/10.1016/j.dss.2009.05.016>