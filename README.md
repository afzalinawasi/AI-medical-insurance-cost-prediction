# Medical Insurance Cost Prediction

## Project overview

This project predicts individual medical insurance charges using patient profile attributes such as age, sex, BMI, number of children, smoking status, and region.

It is a supervised machine learning regression project built using the Medical Insurance dataset. The project includes data inspection, exploratory data analysis, feature engineering, preprocessing, model training, model evaluation, hyperparameter tuning, cross-validation, final model selection, and Streamlit deployment.

After comparing multiple regression models, checking overfitting, tuning shortlisted models, and validating final candidates using cross-validation, **Linear Regression trained on original charges** was selected as the final model.

The final model achieved the following performance:

| Metric | Value |
|---|---:|
| Train RMSE | 4,269.97 |
| Test RMSE | 4,677.16 |
| Train R² | 0.875 |
| Test R² | 0.853 |
| Train Adjusted R² | 0.872 |
| Test Adjusted R² | 0.838 |
| Mean CV RMSE | 4,375.74 |
| Mean CV R² | 0.864 |

Linear Regression was selected because it provided the best overall balance of test performance, low overfitting, cross-validation stability, interpretability, and deployment simplicity.

---

## Project objective

The objective is to build a supervised machine learning regression model that predicts medical insurance cost based on patient attributes.

The target variable is:

```text
charges
```

Since the target variable is continuous, this is a regression problem.

---
## Dataset

The dataset contains medical insurance records with the following columns:
| Column   | Description                                         |
| -------- | --------------------------------------------------- |
| age      | Age of the primary beneficiary                      |
| sex      | Gender of the insurance policyholder                |
| bmi      | Body Mass Index                                     |
| children | Number of dependents covered under insurance        |
| smoker   | Smoking status                                      |
| region   | Residential region                                  |
| charges  | Individual medical costs billed by health insurance |

### Dataset summary:
| Item            |                 Value |
| --------------- | --------------------: |
| Raw rows        |                 1,338 |
| Raw columns     |                     7 |
| Cleaned rows    |                 1,337 |
| Cleaned columns |                     7 |
| Target column   |               charges |
| Problem type    | Supervised regression |

---
## Assignment coverage

| Assignment requirement       | Covered in project                                                                             |
| ---------------------------- | ---------------------------------------------------------------------------------------------- |
| Load dataset                 | Loaded the Medical Insurance dataset using Pandas                                              |
| Inspect data                 | Used `.head()`, `.info()`, `.describe()`, shape, data types, and column checks                 |
| Perform EDA                  | Analyzed target, numerical features, categorical features, correlations, and interactions      |
| Check distributions          | Visualized distributions of `charges`, `age`, `bmi`, `children`, `sex`, `smoker`, and `region` |
| Check missing values         | Confirmed no missing values                                                                    |
| Check duplicates             | Found and removed 1 duplicate row                                                              |
| Check outliers               | Used boxplots and IQR method for numerical variables                                           |
| Check skewness               | Checked skewness of numerical variables and target                                             |
| Transform skewed target      | Compared original `charges` modelling with log-transformed `charges` modelling                 |
| Feature engineering          | Created age group, BMI category, and smoker-BMI interaction features                           |
| Encode categorical variables | Used OneHotEncoder inside sklearn pipeline                                                     |
| Feature scaling              | Used StandardScaler inside sklearn pipeline                                                    |
| Build ML models              | Trained multiple supervised regression models                                                  |
| Evaluate models              | Used MAE, MSE, RMSE, R², and Adjusted R²                                                       |
| Check overfitting            | Compared train and test performance                                                            |
| Hyperparameter tuning        | Tuned shortlisted ensemble models using RandomizedSearchCV                                     |
| Cross-validation             | Validated shortlisted final models using 5-fold cross-validation                               |
| Final model selection        | Selected Linear Regression based on test and cross-validation performance                      |
| Deployment                   | Built a Streamlit app for prediction and management insights                                   |

----
## Data understanding

The dataset contains the following input features:

- Age
- Sex
- BMI
- Number of children
- Smoking status
- Region

The target variable is:
```
Medical insurance charges
```


### Key initial findings:

- The dataset had 1,338 rows and 7 columns.
- One duplicate row was found and removed.
- There were no missing values.
- The target variable charges was strongly right-skewed.
- Smoking status showed the strongest relationship with insurance charges.
- Age and BMI also showed meaningful cost patterns.
- Sex and region showed weaker direct relationships with charges.

---
## Data cleaning summary

| Step                | Action                                                                                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------- |
| Column check        | Verified column names and data types                                                                    |
| Missing value check | No missing values found                                                                                 |
| Duplicate check     | Removed 1 duplicate row                                                                                 |
| Outlier check       | Identified high-value outliers in `charges` and some outliers in `bmi`                                  |
| Outlier decision    | Retained outliers because high medical charges and high BMI values may represent valid patient profiles |
| Final cleaned data  | 1,337 rows and 7 columns                                                                                |

---
## Exploratory data analysis summary

Key EDA findings:

- charges is strongly right-skewed.
- The average charge is higher than the median charge, showing the effect of high-cost cases.
- Smokers have much higher average and median insurance charges than non-smokers.
- Age shows a meaningful positive relationship with insurance charges.
- BMI has a weaker direct relationship overall, but the effect becomes stronger when combined with smoking status.
- Older smokers and high-BMI smokers form the highest-cost customer segments.
- Sex and region have some variation but should not be over-interpreted as standalone cost drivers.
- The number of children does not show a simple increasing cost pattern.

---
## Feature engineering and preprocessing

The following engineered features were created:
| Feature                | Purpose                                                     |
| ---------------------- | ----------------------------------------------------------- |
| age_group              | Groups customers into age bands                             |
| bmi_category           | Converts BMI into standard BMI categories                   |
| smoker_bmi_interaction | Captures combined effect of smoking status and BMI category |

The original numerical features age and bmi were retained because they contain precise continuous information useful for prediction.

Preprocessing decisions:
| Area                    | Decision                                                                 |
| ----------------------- | ------------------------------------------------------------------------ |
| Numerical features      | Scaled using StandardScaler                                              |
| Categorical features    | Encoded using OneHotEncoder                                              |
| Target variable         | Compared original charges and log-transformed charges                    |
| Data leakage prevention | Preprocessing was fitted only on training data through sklearn pipelines |
| Train-test split        | 80% training and 20% testing                                             |

---
## Models trained

The following supervised regression models were trained and compared:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regressor
- K-Nearest Neighbors Regressor
- Gradient Boosting Regressor

The models were evaluated on both:
- Original charges
- Log-transformed charges

---
## Baseline model performance

| Model                       | Test RMSE | Test R² | Overfitting status        |
| --------------------------- | --------: | ------: | ------------------------- |
| Linear Regression           |  4,677.16 |   0.853 | Low                       |
| Random Forest Regressor     |  5,094.63 |   0.826 | High                      |
| Gradient Boosting Regressor |  5,139.09 |   0.827 | Moderate                  |
| KNN Regressor               |  5,713.55 |   0.781 | Moderate                  |
| Decision Tree Regressor     |  6,956.75 |   0.675 | High                      |
| SVR                         | 12,773.65 |  -0.095 | Low, but weak performance |

Linear Regression gave the best baseline performance on the original target variable.

----
## Log-transformed target modelling

Because charges was strongly right-skewed, the same models were also trained using log-transformed charges.

**Key findings:**

- Log transformation improved some non-linear models.
- Random Forest and Gradient Boosting performed better with log-transformed target.
- Linear Regression with log-transformed target performed worse than Linear Regression on original charges.
- Decision Tree continued to overfit.
- SVR improved compared to original target, but still did not become a top model.

---
## Hyperparameter tuning

Hyperparameter tuning was performed on the strongest tunable ensemble candidates:

- Random Forest Regressor with log-transformed target
- Gradient Boosting Regressor with log-transformed target

RandomizedSearchCV was used to identify better hyperparameter combinations.

Linear Regression was not hyperparameter-tuned because the basic Linear Regression model does not have major model-complexity hyperparameters like tree depth, number of estimators, learning rate, or minimum samples split.

---
## Tuned model performance

| Model                                          | Baseline Test RMSE | Tuned Test RMSE | Tuned Test R² | Overfitting status |
| ---------------------------------------------- | -----------------: | --------------: | ------------: | ------------------ |
| Tuned Random Forest Regressor - Log Target     |           4,903.68 |        4,881.85 |         0.840 | Low                |
| Tuned Gradient Boosting Regressor - Log Target |           4,978.01 |        4,925.19 |         0.837 | Low                |


Hyperparameter tuning improved the shortlisted ensemble models, but the tuned models did not outperform the original Linear Regression baseline.

---
## Cross-validation

To avoid selecting the final model based only on one train-test split, 5-fold cross-validation was performed on the training data for the shortlisted final models.

| Model                                          | Mean CV RMSE | CV RMSE Std | Mean CV R² | CV R² Std |
| ---------------------------------------------- | -----------: | ----------: | ---------: | --------: |
| Linear Regression - Original Charges           |     4,375.74 |      511.29 |      0.864 |     0.044 |
| Tuned Random Forest Regressor - Log Target     |     4,394.33 |      589.14 |      0.862 |     0.050 |
| Tuned Gradient Boosting Regressor - Log Target |     4,449.91 |      494.64 |      0.860 |     0.044 |

Linear Regression achieved the lowest Mean CV RMSE and highest Mean CV R² among the shortlisted models.

---
## Final model selection

The final selected model is:
```
Linear Regression - Original Charges
```

**Reasons for selection:**

- Lowest Test RMSE
- Highest Test R²
- Strong Adjusted R²
- Low overfitting
- Best cross-validation performance among shortlisted models
- Simple and interpretable model structure
- Suitable for deployment in a Streamlit app

**Final model performance:**
| Metric             |    Value |
| ------------------ | -------: |
| Train MAE          | 2,301.14 |
| Test MAE           | 2,510.93 |
| Train RMSE         | 4,269.97 |
| Test RMSE          | 4,677.16 |
| Train R²           |    0.875 |
| Test R²            |    0.853 |
| Train Adjusted R²  |    0.872 |
| Test Adjusted R²   |    0.838 |
| R² gap             |    0.022 |
| Overfitting status |      Low |


---
## Streamlit app

A Streamlit app was created for user-facing prediction and management insights.

**The app includes two sections:**

- Prediction app
- Management insights dashboard

**Prediction app**

The prediction app allows users to enter:
- Age
- Sex
- BMI
- Number of children
- Smoking status
- Region

The app automatically creates the same engineered features used during model training:

- Age group
- BMI category
- Smoker-BMI interaction

**The app displays:**

- Input preview
- Derived cost-driver segment
- Predicted insurance charge
- Estimated prediction range
- Cost segment
- Management interpretation
- Management insights dashboard

**The dashboard answers business-facing questions such as:**

- Which customer segments create the highest insurance cost exposure?
- How does smoking amplify cost across BMI groups?
- Does cost exposure rise with age?
- Which segments may need risk monitoring or wellness focus?
- Which model features have the strongest impact on predicted charges?
- Why was the final model selected for deployment?

**View the Streamlit app here:** Link 

---
## Project files

| File / Folder                               | Purpose                                                                                                       |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| code.ipynb                                  | Jupyter notebook containing EDA, preprocessing, model training, tuning, evaluation, and final model selection |
| app.py                                      | Streamlit application for prediction and dashboard                                                            |
| requirements.txt                            | Python dependencies                                                                                           |
| Health_Insurance.csv                        | Dataset used for the project                                                                                  |
| artifacts/final_medical_insurance_model.pkl | Saved final model pipeline                                                                                    |
| artifacts/model_metadata.pkl                | Saved model metadata                                                                                          |
| assets/med-insurance-banner.png             | Banner image used in Streamlit app                                                                            |

---
## Important notes
- This is a learning project built on a historical dataset.
- The model output should be interpreted as a machine learning estimate, not as a real medical, insurance pricing, underwriting, eligibility, or policy decision.
- High insurance charges were retained because they may represent valid high-cost medical cases.
- The prediction range is based on the final model RMSE and should be interpreted as an approximate uncertainty range.
- Feature impact is based on model coefficients and should be interpreted as model influence, not medical causality.
- The final model was selected based on test performance, overfitting check, cross-validation performance, interpretability, and deployment suitability.
