# NYC Airbnb Room Type Classification

## AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026

**Student Name:** Shivanand Kumar  
**Degree:** B.Tech - Computer Science and Engineering 
**Project Title:** House Pricing Predictor
**Internship Program:** IBM SkillsBuild Data Analytics with AI Academic Internship Program 2026  
**Conducted By:** BharatCares in association with AICTE  
**Domain:** Data Analytics, Machine Learning & Artificial Intelligence  
**Project Type:** Multi-Class Classification  
**Programming Language:** Python 3.12.7  

---

# 1. Project Overview

This project has been developed as part of the **IBM SkillsBuild Data Analytics with AI Academic Internship Program 2026**, conducted by **BharatCares in association with AICTE**.

The project focuses on analyzing the **New York City Airbnb Open Data** dataset and developing a Machine Learning classification model that predicts the type of Airbnb listing based on different characteristics of the property.

The model predicts one of the following three room types:

- Entire home/apt
- Private room
- Shared room

The project covers the complete Machine Learning lifecycle:

```text
Dataset Collection
        ↓
Data Loading
        ↓
Data Understanding
        ↓
Exploratory Data Analysis
        ↓
Missing Value Analysis
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Train/Test Split
        ↓
Data Preprocessing
        ↓
Model Training
        ↓
Cross-Validation
        ↓
Model Comparison
        ↓
Hyperparameter Tuning
        ↓
Final Model Evaluation
        ↓
Confusion Matrix
        ↓
Model Serialization
        ↓
FastAPI Deployment
        ↓
Frontend Integration
```

The final tuned Random Forest model achieved:

```text
Test Accuracy : 85.59%
Test Macro F1 : 74.10%
```

The complete preprocessing and Machine Learning pipeline is saved as:

```text
Model_Pipeline.pkl
```

and served through a FastAPI backend.

---

# 2. Problem Statement

Airbnb listings contain many attributes such as location, price, minimum nights, number of reviews, review frequency, host listing count, availability, and neighbourhood information.

The objective of this project is to use these attributes to predict the type of Airbnb listing.

The model predicts:

```text
Entire home/apt
Private room
Shared room
```

This makes the project a:

```text
Multi-Class Classification Problem
```

---

# 3. Project Objectives

The main objectives of this project are:

1. Download and load the NYC Airbnb dataset.
2. Understand the structure of the dataset.
3. Perform Exploratory Data Analysis.
4. Analyze missing values.
5. Analyze numerical and categorical variables.
6. Analyze the target variable.
7. Analyze feature distributions.
8. Analyze relationships between features and target.
9. Analyze correlations between numerical features.
10. Visualize the geographical distribution of listings.
11. Clean unnecessary columns.
12. Handle missing values.
13. Handle extreme values and outliers.
14. Separate features and target.
15. Split the dataset into training and testing datasets.
16. Build a reusable preprocessing pipeline.
17. Train multiple classification algorithms.
18. Compare models using cross-validation.
19. Handle class imbalance.
20. Perform RandomizedSearchCV hyperparameter tuning.
21. Evaluate the final model on an untouched test set.
22. Generate a confusion matrix.
23. Save the final Machine Learning pipeline.
24. Build a FastAPI prediction API.
25. Validate API inputs using Pydantic.
26. Connect the API with a frontend.
27. Provide an end-to-end Machine Learning application.

---

# 4. Dataset Information

## Dataset Name

```text
New York City Airbnb Open Data
```

## Dataset Source

Kaggle:

https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data

## Original Dataset File

```text
AB_NYC_2019.csv
```

## Dataset Download

The Jupyter Notebook downloads the dataset using KaggleHub:

```python
import kagglehub
import os

path = kagglehub.dataset_download(
    "dgomonov/new-york-city-airbnb-open-data"
)

df = pd.read_csv(
    os.path.join(path, "AB_NYC_2019.csv")
)
```

This allows the notebook to download the public dataset directly from Kaggle.

---

# 5. Dataset Size

The original dataset contains:

```text
Rows    : 48,895
Columns : 16
Shape   : (48895, 16)
```

The dataset contains information about Airbnb listings in New York City.

---

# 6. Original Dataset Columns

The original dataset contains the following 16 columns:

```text
id
name
host_id
host_name
neighbourhood_group
neighbourhood
latitude
longitude
room_type
price
minimum_nights
number_of_reviews
last_review
reviews_per_month
calculated_host_listings_count
availability_365
```

---

# 7. Dataset Column Description

| Column | Type | Description |
|---|---|---|
| `id` | Integer | Unique Airbnb listing identifier |
| `name` | Object | Name/title of the listing |
| `host_id` | Integer | Unique host identifier |
| `host_name` | Object | Name of the host |
| `neighbourhood_group` | Object | Major geographical area/borough |
| `neighbourhood` | Object | Specific neighbourhood |
| `latitude` | Float | Latitude coordinate |
| `longitude` | Float | Longitude coordinate |
| `room_type` | Object | Target variable representing room type |
| `price` | Integer | Price per night |
| `minimum_nights` | Integer | Minimum number of nights required |
| `number_of_reviews` | Integer | Total number of reviews |
| `last_review` | Object | Date of the most recent review |
| `reviews_per_month` | Float | Average number of reviews per month |
| `calculated_host_listings_count` | Integer | Number of listings managed by host |
| `availability_365` | Integer | Number of days available during a year |

---

# 8. Initial Dataset Inspection

The dataset was inspected using:

```python
df.head()
df.info()
df.describe()
df.shape
```

The dataset contains:

```text
48,895 records
16 columns
```

The numerical features include:

```text
latitude
longitude
price
minimum_nights
number_of_reviews
reviews_per_month
calculated_host_listings_count
availability_365
```

The categorical features include:

```text
neighbourhood_group
neighbourhood
room_type
```

---

# 9. Statistical Summary

Important statistics from the original dataset include:

| Feature | Mean | Minimum | Maximum |
|---|---:|---:|---:|
| `price` | 152.72 | 0 | 10,000 |
| `minimum_nights` | 7.03 | 1 | 1,250 |
| `number_of_reviews` | 23.27 | 0 | 629 |
| `reviews_per_month` | 1.37 | 0.01 | 58.50 |
| `calculated_host_listings_count` | 7.14 | 1 | 327 |
| `availability_365` | 112.78 | 0 | 365 |

The statistics show that some numerical variables contain highly skewed distributions and extreme values.

---

# 10. Missing Value Analysis

Missing values were checked using:

```python
missing = df.isnull().sum()
missing[missing > 0]
```

The original dataset contains the following missing values:

| Column | Missing Values |
|---|---:|
| `name` | 16 |
| `host_name` | 21 |
| `last_review` | 10,052 |
| `reviews_per_month` | 10,052 |

The relevant missing values were handled during data cleaning and preprocessing.

---

# 11. Target Variable

The target variable used in this project is:

```text
room_type
```

The unique classes are:

```text
Private room
Entire home/apt
Shared room
```

The notebook verifies this using:

```python
df_clean['room_type'].unique()
```

Output:

```text
['Private room', 'Entire home/apt', 'Shared room']
```

---

# 12. Target Class Distribution

The distribution of room types was analyzed using:

```python
df['room_type'].value_counts()
```

A count plot was also created:

```python
sns.countplot(
    x='room_type',
    data=df
)
```

The dataset is imbalanced because the number of listings in the three room-type categories is not equal.

The `Shared room` class contains substantially fewer examples than the other room types.

Because of this imbalance, the project does not rely only on accuracy and also uses:

```text
Macro F1 Score
```

---

# 13. Exploratory Data Analysis

Exploratory Data Analysis was performed before Machine Learning.

The EDA includes:

```text
Missing Value Analysis
        ↓
Target Distribution
        ↓
Univariate Analysis
        ↓
Bivariate Analysis
        ↓
Correlation Analysis
        ↓
Outlier Analysis
        ↓
Geographical Analysis
```

---

# 14. Univariate Analysis

Numerical variables were analyzed individually.

The main numerical variables were:

```text
price
minimum_nights
number_of_reviews
reviews_per_month
calculated_host_listings_count
availability_365
```

Histograms were used to understand:

- Distribution
- Skewness
- Extreme values
- Data spread
- Potential outliers

Example:

```python
df[numerical_cols].hist(
    bins=30,
    figsize=(12, 8)
)
```

---

# 15. Categorical Analysis

Categorical variables were also analyzed.

The project analyzes:

```text
neighbourhood_group
neighbourhood
room_type
```

For example:

```python
sns.countplot(
    data=df,
    x='neighbourhood_group'
)
```

This helps understand the geographical distribution of Airbnb listings.

---

# 16. Bivariate Analysis

The relationship between numerical variables and room type was analyzed.

For example, price versus room type was visualized using:

```python
sns.boxplot(
    x='room_type',
    y='price',
    data=df
)
```

This helps understand how price distributions differ among:

```text
Entire home/apt
Private room
Shared room
```

---

# 17. Geographic Analysis

The project also analyzes the geographical distribution of Airbnb listings.

The following features were used:

```text
latitude
longitude
room_type
```

A scatter plot was created using:

```python
sns.scatterplot(
    x='longitude',
    y='latitude',
    data=df,
    hue='room_type',
    alpha=0.4,
    s=10
)
```

This visualization helps show where different room types are located across New York City.

---

# 18. Data Cleaning

The notebook performs the following cleaning steps:

```text
1. Remove unnecessary columns
2. Handle missing review information
3. Cap extreme price values
4. Cap extreme minimum-night values
5. Separate features and target
```

---

# 19. Removing Unnecessary Columns

The following columns were removed:

```text
id
name
host_id
host_name
last_review
```

The code used is:

```python
df_clean = df.drop(
    columns=[
        'id',
        'name',
        'host_id',
        'host_name',
        'last_review'
    ]
)
```

These columns were not used in the final tabular Machine Learning model because they are identifiers, free-text fields, host names, or review-date information that was not included in the final feature set.

---

# 20. Handling Missing Reviews

The `reviews_per_month` column contains missing values.

The project replaces missing values with:

```text
0
```

The code is:

```python
df_clean['reviews_per_month'] = (
    df_clean['reviews_per_month'].fillna(0)
)
```

The interpretation is that a missing monthly review value can represent no recorded review activity.

---

# 21. Outlier Treatment

The original dataset contains extreme values.

Examples include:

```text
Maximum price          = 10,000
Maximum minimum_nights = 1,250
```

These values can have a strong effect on some Machine Learning algorithms.

Therefore, percentile-based clipping was used.

The upper 99th percentile was used as the clipping threshold.

```python
price_cap = df_clean['price'].quantile(0.99)

nights_cap = (
    df_clean['minimum_nights'].quantile(0.99)
)

df_clean['price'] = (
    df_clean['price'].clip(
        upper=price_cap
    )
)

df_clean['minimum_nights'] = (
    df_clean['minimum_nights'].clip(
        upper=nights_cap
    )
)
```

Instead of deleting complete rows, the extreme values are clipped.

This preserves the observations while reducing the influence of extreme values.

---

# 22. Feature and Target Separation

After cleaning:

```python
X = df_clean.drop(
    columns=['room_type']
)

y = df_clean['room_type']
```

Where:

```text
X = Input Features
y = Target Variable
```

---

# 23. Final Features

The final Machine Learning features are:

## Numerical Features

```text
latitude
longitude
price
minimum_nights
number_of_reviews
reviews_per_month
calculated_host_listings_count
availability_365
```

## Categorical Features

```text
neighbourhood_group
neighbourhood
```

## Target

```text
room_type
```

---

# 24. Train-Test Split

The dataset is divided into training and testing data.

The project uses:

```text
67% Training Data
33% Testing Data
```

The split is performed using:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
    stratify=y
)
```

---

# 25. Stratified Sampling

The project uses:

```python
stratify=y
```

because the target classes are imbalanced.

Stratification maintains approximately the same class proportions in:

```text
Training Data
Testing Data
```

This is important for reliable evaluation of an imbalanced classification problem.

---

# 26. Data Preprocessing

The project uses Scikit-learn:

```text
Pipeline
ColumnTransformer
SimpleImputer
StandardScaler
OneHotEncoder
```

The preprocessing is performed separately for numerical and categorical variables.

---

# 27. Numerical Preprocessing

Numerical columns are processed using:

```text
Median Imputation
        ↓
Standard Scaling
```

The pipeline is:

```python
numeric_pipeline = Pipeline(
    steps=[
        (
            'impute',
            SimpleImputer(
                strategy='median'
            )
        ),
        (
            'scale',
            StandardScaler()
        )
    ]
)
```

---

# 28. Categorical Preprocessing

Categorical columns are processed using:

```text
Most-Frequent Imputation
        ↓
One-Hot Encoding
```

The pipeline is:

```python
categorical_pipeline = Pipeline(
    steps=[
        (
            'impute',
            SimpleImputer(
                strategy='most_frequent'
            )
        ),
        (
            'encode',
            OneHotEncoder(
                handle_unknown='ignore'
            )
        )
    ]
)
```

The parameter:

```text
handle_unknown='ignore'
```

prevents errors if an unseen categorical value appears during prediction.

---

# 29. ColumnTransformer

The numerical and categorical pipelines are combined using:

```python
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numeric_pipeline,
            numerical_cols
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_cols
        )
    ]
)
```

This allows the same preprocessing logic to be used during:

```text
Training
Cross-Validation
Testing
Prediction
```

---

# 30. Why Pipeline Was Used

The complete preprocessing and model are combined in a Scikit-learn Pipeline.

This provides several advantages:

- Consistent preprocessing
- Easier deployment
- Reduced risk of preprocessing mismatch
- Reduced risk of data leakage
- One-step prediction
- Easy model serialization

The final artifact therefore contains both:

```text
Preprocessing
+
Machine Learning Model
```

---

# 31. Machine Learning Models

Four classification algorithms were compared:

```text
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
```

All candidate models were evaluated using the same preprocessing pipeline.

---

# 32. Logistic Regression

Logistic Regression was used as a baseline classification model.

Configuration:

```python
LogisticRegression(
    class_weight="balanced",
    random_state=42
)
```

The balanced class weight was used because of the class imbalance.

---

# 33. Decision Tree

Decision Tree was used to capture non-linear relationships between the input features and target.

Configuration:

```python
DecisionTreeClassifier(
    class_weight="balanced",
    random_state=42
)
```

---

# 34. Random Forest

Random Forest is an ensemble algorithm that combines multiple decision trees.

Configuration:

```python
RandomForestClassifier(
    class_weight="balanced",
    random_state=42
)
```

The model uses:

```text
class_weight="balanced"
```

to give additional importance to minority classes.

---

# 35. Gradient Boosting

Gradient Boosting was also tested.

Configuration:

```python
GradientBoostingClassifier(
    random_state=42
)
```

The implementation used in this project does not use the same `class_weight` parameter, so it was evaluated without class weighting.

---

# 36. Class Imbalance Handling

The dataset has an imbalanced target variable.

The minority class is:

```text
Shared room
```

For models supporting class weights, the project uses:

```python
class_weight="balanced"
```

This helps reduce the tendency of the model to focus mainly on majority classes.

The project also uses:

```text
Macro F1 Score
```

because it treats each class equally during evaluation.

---

# 37. Cross-Validation

The models were compared using:

```text
3-Fold Cross-Validation
```

The notebook uses:

```python
cross_val_score(
    pipe,
    X_train,
    y_train,
    cv=3,
    scoring="accuracy"
)
```

and:

```python
cross_val_score(
    pipe,
    X_train,
    y_train,
    cv=3,
    scoring="f1_macro"
)
```

The two metrics used were:

```text
Accuracy
Macro F1 Score
```

---

# 38. Model Comparison Results

The actual cross-validation results obtained in the notebook are:

| Model | Cross-Validation Accuracy | Cross-Validation Macro F1 |
|---|---:|---:|
| Logistic Regression | 0.659 | 0.522 |
| Decision Tree | 0.782 | 0.647 |
| Random Forest | 0.851 | 0.715 |
| Gradient Boosting | 0.850 | 0.705 |

The Random Forest model provided the highest cross-validation Macro F1 score among the tested candidate models.

---

# 39. Why Random Forest Was Selected

Based on the cross-validation results:

```text
Random Forest
Accuracy : 0.851
Macro F1 : 0.715
```

Random Forest showed the strongest Macro F1 performance among the tested models.

Because the project has imbalanced target classes, Macro F1 was given particular importance when selecting the model for further tuning.

---

# 40. Hyperparameter Tuning

Random Forest was selected for hyperparameter tuning.

The project uses:

```text
RandomizedSearchCV
```

instead of exhaustive Grid Search.

The reason is to explore multiple parameter combinations efficiently.

The parameter search space is:

```python
param_distribution = {
    "classifier__n_estimators": [
        100,
        200,
        150,
        300
    ],
    "classifier__max_depth": [
        8,
        12,
        15,
        20,
        None
    ],
    "classifier__min_samples_split": [
        2,
        5,
        10
    ]
}
```

---

# 41. RandomizedSearchCV Configuration

The search configuration is:

```python
search = RandomizedSearchCV(
    estimator=best_pipeline,
    param_distributions=param_distribution,
    n_iter=10,
    cv=3,
    scoring="f1_macro",
    random_state=42
)

search.fit(
    X_train,
    y_train
)
```

The optimization metric is:

```text
Macro F1 Score
```

---

# 42. Best Hyperparameters

The actual best parameters obtained from the notebook are:

```text
n_estimators       = 200
min_samples_split  = 10
max_depth          = None
```

Complete result:

```python
{
    'classifier__n_estimators': 200,
    'classifier__min_samples_split': 10,
    'classifier__max_depth': None
}
```

---

# 43. Best Cross-Validation Score

The best Macro F1 score obtained during hyperparameter tuning was:

```text
0.7299771194688965
```

Approximately:

```text
72.9977%
```

---

# 44. Final Model

The final Machine Learning model is:

```text
Random Forest Classifier
```

with:

```text
n_estimators      = 200
min_samples_split = 10
max_depth         = None
class_weight      = balanced
```

The final model is stored inside the complete Scikit-learn Pipeline.

---

# 45. Final Test Evaluation

The final tuned model was evaluated on the held-out test dataset.

The test data was not used during hyperparameter selection.

The notebook produced:

```text
Accuracy Score: 0.855912245909767
F1 Score:      0.7410370578191872
```

Therefore:

```text
Test Accuracy = 85.59%
Test Macro F1 = 74.10%
```

---

# 46. Final Performance Summary

| Metric | Result |
|---|---:|
| Cross-Validation Accuracy - Logistic Regression | 65.9% |
| Cross-Validation Macro F1 - Logistic Regression | 52.2% |
| Cross-Validation Accuracy - Decision Tree | 78.2% |
| Cross-Validation Macro F1 - Decision Tree | 64.7% |
| Cross-Validation Accuracy - Random Forest | 85.1% |
| Cross-Validation Macro F1 - Random Forest | 71.5% |
| Cross-Validation Accuracy - Gradient Boosting | 85.0% |
| Cross-Validation Macro F1 - Gradient Boosting | 70.5% |
| Best Tuned CV Macro F1 | 72.9977% |
| Final Test Accuracy | 85.59% |
| Final Test Macro F1 | 74.10% |

---

# 47. Confusion Matrix

A confusion matrix was generated for the final model using:

```python
from sklearn.metrics import confusion_matrix

confusion_matrix(
    y_test,
    y_pred
)
```

The matrix was visualized using:

```python
sns.heatmap(
    confusion_matrix(
        y_test,
        y_pred
    ),
    annot=True,
    fmt=".2f",
    xticklabels=best_pipeline.classes_,
    yticklabels=best_pipeline.classes_
)
```

The confusion matrix helps understand:

- Correct predictions
- Incorrect predictions
- Class-level errors
- Confusion between room types

The classes are:

```text
Entire home/apt
Private room
Shared room
```

---

# 48. Model Artifact

After training, the complete Machine Learning pipeline was saved using Joblib.

The code is:

```python
import joblib

joblib.dump(
    best_pipeline,
    "Model_Pipeline.pkl",
    compress=3
)
```

The generated artifact is:

```text
Model_Pipeline.pkl
```

---

# 49. Why the Complete Pipeline Was Saved

The project saves the entire pipeline rather than saving only the classifier.

The pipeline contains:

```text
Input Data
     ↓
Numerical Imputation
     ↓
Numerical Scaling
     ↓
Categorical Imputation
     ↓
One-Hot Encoding
     ↓
Random Forest Classifier
     ↓
Prediction
```

Therefore, during inference, the application can directly pass raw listing information to the saved pipeline.

No manual preprocessing is required.

---

# 50. FastAPI Deployment

The trained model is deployed through FastAPI.

The backend file is:

```text
main.py
```

The application loads the saved model using:

```python
model = joblib.load(
    "Model_Pipeline.pkl"
)
```

The backend provides:

```text
GET /
POST /predict
```

---

# 51. FastAPI Application

The application is initialized using:

```python
from fastapi import FastAPI

app = FastAPI()
```

---

# 52. CORS Configuration

The backend uses FastAPI CORS middleware:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

This allows the frontend application to communicate with the API.

---

# 53. API Input Features

The prediction API accepts:

```text
latitude
longitude
price
minimum_nights
number_of_reviews
reviews_per_month
calculated_host_listings_count
availability_365
neighbourhood_group
neighbourhood
```

These are exactly the features expected by the saved Machine Learning pipeline.

---

# 54. Pydantic Input Validation

The API uses Pydantic to validate the input.

Important validation rules include:

| Feature | Validation |
|---|---|
| `latitude` | -90 to 90 |
| `longitude` | -180 to 180 |
| `price` | Greater than 0 |
| `minimum_nights` | 1 to 365 |
| `number_of_reviews` | Greater than or equal to 0 |
| `reviews_per_month` | Greater than or equal to 0 |
| `calculated_host_listings_count` | Greater than or equal to 0 |
| `availability_365` | 0 to 365 |
| `neighbourhood_group` | Non-empty string |
| `neighbourhood` | Non-empty string |

Example:

```python
class Features(BaseModel):

    latitude: float = Field(
        ...,
        ge=-90,
        le=90
    )

    longitude: float = Field(
        ...,
        ge=-180,
        le=180
    )

    price: float = Field(
        ...,
        gt=0
    )

    minimum_nights: int = Field(
        ...,
        ge=1,
        le=365
    )

    number_of_reviews: int = Field(
        ...,
        ge=0
    )

    reviews_per_month: float = Field(
        ...,
        ge=0
    )

    calculated_host_listings_count: int = Field(
        ...,
        ge=0
    )

    availability_365: int = Field(
        ...,
        ge=0,
        le=365
    )

    neighbourhood_group: str = Field(
        ...,
        min_length=1
    )

    neighbourhood: str = Field(
        ...,
        min_length=1
    )
```

---

# 55. API Endpoint - Health Check

The root endpoint is:

```text
GET /
```

Implementation:

```python
@app.get('/')
def greet():
    return "Hello Guyss"
```

This endpoint can be used to check whether the FastAPI application is running.

---

# 56. API Endpoint - Prediction

The main Machine Learning endpoint is:

```text
POST /predict
```

Implementation:

```python
@app.post('/predict')
def predict(features: Features):

    row = pd.DataFrame(
        [features.dict()],
        columns=COLUMNS
    )

    prediction = model.predict(row)

    probability = model.predict_proba(row)

    return {
        "Predicted_room_type": prediction[0],
        "Probability": probability.tolist()[0]
    }
```

---

# 57. Example Prediction Request

The API accepts JSON in the following format:

```json
{
    "latitude": 40.7128,
    "longitude": -74.0060,
    "price": 150,
    "minimum_nights": 2,
    "number_of_reviews": 25,
    "reviews_per_month": 2.5,
    "calculated_host_listings_count": 3,
    "availability_365": 200,
    "neighbourhood_group": "Manhattan",
    "neighbourhood": "Midtown"
}
```

---

# 58. Example Prediction Response

The API returns:

```json
{
    "Predicted_room_type": "Private room",
    "Probability": [
        0.15,
        0.78,
        0.07
    ]
}
```

The actual probability values depend on the input data and the trained model.

---

# 59. Prediction Flow

The complete prediction process is:

```text
User enters listing information
              ↓
Frontend sends JSON request
              ↓
FastAPI receives request
              ↓
Pydantic validates input
              ↓
Input converted to Pandas DataFrame
              ↓
Saved Model Pipeline receives data
              ↓
Numerical preprocessing
              ↓
Categorical preprocessing
              ↓
Random Forest prediction
              ↓
Prediction probability
              ↓
JSON response
              ↓
Frontend displays result
```

---

# 60. Frontend Application

The project includes a web frontend.

Frontend files:

```text
index.html
style.css
script.js
```

The frontend provides a user-friendly form for entering Airbnb listing details.

---

# 61. Frontend Features

The frontend allows users to enter:

### Location

```text
Latitude
Longitude
Borough
Neighbourhood
```

### Pricing and Stay

```text
Price per night
Minimum nights
Availability
```

### Reviews and Host

```text
Total reviews
Reviews per month
Listings by host
```

The user can then click:

```text
Predict room type
```

to receive the Machine Learning prediction.

---

# 62. Frontend API Configuration

The frontend connects to the deployed FastAPI backend using:

```javascript
const API_BASE_URL =
    "https://nyc-airbnb-room-type-predictor.onrender.com";

const PREDICT_ENDPOINT =
    `${API_BASE_URL}/predict`;

const HEALTH_ENDPOINT =
    `${API_BASE_URL}/`;
```

The frontend therefore sends prediction requests to the deployed API.

---

# 63. Frontend Example Data

The frontend also provides sample listings through the:

```text
Try an example
```

button.

Example listing information includes:

```text
Latitude
Longitude
Price
Minimum Nights
Number of Reviews
Reviews per Month
Host Listings
Availability
Neighbourhood Group
Neighbourhood
```

This allows users to test the application without entering every value manually.

---

# 64. Prediction Visualization

After receiving a prediction, the frontend displays:

```text
Predicted Room Type
Probability of Each Room Type
```

The frontend visually represents the probabilities for:

```text
Entire home/apt
Private room
Shared room
```

This provides a more understandable representation of the model output.

---

# 65. API Health Status

The frontend also checks the backend API.

It sends a request to:

```text
GET /
```

and displays the API state as:

```text
API connected
```

or:

```text
API unreachable
```

---

# 66. Project Architecture

The complete system architecture is:

```text
                 NYC Airbnb Dataset
                         |
                         v
                Jupyter Notebook
                         |
                         v
               Data Preprocessing
                         |
                         v
             Machine Learning Training
                         |
                         v
              Hyperparameter Tuning
                         |
                         v
                Final Random Forest
                         |
                         v
                Model_Pipeline.pkl
                         |
                         v
                  FastAPI Backend
                         |
                         v
                    /predict
                         |
                         v
                Frontend Application
                         |
                         v
                   User Prediction
```

---

# 67. Project Directory Structure

The project contains:

```text
House Pricing/
│
├── Model_Pipeline.pkl
│
├── house_price.ipynb
│
├── main.py
│
├── requirements.txt
│
├── runtime.txt
│
├── index.html
│
├── script.js
│
├── style.css
│
└── the_build_line_guide.html
```

---

# 68. File Description

| File | Description |
|---|---|
| `house_price.ipynb` | Complete Data Analytics, EDA, preprocessing, model training, tuning and evaluation |
| `Model_Pipeline.pkl` | Saved complete Machine Learning pipeline |
| `main.py` | FastAPI backend and prediction API |
| `requirements.txt` | Python dependencies for the API |
| `runtime.txt` | Python runtime configuration |
| `index.html` | Frontend structure |
| `script.js` | Frontend logic and API communication |
| `style.css` | Frontend design and styling |
| `the_build_line_guide.html` | Supporting HTML resource |

---

# 69. Requirements

The project contains a `requirements.txt` file with:

```text
fastapi==0.115.6
uvicorn[standard]==0.34.0
pydantic==2.10.4
pandas==2.2.3
scikit-learn==1.6.1
joblib==1.4.2
```

The notebook additionally uses libraries such as:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
kagglehub
```

---

# 70. Python Runtime

The project specifies:

```text
Python 3.12.7
```

This is defined in:

```text
runtime.txt
```

The file contains:

```text
python-3.12.7
```

---

# 71. Installation - Complete Setup

## Step 1: Extract the Project

Extract the project ZIP file.

Open the project directory:

```bash
cd "House Pricing"
```

---

## Step 2: Create Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
```

Activate:

```bash
source venv/bin/activate
```

---

# 72. Install Dependencies

Install backend dependencies:

```bash
pip install -r requirements.txt
```

For running the notebook, install Jupyter and notebook-specific packages if required:

```bash
pip install jupyter numpy matplotlib seaborn kagglehub
```

---

# 73. Run the Jupyter Notebook

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
house_price.ipynb
```

Run the notebook cells sequentially.

The notebook performs:

```text
Dataset Download
        ↓
Dataset Loading
        ↓
EDA
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Train/Test Split
        ↓
Preprocessing
        ↓
Model Comparison
        ↓
Hyperparameter Tuning
        ↓
Final Evaluation
        ↓
Model Saving
```

---

# 74. Run FastAPI Backend

Make sure the following file is present:

```text
Model_Pipeline.pkl
```

Then run:

```bash
uvicorn main:app --reload
```

The local API will normally be available at:

```text
http://127.0.0.1:8000
```

---

# 75. Open API Documentation

FastAPI automatically provides Swagger documentation.

Open:

```text
http://127.0.0.1:8000/docs
```

This provides an interactive interface for testing:

```text
GET /
POST /predict
```

---

# 76. Test the Prediction API

Open:

```text
http://127.0.0.1:8000/docs
```

Select:

```text
POST /predict
```

Click:

```text
Try it out
```

Enter:

```json
{
    "latitude": 40.7128,
    "longitude": -74.0060,
    "price": 150,
    "minimum_nights": 2,
    "number_of_reviews": 25,
    "reviews_per_month": 2.5,
    "calculated_host_listings_count": 3,
    "availability_365": 200,
    "neighbourhood_group": "Manhattan",
    "neighbourhood": "Midtown"
}
```

Click:

```text
Execute
```

The API will return the predicted room type and probability values.

---

# 77. Run Frontend

The frontend consists of:

```text
index.html
style.css
script.js
```

The JavaScript file is configured to communicate with the deployed FastAPI backend:

```text
https://nyc-airbnb-room-type-predictor.onrender.com
```

The frontend can be served using any static web server.

For example, using Python:

```bash
python -m http.server 5500
```

Then open:

```text
http://localhost:5500
```

---

# 78. End-to-End Usage

The complete application can be used as follows:

### Step 1

Open the frontend.

### Step 2

Enter Airbnb listing information.

### Step 3

Provide:

```text
Latitude
Longitude
Borough
Neighbourhood
Price
Minimum Nights
Availability
Total Reviews
Reviews per Month
Host Listing Count
```

### Step 4

Click:

```text
Predict room type
```

### Step 5

The frontend sends the information to:

```text
POST /predict
```

### Step 6

FastAPI validates the request.

### Step 7

The saved Machine Learning pipeline performs preprocessing and prediction.

### Step 8

The API returns:

```text
Predicted Room Type
Prediction Probability
```

### Step 9

The frontend displays the result.

---

# 79. Model Evaluation Strategy

The evaluation strategy consists of three major stages:

```text
Stage 1
Candidate Model Cross-Validation
        ↓
Stage 2
Hyperparameter Tuning
        ↓
Stage 3
Final Evaluation on Held-Out Test Set
```

The test dataset remains separate from the model-selection process.

---

# 80. Evaluation Metrics

## Accuracy

Accuracy measures the percentage of correct predictions:

```text
Accuracy =
Correct Predictions / Total Predictions
```

---

## Macro F1

Macro F1 calculates the F1 score independently for every class and then takes the average.

This is particularly useful for this project because the target variable is imbalanced.

---

## Confusion Matrix

The confusion matrix shows how many observations from each actual class were predicted as each class.

It helps identify class-specific errors.

---

# 81. Why Accuracy Alone Is Not Enough

The dataset has an imbalanced distribution of room types.

If a model performs very well on the majority class but poorly on the minority class, accuracy can still remain relatively high.

Macro F1 gives equal importance to:

```text
Entire home/apt
Private room
Shared room
```

Therefore, Macro F1 was used during hyperparameter optimization.

---

# 82. Data Leakage Prevention

The project uses a Pipeline and ColumnTransformer so that preprocessing is fitted as part of the Machine Learning workflow.

The preprocessing includes:

```text
Median Imputation
StandardScaler
Most-Frequent Imputation
OneHotEncoder
```

These steps are learned within the training process and applied consistently to validation and test data.

The final test set is evaluated only after model selection and hyperparameter tuning.

---

# 83. Reproducibility

The project uses fixed random seeds where applicable:

```text
random_state = 42
```

This is used in:

```text
Train-Test Split
Logistic Regression
Decision Tree
Random Forest
Gradient Boosting
RandomizedSearchCV
```

This helps make the Machine Learning workflow reproducible.

---

# 84. Key Results

The major results of the project are:

```text
Dataset Size
48,895 rows
16 columns
```

```text
Candidate Models
4
```

```text
Best Candidate Model
Random Forest
```

```text
Best Tuned Parameters
n_estimators = 200
min_samples_split = 10
max_depth = None
```

```text
Best CV Macro F1
72.9977%
```

```text
Final Test Accuracy
85.59%
```

```text
Final Test Macro F1
74.10%
```

---

# 85. Learning Outcomes

This project provided practical experience in Data Analytics and Machine Learning.

## Data Analytics

- Dataset loading
- Data inspection
- Statistical analysis
- Missing value analysis
- Exploratory Data Analysis
- Data visualization
- Correlation analysis
- Outlier analysis

## Data Preprocessing

- Feature selection
- Missing value handling
- Numerical imputation
- Categorical imputation
- Feature scaling
- One-hot encoding
- Train-test splitting
- Data leakage prevention

## Machine Learning

- Multi-class classification
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Cross-validation
- Class imbalance handling
- Hyperparameter tuning
- RandomizedSearchCV

## Model Evaluation

- Accuracy
- Macro F1 Score
- Confusion Matrix

## Deployment

- Joblib
- FastAPI
- Pydantic
- Uvicorn
- REST API
- CORS
- Frontend integration

---

# 86. Real-World Applications

The project can potentially be used as a foundation for:

- Airbnb listing analytics
- Hospitality analytics
- Property management
- Accommodation recommendation systems
- Travel platforms
- Real estate analytics
- Market research
- Property comparison
- Listing classification
- Data-driven accommodation analysis

---

# 87. Project Limitations

The current project has several limitations.

## 1. Historical Dataset

The model is trained using the publicly available NYC Airbnb dataset and therefore represents the information contained in that dataset.

It does not automatically represent current Airbnb market conditions.

## 2. Geographic Scope

The dataset focuses on New York City.

The model should not automatically be assumed to generalize to other cities or countries.

## 3. Limited Feature Types

The current model uses structured tabular information.

It does not directly use:

- Listing images
- Listing descriptions
- Review text
- Detailed amenities
- Host biography
- Real-time market information

## 4. Class Imbalance

The target variable is imbalanced, particularly because Shared room listings are much less common.

Class weighting and Macro F1 were used, but minority-class prediction can still be more difficult.

## 5. No Real-Time Dataset Integration

The prediction system uses the trained model and does not automatically fetch current Airbnb listing information.

---

# 88. Future Improvements

Future versions of this project can include:

- XGBoost
- LightGBM
- Advanced feature engineering
- Natural Language Processing
- Review sentiment analysis
- Listing description analysis
- Image-based Machine Learning
- Deep Learning
- Advanced geospatial features
- Real-time data integration
- Model monitoring
- Automated model retraining
- Model versioning
- Docker deployment
- Cloud deployment
- API authentication
- User authentication
- Database integration
- Interactive analytics dashboard
- Advanced frontend
- Batch prediction
- Model explainability using SHAP

---

# 89. Possible Advanced Architecture

A future production version could use:

```text
Airbnb Data
     ↓
Data Pipeline
     ↓
Data Validation
     ↓
Feature Engineering
     ↓
Feature Store
     ↓
Machine Learning Training
     ↓
Model Registry
     ↓
FastAPI
     ↓
Cloud Deployment
     ↓
Web Application
     ↓
Monitoring
```

---

# 90. Technologies Summary

| Category | Technology |
|---|---|
| Language | Python 3.12.7 |
| Data Analysis | Pandas |
| Numerical Computing | NumPy |
| Visualization | Matplotlib |
| Visualization | Seaborn |
| Machine Learning | Scikit-learn |
| Model Serialization | Joblib |
| API | FastAPI |
| API Server | Uvicorn |
| Validation | Pydantic |
| Frontend | HTML, CSS, JavaScript |
| Dataset Source | Kaggle |
| Dataset Downloader | KaggleHub |
| Runtime | Python 3.12.7 |

---

# 91. Project Files

```text
House Pricing/
│
├── Model_Pipeline.pkl
│   └── Trained Machine Learning pipeline
│
├── house_price.ipynb
│   └── Complete Data Analytics and ML notebook
│
├── main.py
│   └── FastAPI backend
│
├── requirements.txt
│   └── Python dependencies
│
├── runtime.txt
│   └── Python runtime version
│
├── index.html
│   └── Frontend HTML
│
├── script.js
│   └── Frontend JavaScript and API communication
│
├── style.css
│   └── Frontend styling
│
└── the_build_line_guide.html
    └── Supporting HTML resource
```

---

# 92. Complete Backend Dependency File

The project `requirements.txt` contains:

```text
fastapi==0.115.6
uvicorn[standard]==0.34.0
pydantic==2.10.4
pandas==2.2.3
scikit-learn==1.6.1
joblib==1.4.2
```

---

# 93. Runtime Configuration

The project contains:

```text
runtime.txt
```

with:

```text
python-3.12.7
```

---

# 94. FastAPI Backend Structure

The backend performs the following operations:

```text
Load FastAPI
       ↓
Enable CORS
       ↓
Load Model_Pipeline.pkl
       ↓
Define Input Validation
       ↓
Create Health Endpoint
       ↓
Create Prediction Endpoint
       ↓
Receive JSON
       ↓
Validate Input
       ↓
Create DataFrame
       ↓
Predict Room Type
       ↓
Generate Probability
       ↓
Return JSON
```

---

# 95. Frontend Structure

The frontend consists of three main files:

```text
index.html
style.css
script.js
```

### index.html

Provides:

- Page structure
- Input form
- Listing fields
- Prediction panel
- API status

### style.css

Provides:

- Layout
- Typography
- Colors
- Responsive design
- Prediction visualization

### script.js

Provides:

- Form handling
- API requests
- Example data
- Prediction rendering
- Probability visualization
- API health checking

---

# 96. API and Frontend Integration

The frontend sends a POST request:

```javascript
fetch(PREDICT_ENDPOINT, {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
});
```

The backend receives the JSON request and passes the validated features to the saved Machine Learning pipeline.

The response is then displayed on the frontend.

---

# 97. Model Prediction Probability

The backend uses:

```python
prediction = model.predict(row)
probability = model.predict_proba(row)
```

The API returns both:

```text
Predicted Room Type
Prediction Probability
```

This provides more information than returning only the predicted class.

---

# 98. Security and Validation Considerations

The API uses Pydantic validation to reject invalid input values.

Examples:

```text
Invalid latitude
Invalid longitude
Negative price
Invalid minimum nights
Negative review count
Invalid availability
Empty neighbourhood
Empty neighbourhood group
```

These validations help maintain data quality at the API layer.

For production deployment, additional security measures such as authentication, rate limiting, restricted CORS origins, HTTPS, and API monitoring should be added.

---

# 99. Testing the Application

The API can be tested through:

```text
FastAPI Swagger UI
```

at:

```text
http://127.0.0.1:8000/docs
```

The frontend can also be used to test the complete prediction flow.

Testing should include:

```text
Valid Input
Invalid Input
Boundary Values
Different Neighbourhoods
Different Prices
Different Availability
Different Review Counts
```

---

# 100. Conclusion

The **NYC Airbnb Room Type Classification** project demonstrates a complete end-to-end Machine Learning workflow.

The project begins with the public NYC Airbnb dataset and progresses through:

```text
Data Collection
        ↓
Data Understanding
        ↓
Exploratory Data Analysis
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Preprocessing
        ↓
Model Training
        ↓
Model Comparison
        ↓
Cross-Validation
        ↓
Hyperparameter Tuning
        ↓
Model Evaluation
        ↓
Model Serialization
        ↓
FastAPI Deployment
        ↓
Frontend Integration
```

Four Machine Learning algorithms were compared:

```text
Logistic Regression
Decision Tree
Random Forest
Gradient Boosting
```

Random Forest achieved the strongest cross-validation Macro F1 among the tested candidate models and was therefore selected for hyperparameter tuning.

The final tuned model achieved:

```text
Test Accuracy : 85.59%
Test Macro F1 : 74.10%
```

The trained model and preprocessing steps were saved together in:

```text
Model_Pipeline.pkl
```

The model was then integrated with a FastAPI backend and a frontend application, creating a complete system capable of receiving Airbnb listing information and returning a predicted room type.

This project demonstrates practical knowledge of:

```text
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Machine Learning
Data Analytics
Feature Preprocessing
Cross-Validation
Hyperparameter Tuning
Joblib
FastAPI
Pydantic
REST API
HTML
CSS
JavaScript
```

---

# 101. Internship Submission Information

This project was developed for:

```text
AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026
```

Conducted by:

```text
BharatCares in association with AICTE
```

Student:

```text
Shivanand Kumar
```

Degree:

```text
B.Tech - Computer Science and Engineering
```

Project:

```text
NYC Airbnb Room Type Classification
```

---

# 102. Acknowledgement

I would like to express my sincere gratitude to **AICTE, BharatCares, and IBM SkillsBuild** for providing this academic internship opportunity.

This internship provided valuable practical exposure to:

- Data Analytics
- Data Cleaning
- Exploratory Data Analysis
- Machine Learning
- Model Evaluation
- Hyperparameter Tuning
- Machine Learning Deployment
- API Development

I would also like to acknowledge the publicly available **New York City Airbnb Open Data** dataset from Kaggle, which was used for this educational Machine Learning project.

---

# 103. Author

## Shivanand Kumar

**B.Tech - Computer Science and Engineering**

**Project:** NYC Airbnb Room Type Classification

**Internship:** IBM SkillsBuild Data Analytics with AI Internship 2026

**Organization:** BharatCares in association with AICTE

---

# 104. Dataset Reference

New York City Airbnb Open Data:

https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data

---

# 105. Final Project Summary

```text
PROJECT
NYC Airbnb Room Type Classification

STUDENT
Shivanand Kumar

DOMAIN
Data Analytics + Machine Learning + Artificial Intelligence

DATASET
New York City Airbnb Open Data

DATASET SIZE
48,895 Rows × 16 Columns

TARGET
room_type

CLASSES
Entire home/apt
Private room
Shared room

MODELS TESTED
Logistic Regression
Decision Tree
Random Forest
Gradient Boosting

BEST MODEL
Random Forest

BEST PARAMETERS
n_estimators = 200
min_samples_split = 10
max_depth = None
class_weight = balanced

BEST CV MACRO F1
72.9977%

FINAL TEST ACCURACY
85.59%

FINAL TEST MACRO F1
74.10%

MODEL FILE
Model_Pipeline.pkl

BACKEND
FastAPI

API SERVER
Uvicorn

VALIDATION
Pydantic

FRONTEND
HTML + CSS + JavaScript

RUNTIME
Python 3.12.7
```

---

# 106. Educational Use

This project has been developed for educational and academic internship purposes as part of the:

**AICTE | IBM SkillsBuild Data Analytics with AI Internship Program 2026**

The dataset is obtained from a publicly available Kaggle source and remains subject to the terms and conditions of the original dataset provider.

---
