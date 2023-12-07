# Thyroid Type Prediction

## Problem Statement
The objective is to develop a classification methodology for predicting the type of thyroid a person has based on specified features. The dataset comprises 3772 instances and 30 features.

## Data Preprocessing
- Checked dataset shape: (3772, 30).
- Explored the first few rows of the dataset.

### Missing Values Handling
- Identified missing values replaced with '?' in the dataset.
- Replaced '?' with 'nan' and dropped irrelevant columns ('TBG').
- Removed columns indicating the presence of values in subsequent columns.

### Categorical Data Handling
- Mapped binary columns for efficient encoding.
- Utilized LabelEncoder for the output class and get_dummies for columns with more than two categories.

### Imputation of Missing Values
- Utilized KNNImputer for imputing missing values effectively.

### Data Distribution Analysis
- Analyzed and visualized the distribution of continuous data (age, TSH, T3, TT4, T4U, FTI).
- Applied log transformation to skewed data and dropped the 'TSH' column due to undesirable trends.

## Handling Imbalanced Data
- Recognized the highly imbalanced dataset.
- Used RandomOverSampler from imbalanced-learn library to address imbalances.

## Model Training and Metrics
- Trained classification models using the balanced dataset.
- Evaluated model performance using relevant metrics:
  - Accuracy: XX%
  - Precision: XX%
  - Recall: XX%
  - F1-score: XX%

### Model Training (Continued)
- Split the dataset into training and testing sets (80-20 split).
- Utilized a Decision Tree Classifier for model training.

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
```

### Model Prediction
- Predicted the output on the training set.

```python
clf.predict(X_train)
# Output: array([1., 1., 1., ..., 1., 1., 1.])
```

### Model Evaluation
- Evaluated the model performance on the test set.

```python
clf.score(X_test, y_test)
# Output: 0.8913907284768212
```

### Metrics
- **Accuracy:** 89.14%

## Conclusion
- The Decision Tree Classifier achieved an accuracy of 89.14% on the test set.

**Note:** Further model evaluation and optimization can be explored to enhance predictive performance.
