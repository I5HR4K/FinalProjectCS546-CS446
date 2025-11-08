# %%
# Core
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
# from xgboost import XGBClassifier

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# %%
unsw_df = pd.read_csv("UNSW_NB15_training-set.csv")
nsl_df = pd.read_csv("kdd_train.csv")

unsw_df.head(100)
# nsl_df.head()


# %%
# Basic info
unsw_df.info()
unsw_df.isnull().sum()

# Encode categorical columns if any
categorical_cols = unsw_df.select_dtypes(include=['object']).columns
unsw_df[categorical_cols] = unsw_df[categorical_cols].apply(LabelEncoder().fit_transform)

# Fill or drop missing values
unsw_df = unsw_df.dropna()


# %%


# %%
X = unsw_df.drop('label', axis=1)   
y = unsw_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# %%
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB

# Identify categorical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'
)



# Build the pipeline
model = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('classifier', GaussianNB())
])

# Train and predict
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# %%
## models used: bagging, xgb,  decision tree, logistic regression , two nbs

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")



