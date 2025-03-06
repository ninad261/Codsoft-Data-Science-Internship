import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE

# Load the dataset from CSV file
file_path = r"D:\System\Desktop\Codsoft Data Science IInternship\Credit Card Fraud Detection\creditcard.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset not found at {file_path}")

df = pd.read_csv(file_path)
print("Dataset loaded successfully!")

# Check for missing values
df.fillna(df.median(), inplace=True)
print("Missing values handled.")

# Split features and target
X = df.iloc[:, :-1].values  # Transaction features
y = df.iloc[:, -1].values  # Fraudulent (1) or Genuine (0)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
print("Class imbalance handled using SMOTE.")

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split into training ({X_train.shape[0]}) and testing ({X_test.shape[0]}) sets.")

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Features standardized.")

# Train a RandomForest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)
print("Model training completed.")

# Make predictions
y_pred = classifier.predict(X_test)
print("Predictions made.")

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {f1:.2f}')
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
