# Step 1: Install & Import Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 2: Load the dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Step 3: Data Preprocessing

## 3.1 Check for missing values
print("Missing values before processing:\n", df.isnull().sum())

## 3.2 Handle missing values
df["Age"].fillna(df["Age"].median(), inplace=True)  # Fill missing Age with median
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)  # Fill missing Embarked with mode
df.drop(columns=["Cabin"], inplace=True)  # Drop Cabin (too many missing values)

## 3.3 Convert categorical variables
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})  # Convert 'Sex' to binary
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)  # One-hot encode Embarked

print("\nMissing values after processing:\n", df.isnull().sum())
print("\nPreview of cleaned dataset:\n", df.head())

# Step 4: Feature Selection
X = df[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked_Q", "Embarked_S"]]
y = df["Survived"]

# Step 5: Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining size: {X_train.shape[0]}, Testing size: {X_test.shape[0]}")

# Step 6: Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Make Predictions & Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Step 8: Predict Survival for a New Passenger
new_passenger = np.array([[3, 1, 22, 7.25, 1, 0, 0, 1]])  # Example passenger details
prediction = model.predict(new_passenger)
print("\nPrediction for new passenger:", "Survived" if prediction[0] == 1 else "Did Not Survive")
