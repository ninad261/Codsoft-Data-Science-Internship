import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
file_path = "IMDb Movies India.csv"
df = pd.read_csv(file_path, encoding="ISO-8859-1")

# Data Cleaning
df["Year"] = df["Year"].str.extract("(\d{4})").astype(float)
df["Duration"] = df["Duration"].str.extract("(\d+)").astype(float)
df["Votes"] = df["Votes"].astype(str).str.replace(",", "")
df["Votes"] = pd.to_numeric(df["Votes"], errors='coerce')

# Handle missing values
df["Year"].fillna(df["Year"].median(), inplace=True)
df["Duration"].fillna(df["Duration"].median(), inplace=True)
df.dropna(subset=["Rating"], inplace=True)
categorical_columns = ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]
df[categorical_columns] = df[categorical_columns].fillna("Unknown")

# Feature Engineering
# One-hot encode Genre
genre_encoded = df["Genre"].str.get_dummies(sep=", ")

# Frequency encoding for Director and Actors
for col in ["Director", "Actor 1", "Actor 2", "Actor 3"]:
    freq_encoding = df[col].value_counts().to_dict()
    df[col] = df[col].map(freq_encoding)

# Concatenate one-hot encoded genre with original dataframe
df = pd.concat([df, genre_encoded], axis=1).drop(columns=["Genre", "Name"])

# Split data into train and test sets
X = df.drop(columns=["Rating"])
y = df["Rating"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")