# train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
df = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")

# Drop car name (not very useful for this model directly)
df = df.drop("name", axis=1)

# Encode categorical variables
le_fuel = LabelEncoder()
le_seller = LabelEncoder()
le_trans = LabelEncoder()
le_owner = LabelEncoder()

df["fuel"] = le_fuel.fit_transform(df["fuel"])
df["seller_type"] = le_seller.fit_transform(df["seller_type"])
df["transmission"] = le_trans.fit_transform(df["transmission"])
df["owner"] = le_owner.fit_transform(df["owner"])

# Features and target
X = df.drop("selling_price", axis=1)
y = df["selling_price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model training
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.4f}")

# Save model and encoders
joblib.dump(model, "car_price_model.pkl")
joblib.dump(le_fuel, "le_fuel.pkl")
joblib.dump(le_seller, "le_seller.pkl")
joblib.dump(le_trans, "le_trans.pkl")
joblib.dump(le_owner, "le_owner.pkl")
