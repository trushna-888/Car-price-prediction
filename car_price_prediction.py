# ==========================================
# CAR PRICE PREDICTOR - PART 1
# ==========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# -------------------------------
# LOAD DATASET
# -------------------------------

df = pd.read_csv("quikr_car.csv")

print("Loading Dataset...")

# -------------------------------
# DATA CLEANING
# -------------------------------

# Remove Ask For Price rows
df = df[df["Price"] != "Ask For Price"]

# Price
df["Price"] = df["Price"].astype(str)
df["Price"] = df["Price"].str.replace(",", "", regex=False)
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

# Year
df["year"] = pd.to_numeric(df["year"], errors="coerce")

# Kilometers
df["kms_driven"] = df["kms_driven"].astype(str)
df["kms_driven"] = df["kms_driven"].str.replace(",", "", regex=False)
df["kms_driven"] = df["kms_driven"].str.replace(" kms", "", regex=False)
df["kms_driven"] = pd.to_numeric(df["kms_driven"], errors="coerce")

# Fuel Type
df["fuel_type"] = df["fuel_type"].fillna("Petrol")

# Remove invalid rows
df.dropna(inplace=True)

# Convert datatype
df["Price"] = df["Price"].astype(int)
df["year"] = df["year"].astype(int)
df["kms_driven"] = df["kms_driven"].astype(int)

# Short Name
df["name"] = df["name"].apply(
    lambda x: " ".join(str(x).split()[:3])
)

df.reset_index(drop=True, inplace=True)

print("Dataset Loaded Successfully!")
print("Total Cars :", len(df))
# ==========================================
# CAR PRICE PREDICTOR - PART 2
# MODEL TRAINING
# ==========================================

# Features and Target
X = df[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = df['Price']

# One Hot Encoding
column_transformer = ColumnTransformer(
    transformers=[
        ('encoder',
         OneHotEncoder(handle_unknown='ignore'),
         ['name', 'company', 'fuel_type'])
    ],
    remainder='passthrough'
)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Random Forest Model
model = Pipeline([
    ('preprocessor', column_transformer),
    ('regressor', RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

print("\nTraining Model...")

model.fit(X_train, y_train)

print("Model Trained Successfully!")

# ---------------------------------------
# AVAILABLE VALUES
# ---------------------------------------

companies = sorted(df['company'].unique())

years = sorted(df['year'].unique())

fuel_types = sorted(df['fuel_type'].unique())

print("\n====================================")
print("        CAR PRICE PREDICTOR")
print("====================================")

print("\nAvailable Companies:")

for i, company in enumerate(companies, start=1):
    print(f"{i}. {company}")

company = input("\nEnter Company: ")

models = sorted(
    df[df['company'] == company]['name'].unique()
)

if len(models) == 0:
    print("Invalid Company Name")
    exit()

print("\nAvailable Models:")

for i, car in enumerate(models, start=1):
    print(f"{i}. {car}")

car_name = input("\nEnter Model: ")

year = int(input("\nEnter Purchase Year: "))

print("\nAvailable Fuel Types:")

for i, fuel in enumerate(fuel_types, start=1):
    print(f"{i}. {fuel}")

fuel = input("\nEnter Fuel Type: ")

kms = int(input("\nEnter Kilometers Driven: "))
# ==========================================
# CAR PRICE PREDICTOR - PART 3
# PRICE PREDICTION
# ==========================================

input("\n------------------------------------")
input("Press ENTER to Predict Price...")

# Create Input DataFrame
input_df = pd.DataFrame({
    "name": [car_name],
    "company": [company],
    "year": [year],
    "kms_driven": [kms],
    "fuel_type": [fuel]
})

# Predict Price
predicted_price = model.predict(input_df)

print("\n====================================")
print("        PREDICTION RESULT")
print("====================================")

print(f"\nCompany            : {company}")
print(f"Model              : {car_name}")
print(f"Purchase Year      : {year}")
print(f"Fuel Type          : {fuel}")
print(f"Kilometers Driven  : {kms}")

print("\n------------------------------------")
print("Predicted Price")
print(f"₹ {predicted_price[0]:,.0f}")
print("------------------------------------")

print("\nThank You for Using Car Price Predictor!")