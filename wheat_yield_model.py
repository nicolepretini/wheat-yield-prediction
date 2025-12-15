import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import pandas as pd


# 1. Load data
df = pd.read_csv("faostat_qcl_yield_all.csv")

# Quick sanity check
print("Columns:", df.columns.tolist())
print("Head:")
print(df.head())

# 2. Keep only wheat
# Filter rows for wheat
wheat = df[df["Item"].str.contains("Wheat", case=False, na=False)]

# Keep only yield element
wheat = wheat[wheat["Element"].str.contains("Yield", case=False, na=False)]

wheat = wheat.rename(columns={
    "Area": "Region",
    "Value": "Yield_ton_ha"
})

# Keep only relevant columns
wheat = wheat[["Year", "Region", "Yield_ton_ha"]].dropna()

# Sort chronologically (CRITICAL for temporal validation)
wheat = wheat.sort_values("Year").reset_index(drop=True)

X = wheat[["Year", "Region"]]
y = wheat["Yield_ton_ha"]

print(wheat.head())
print("Rows:", len(wheat))
print("\nWheat subset head:")
print(wheat.head())
print("\nNumber of wheat rows:", len(wheat))

# 4. Define features and target
X = wheat[["Year", "Region"]]
y = wheat["Yield_ton_ha"]

# 6. Preprocess: one-hot encode Region, pass Year as numeric
numeric_features = ["Year"]
categorical_features = ["Region"]

numeric_transformer = "passthrough"
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 7. Model 1: Linear Regression
linreg_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

# 8. Model 2: Random Forest
rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

# =========================
# Rolling 10-year evaluation
# =========================

window = 10
results = []

min_year = int(wheat["Year"].min())
max_year = int(wheat["Year"].max())

# Start after at least 20 years of training data to avoid tiny training sets
for split_year in range(min_year + 20, max_year - window + 1, window):
    train_mask = wheat["Year"] <= split_year
    test_mask = (wheat["Year"] > split_year) & (wheat["Year"] <= split_year + window)

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    # Skip if not enough test data in that decade
    if len(X_test) < 30:
        continue

    # Fit models
    linreg_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    # Predict
    y_pred_lin = linreg_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)

    # Metrics (RMSE computed manually for your sklearn version)
    rmse_lin = mean_squared_error(y_test, y_pred_lin) ** 0.5
    r2_lin = r2_score(y_test, y_pred_lin)

    rmse_rf = mean_squared_error(y_test, y_pred_rf) ** 0.5
    r2_rf = r2_score(y_test, y_pred_rf)

    results.append({
        "train_until": split_year,
        "test_period": f"{split_year+1}-{split_year+window}",
        "n_test": len(X_test),
        "rmse_lin": rmse_lin,
        "r2_lin": r2_lin,
        "rmse_rf": rmse_rf,
        "r2_rf": r2_rf
    })


results_df = pd.DataFrame(results)

print("\n=== Rolling 10-year results ===")
print(results_df)

print("\nSummary (means across decades):")
print(results_df[["rmse_lin", "r2_lin", "rmse_rf", "r2_rf"]].mean())
