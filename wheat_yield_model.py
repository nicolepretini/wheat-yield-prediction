import pandas as pd
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

print(wheat.head())
print("Rows:", len(wheat))
print("\nWheat subset head:")
print(wheat.head())
print("\nNumber of wheat rows:", len(wheat))

# 3. Drop obvious missing values
wheat = wheat.dropna(subset=["Yield_ton_ha", "Year", "Region"])

# 4. Define features and target
X = wheat[["Year", "Region"]]
y = wheat["Yield_ton_ha"]

# 5. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

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

linreg_model.fit(X_train, y_train)
y_pred_lin = linreg_model.predict(X_test)

mse_lin = mean_squared_error(y_test, y_pred_lin)
rmse_lin = mse_lin ** 0.5
r2_lin = r2_score(y_test, y_pred_lin)

print("\n=== Linear Regression ===")
print("RMSE:", rmse_lin)
print("R²:", r2_lin)

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

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = mse_rf ** 0.5
r2_rf = r2_score(y_test, y_pred_rf)

print("\n=== Random Forest ===")
print("RMSE:", rmse_rf)
print("R²:", r2_rf)

# 9. Plot: actual vs predicted (Random Forest)
plt.scatter(y_test, y_pred_rf, alpha=0.6)
plt.xlabel("Actual yield (ton/ha)")
plt.ylabel("Predicted yield (ton/ha)")
plt.title("Wheat yield – Random Forest\nActual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()
