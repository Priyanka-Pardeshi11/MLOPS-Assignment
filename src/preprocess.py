import pandas as pd
from sklearn.datasets import fetch_california_housing

# -----------------------------
# STEP 1: Load California Housing Dataset
# -----------------------------
housing = fetch_california_housing(as_frame=True)

# Convert to DataFrame (features + target)
df = housing.frame
df.rename(columns={'MedHouseVal': 'target'}, inplace=True)

# -----------------------------
# STEP 2: Save Raw Dataset
# -----------------------------
raw_path = "data/raw/california_housing.csv"
df.to_csv(raw_path, index=False)
print(f"✅ Raw California Housing dataset saved to {raw_path}")

# -----------------------------
# STEP 3: Preprocess Dataset
# - Normalize numeric features
# -----------------------------
features = df.columns[:-1]  # all columns except target
df_normalized = df.copy()

# Min-Max normalization
df_normalized[features] = (
    df[features] - df[features].min()
) / (df[features].max() - df[features].min())

# -----------------------------
# STEP 4: Save Processed Dataset
# -----------------------------
processed_path = "data/processed/california_housing_processed.csv"
df_normalized.to_csv(processed_path, index=False)
print(f"✅ Processed California Housing dataset saved to {processed_path}")

