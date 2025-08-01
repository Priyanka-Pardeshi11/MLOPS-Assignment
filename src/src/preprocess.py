import pandas as pd
from sklearn.datasets import fetch_california_housing

# Load California Housing dataset
housing = fetch_california_housing(as_frame = True)

# Convert to Dataframe
df = housing.frame
df.rename(columns = {'MedHouseVal' : 'target'} , inplace = true)

raw_path = "data/raw/california_housing.csv"
df.to_csv(raw_path, index=False)
print(f"Raw California Housing dataset saved to {raw_path}")

# Normalize numeric features
features = df.columns[:-1]  # all columns except target column
df_normalized = df.copy()

df_normalized[features] = (
    df[features] - df[features].min()
) / (df[features].max() - df[features].min())

processed_path = "data/processed/california_housing_processed.csv"
df_normalized.to_csv(processed_path , index=False)
print(f" Processed California Housing dataset saved to {processed_path}")

