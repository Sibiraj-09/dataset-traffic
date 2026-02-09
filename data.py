import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# 1. Load Dataset
df = pd.read_csv("traffic_dataset.csv")
print("✅ Dataset Loaded Successfully!")
print(df.head())
# 2. Dataset Information
print("\nDataset Info:")
print(df.info())
# 3. Drop Unwanted Column
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)
print("\n✅ Removed Unwanted Column")
print(df.columns)
# 4. Check Missing Values
print("\nMissing Values Count:")
print(df.isnull().sum())
# Fill missing values if any
df.fillna(df.mean(numeric_only=True), inplace=True)
print("\n✅ Missing Values Handled!")
# 5. Encode Categorical Column
if "time_of_day" in df.columns:
    df = pd.get_dummies(df, columns=["time_of_day"], drop_first=True)
print("\n✅ Categorical Encoding Done!")
print(df.head())
# 6. Feature Scaling (Standardization)
scaler = StandardScaler()
numeric_cols = ["vehicle_count", "average_speed",
                "lane_occupancy", "flow_rate", "waiting_time"]
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print("\n✅ Feature Scaling Completed!")
# 7. Split Dataset into Train & Test
X = df.drop("waiting_time", axis=1)   # Features
y = df["waiting_time"]               # Target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\n✅ Train-Test Split Done!")
print("Training Data Shape:", X_train.shape)
print("Testing Data Shape :", X_test.shape)
# 8. Save Preprocessed Dataset
df.to_csv("traffic_preprocessed.csv", index=False)
print("\n✅ Preprocessed Dataset Saved as 'traffic_preprocessed.csv'")
# Done
print("\n🎉 FULL PREPROCESSING COMPLETED SUCCESSFULLY!")
