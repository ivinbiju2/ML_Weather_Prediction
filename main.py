# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

kaggle_data_path = "weatherHistory.csv"
tensorflow_data_path = "mpi_saale.csv"

try:
    kaggle_data = pd.read_csv(kaggle_data_path)
    tensorflow_data = pd.read_csv(tensorflow_data_path)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

kaggle_data = kaggle_data.dropna(thresh=len(kaggle_data.columns) * 0.7)  # Retain rows with at least 70% data
tensorflow_data = tensorflow_data.dropna(thresh=len(tensorflow_data.columns) * 0.7)

# Fill numeric columns with the mean
for df in [kaggle_data, tensorflow_data]:
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Normalize data
def normalize_data(df, columns):
    for col in columns:
        if col in df.columns:  # Ensure column exists before normalization
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

# Kaggle dataset features
kaggle_features = ['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)']

# TensorFlow dataset features
tensorflow_features = [
    "T (degC)", "rh (%)", "wv (m/s)", "p (mbar)"
]  # Selected features for similarity to Kaggle dataset

# Normalize selected features
kaggle_data = normalize_data(kaggle_data, kaggle_features)
tensorflow_data = normalize_data(tensorflow_data, tensorflow_features)

# Combine datasets (if applicable) or train models separately
# Only combine columns that exist in both datasets
common_features = ['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)',
                   "T (degC)", "rh (%)", "wv (m/s)", "p (mbar)"]

# Rename TensorFlow columns to align with Kaggle naming conventions
rename_mapping = {
    "T (degC)": "Temperature (C)",
    "rh (%)": "Humidity",
    "wv (m/s)": "Wind Speed (km/h)",
    "p (mbar)": "Pressure (millibars)"
}
tensorflow_data = tensorflow_data.rename(columns=rename_mapping)

# Ensure both datasets have the same features for combination
common_features = list(rename_mapping.values())
kaggle_data = kaggle_data[common_features]
tensorflow_data = tensorflow_data[common_features]

combined_data = pd.concat([kaggle_data, tensorflow_data])

# Step 3: Define input (X) and output (y) variables
# Here we are predicting 'Temperature (C)' as the target
if 'Temperature (C)' in combined_data.columns:
    X = combined_data.drop(columns=['Temperature (C)'], errors='ignore')  # Remove 'Temperature (C)' from inputs
    y = combined_data['Temperature (C)']  # Target prediction is temperature here
else:
    print("Error: 'Temperature (C)' column not found in the dataset.")
    exit()

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = linear_model.predict(X_test)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Step 7: Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linewidth=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Temperature (C)")
plt.show()
