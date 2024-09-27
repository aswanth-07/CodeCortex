import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load and preprocess the data (reusing code from PreProcess.py)
df1 = pd.read_csv('Dataset/TableData (6).csv')
df2 = pd.read_csv('Dataset/sensor data.csv')

datetime_col_df1 = 'Time'
df1[datetime_col_df1] = pd.to_datetime(df1[datetime_col_df1])
df2['DateTime'] = pd.to_datetime(df2['DateTime'])

df1.set_index(datetime_col_df1, inplace=True)
df2.set_index('DateTime', inplace=True)

df1_hourly = df1.resample('h').mean()
df2_hourly = df2.resample('h').mean()

merged_df = pd.merge(df1_hourly, df2_hourly, left_index=True, right_index=True, suffixes=('_1', '_2'))

# Select features for the model
features = ['CHWR', 'CHWS', 'GPM', 'Temperature [C]', 'RH [%]', 'WBT_C']
target = 'CH Load'

# Clean the data
merged_df = merged_df.dropna(subset=features + [target])

# Prepare the data
X = merged_df[features]
y = merged_df[target]

# Print some information about the data
print("Data shape after cleaning:", X.shape)
print("\nFeature statistics:")
print(X.describe())
print("\nTarget variable statistics:")
print(y.describe())

# Check for any remaining NaN values
print("\nRemaining NaN values:")
print(X.isna().sum())
print(y.isna().sum())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Plot feature importances
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.barh(pos, feature_importance[sorted_idx], align='center')
ax1.set_yticks(pos)
ax1.set_yticklabels(np.array(features)[sorted_idx])
ax1.set_title('Feature Importance')

# Plot actual vs predicted values
ax2.scatter(y_test, y_pred, alpha=0.5)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('Actual Chiller Load')
ax2.set_ylabel('Predicted Chiller Load')
ax2.set_title('Actual vs Predicted Chiller Load')

plt.tight_layout()
plt.savefig('chiller_load_model_results.png')
plt.close()

# Print example predictions
print("\nExample Predictions:")
example_data = X_test.head(5)
example_predictions = model.predict(example_data)
for i, (actual, predicted) in enumerate(zip(y_test.head(5), example_predictions)):
    print(f"Sample {i+1}: Actual: {actual:.2f}, Predicted: {predicted:.2f}")

# Save the model
import joblib
joblib.dump(model, 'chiller_load_model.joblib')
print("\nModel saved as 'chiller_load_model.joblib'")