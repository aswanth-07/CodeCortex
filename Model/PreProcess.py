import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Read the CSV files
df1 = pd.read_csv('Dataset/TableData (6).csv')
df2 = pd.read_csv('Dataset/sensor data.csv')

# Print column names to verify
print("Columns in df1:", df1.columns)
print("Columns in df2:", df2.columns)

# Identify the datetime column in df1 (assuming it's the 'Time' column)
datetime_col_df1 = 'Time'

# Convert DateTime to datetime object
df1[datetime_col_df1] = pd.to_datetime(df1[datetime_col_df1])
df2['DateTime'] = pd.to_datetime(df2['DateTime'])

# Set DateTime as index
df1.set_index(datetime_col_df1, inplace=True)
df2.set_index('DateTime', inplace=True)

# Resample data to hourly frequency
df1_hourly = df1.resample('h').mean()
df2_hourly = df2.resample('h').mean()

# Merge the two dataframes
merged_df = pd.merge(df1_hourly, df2_hourly, left_index=True, right_index=True, suffixes=('_1', '_2'))

# Identify temperature and humidity columns in df1
temp_col_df1 = 'CHWR'  # Chilled Water Return Temperature
rh_col_df1 = 'RH [%]_2'  # Using RH from df2 as df1 doesn't seem to have RH

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(15, 20))
fig.suptitle('Sensor Data Comparison', fontsize=16)

# Plot Temperature
axs[0].plot(merged_df.index, merged_df[temp_col_df1], label='Sensor 1 (CHWR)')
axs[0].plot(merged_df.index, merged_df['Temperature [C]'], label='Sensor 2')
axs[0].set_ylabel('Temperature [C]')
axs[0].legend()
axs[0].grid(True)

# Plot Relative Humidity (only from Sensor 2)
axs[1].plot(merged_df.index, merged_df['RH [%]'], label='Sensor 2')
axs[1].set_ylabel('Relative Humidity [%]')
axs[1].legend()
axs[1].grid(True)

# Plot Wet Bulb Temperature
axs[2].plot(merged_df.index, merged_df['WBT_C'], label='Sensor 2')
axs[2].set_ylabel('Wet Bulb Temperature [C]')
axs[2].legend()
axs[2].grid(True)

# Set x-axis label for the bottom subplot
axs[2].set_xlabel('Date')

# Rotate x-axis labels for better readability
for ax in axs:
    ax.tick_params(axis='x', rotation=45)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('sensor_data_comparison.png')
plt.close()

# Calculate and print statistics
print("\nTemperature Statistics:")
print(merged_df[[temp_col_df1, 'Temperature [C]']].describe())

print("\nRelative Humidity Statistics (Sensor 2 only):")
print(merged_df['RH [%]'].describe())

print("\nWet Bulb Temperature Statistics:")
print(merged_df['WBT_C'].describe())

# Calculate correlations
temp_corr = merged_df[temp_col_df1].corr(merged_df['Temperature [C]'])

print(f"\nTemperature Correlation between Sensor 1 (CHWR) and Sensor 2: {temp_corr:.4f}")