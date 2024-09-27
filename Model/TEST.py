import joblib
import pandas as pd

# Load the saved model
model = joblib.load('chiller_load_model.joblib')

# Function to prepare input data
def prepare_input(chwr, chws, gpm, temperature, rh, wbt):
    input_data = pd.DataFrame({
        'CHWR': [chwr],
        'CHWS': [chws],
        'GPM': [gpm],
        'Temperature [C]': [temperature],
        'RH [%]': [rh],
        'WBT_C': [wbt]
    })
    return input_data

# Example usage
chwr = 51.0  # Chilled Water Return temperature
chws = 45.5  # Chilled Water Supply temperature
gpm = 1300   # Gallons Per Minute
temperature = 30.0  # Ambient Temperature in Celsius
rh = 65.0    # Relative Humidity in percentage
wbt = 25.0   # Wet Bulb Temperature in Celsius

# Prepare the input
input_data = prepare_input(chwr, chws, gpm, temperature, rh, wbt)

# Make the prediction
prediction = model.predict(input_data)

print(f"Predicted Chiller Load: {prediction[0]:.2f} kW")
