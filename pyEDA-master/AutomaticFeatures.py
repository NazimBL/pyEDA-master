from os import makedirs

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from main import prepare_automatic, process_automatic


# File path for your single CSV file
file_path = "happy_data.csv"  # Replace with the actual file path

# Create directories if they don't exist
checkpoint_dir = 'pyEDA/pyEDA'
makedirs(checkpoint_dir, exist_ok=True)
# Load EDA data from the CSV file using pandas (assuming the first row contains column names)
df = pd.read_csv(file_path)

# Extract the EDA data from the DataFrame
eda_data = df.values  # This assumes that all columns contain EDA sensor data

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the EDA data to the 0-1 range
eda_data_normalized = scaler.fit_transform(eda_data)

# Ensure that the normalized data is in the correct shape
eda_data_normalized = np.squeeze(eda_data_normalized)

print(eda_data.shape)

# Train and save the autoencoder model
prepare_automatic(eda_data_normalized, sample_rate=25, new_sample_rate=17.5, k=32, epochs=100, batch_size=10)

# Choose a specific EDA signal to process (you can change the index)
eda_signal_to_process = eda_data_normalized[0]



# Process the EDA signal using the autoencoder
automatic_features = process_automatic(eda_signal_to_process)

# Print the extracted automatic features
print(automatic_features)







