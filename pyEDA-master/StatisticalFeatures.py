import numpy as np
import pandas as pd
from main import process_statistical

# File path for your single CSV file
file_path = "sad_clean2.csv"  # Replace with the actual file path

# Load EDA data from the CSV file using pandas (assuming the first row contains column names)
df = pd.read_csv(file_path)

# Extract the EDA data from the DataFrame
eda_data = df.values  # This assumes that all columns contain EDA sensor data

# Call the function to extract statistical features
m, wd, eda_clean = process_statistical(eda_data, use_scipy=True, sample_rate=25, new_sample_rate=17.5, segment_width=600, segment_overlap=0)

print('m: all the measurements of the signals for each of the segment indices (number of peaks, mean of EDA, maximum value of the peaks)')
print(m)
print('wd: filtered phasic gsr, phasic gsr, tonic gsr, and peacklist for each of the segment indices')
print(wd)
print('eda_clean: preprocessed gsr data')
print(eda_clean)
# Save dictionaries to CSV
for dict_name, dict_values in [("m", m), ("wd", wd)]:
    dict_df = pd.DataFrame.from_dict(dict_values)
    output_file_path = f"{dict_name}.csv"
    dict_df.to_csv(output_file_path, index=False)
    print(f"{dict_name} saved to {output_file_path}")

# Save arrays to CSV
for array_name, array_values in [("eda_clean", eda_clean)]:
    array_df = pd.DataFrame({array_name: array_values})
    output_file_path = f"{array_name}.csv"
    array_df.to_csv(output_file_path, index=False)
    print(f"{array_name} saved to {output_file_path}")
