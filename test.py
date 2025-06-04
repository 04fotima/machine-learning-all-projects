import os
import pandas as pd

# Print the current working directory
print("Current working directory:", os.getcwd())

# Attempt to load data from the correct path
try:
    data = pd.read_csv('C:/Users/Sozla.uz/.vscode/.vscode/Online_Retail.csv')
except FileNotFoundError as e:
    print(f"Error: {e}")
