import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, classification_report, mean_squared_error, r2_score

df = pd.read_csv('exp1_14drivers_14cars_dailyRoutes.csv')

renamed_cols = {
    'VEHICLE_ID':'id',
    'ENGINE_RPM':'rpm',
    'ENGINE_LOAD': 'load',
    'ENGINE_COOLANT_TEMP': 'coolant_temp',
    'SPEED': 'speed'
}

new_df = df.rename(columns=renamed_cols)

new_df = new_df[['id', 'rpm', 'load', 'coolant_temp', 'speed']].dropna()

new_df['id'] = new_df['id'].astype(str).str.strip()
new_df['rpm'] = pd.to_numeric(new_df['rpm'], errors='coerce')
new_df['coolant_temp'] = pd.to_numeric(new_df['coolant_temp'], errors='coerce')

if new_df['load'].dtype == 'str':
    new_df['load'] = new_df['load'].astype(str).str.replace('%', '', regex=False)
    new_df['load'] = new_df['load'].str.replace(',', '.', regex=False)
    new_df['load'] = new_df['load'].str.strip()

new_df['load'] = pd.to_numeric(new_df['load'], errors='coerce')

new_df = new_df.dropna(subset=['id', 'rpm', 'load', 'coolant_temp']).copy()

car_maxes = new_df.groupby('id')[['rpm', 'load']].transform('max')

new_df['rpm_pct'] = (new_df['rpm'] / car_maxes['rpm'])
new_df['load_pct'] = (new_df['load'] / car_maxes['load'])
new_df['coolant_pct'] = (new_df['coolant_temp'] / 115.0)

def compute_stress_level(r):
    if r['coolant_pct'] > 0.88 and r['load_pct'] > 0.80:
        return 2
    elif r['rpm_pct'] > 0.80:
        1
    return 0

new_df['stress_level'] = new_df.apply(compute_stress_level, axis=1)

X = new_df[['rpm_pct', 'load_pct', 'coolant_pct', 'speed']]
y = new_df['stress_level']