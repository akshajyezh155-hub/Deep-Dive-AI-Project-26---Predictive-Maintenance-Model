import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

df = pd.read_csv('exp1_14drivers_14cars_dailyRoutes.csv')

renamed_cols = {
    'VEHICLE_ID': 'id',
    'ENGINE_RPM': 'rpm',
    'ENGINE_LOAD': 'load',
    'ENGINE_COOLANT_TEMP': 'coolant_temp',
    'SPEED': 'speed'
}
new_df = df.rename(columns=renamed_cols)

def map_trouble_code(code):
    code = str(code).strip().upper()
    if code in ['NORMAL', '', 'NAN'] or pd.isna(code):
        return 0
    elif code.startswith('P'):
        return 1
    elif code.startswith('C'):
        return 2
    elif code.startswith('B'):
        return 3
    elif code.startswith('U'):
        return 4
    else:
        return 5

new_df['failure_type'] = new_df['TROUBLE_CODES'].apply(map_trouble_code)

new_df = new_df[['id', 'rpm', 'load', 'coolant_temp', 'speed', 'failure_type']]

new_df['id'] = new_df['id'].astype(str).str.strip()

for col in ['rpm', 'load', 'coolant_temp', 'speed']:
    new_df[col] = new_df[col].astype(str).str.replace('%', '', regex=False)
    new_df[col] = new_df[col].str.replace(',', '.', regex=False)
    new_df[col] = new_df[col].str.strip()
    new_df[col] = pd.to_numeric(new_df[col], errors='coerce')

new_df = new_df.dropna(subset=['id', 'rpm', 'load', 'coolant_temp', 'speed', 'failure_type']).copy()

car_maxes = new_df.groupby('id')[['rpm', 'load']].transform('max')

new_df['rpm_pct'] = (new_df['rpm'] / car_maxes['rpm'])
new_df['load_pct'] = (new_df['load'] / car_maxes['load'])
new_df['coolant_pct'] = (new_df['coolant_temp'] / 115.0)

X = new_df[['rpm_pct', 'load_pct', 'coolant_pct', 'speed']]
y = new_df['failure_type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sgd_model = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=0.01, random_state=42)
classes = np.unique(y)

for epoch in range(1, 21):
    sgd_model.partial_fit(X_train_scaled, y_train, classes=classes)
    y_train_prob = sgd_model.predict_proba(X_train_scaled)
    epoch_loss = log_loss(y_train, y_train_prob, labels=classes)
    print(f"Epoch {epoch:02d} | Cross-Entropy Loss: {epoch_loss:.5f}")