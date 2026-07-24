    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    import matplotlib.pyplot as plt
    import seaborn as sns

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

    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_probs = model.predict_proba(X_test)
    classes = model.classes_

    miss_costs = {0: 0, 1: 5000, 2: 2000, 3: 500, 4: 1000, 5: 300}
    class_costs = np.array([miss_costs.get(c, 0) for c in classes])
    expected_ignore_costs = np.dot(y_probs, class_costs)

    decisions = (expected_ignore_costs > 25).astype(int)

    wasted_inspections = 0
    prevented_failures = 0
    unprevented_failures_cost = 0
    total_preventive_cost = 0

    for true_val, decision in zip(y_test, decisions):
       if decision == 1:
           total_preventive_cost += 50
           if true_val == 0:
               wasted_inspections += 1
           else:
               prevented_failures += 1
       else:
           if true_val > 0:
               unprevented_failures_cost += miss_costs.get(true_val, 0)

    total_loss = total_preventive_cost + unprevented_failures_cost

    print(f"Preventive Maintenance Cost: ${total_preventive_cost:,}")
    print(f"Wasted Inspections (False Alarms): {wasted_inspections}")
    print(f"Prevented Catastrophic Failures: {prevented_failures}")
    print(f"Cost of Missed Failures: ${unprevented_failures_cost:,}")
    print(f"Total Operational Loss: ${total_loss:,}")

    importances = model.feature_importances_
    feature_names = X.columns

    sensor_labels = {
       'load_pct': 'Engine Load %',
       'coolant_pct': 'Coolant Temp %',
       'rpm_pct': 'Engine RPM % ',
       'speed': 'Vehicle Speed'
    }

    feat_df = pd.DataFrame({
       'Sensor': [sensor_labels.get(col, col) for col in feature_names],
       'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(9, 4.5))
    sns.barplot(
       x='Importance',
       y='Sensor',
       data=feat_df,
       hue='Sensor',
       palette='Reds_r',
       legend=False
    )

    plt.title("Inside the Model: Which Sensors Matter Most?", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Relative Influence on Failure Prediction", fontsize=11)
    plt.ylabel("Car Feature", fontsize=11)
    plt.xlim(0, 0.5)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()