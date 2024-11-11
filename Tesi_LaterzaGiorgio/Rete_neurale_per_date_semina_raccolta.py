import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)


data = pd.read_csv('improved_rice_dataset.csv')


data['Sowing_dayofyear'] = data['Sowing'].apply(lambda x: pd.Timestamp(x).dayofyear)
data['Harvesting_dayofyear'] = data['Harvesting'].apply(lambda x: pd.Timestamp(x).dayofyear)

encoder = OneHotEncoder(sparse_output=False, drop='first')
rice_variety_encoded = encoder.fit_transform(data[['Rice_Variety']])
rice_variety_columns = encoder.get_feature_names_out(['Rice_Variety'])


rice_variety_df = pd.DataFrame(rice_variety_encoded, columns=rice_variety_columns)

data = pd.concat([data, rice_variety_df], axis=1)

original_feature_columns = [
    'SowingRate', 'Nitrogen', 'pH', 'Clay', 'CEC',
    'Avg_Temperature', 'Avg_Irradiation', 'Soil_Moisture', 'Precipitation'
] + list(rice_variety_columns)


def create_fourier_features(df, columns, num_frequencies=3):
    fourier_features = pd.DataFrame()
    for col in columns:
        for k in range(1, num_frequencies+1):
            fourier_features[f'{col}_sin_{k}'] = np.sin(2 * np.pi * k * df[col] / df[col].max())
            fourier_features[f'{col}_cos_{k}'] = np.cos(2 * np.pi * k * df[col] / df[col].max())
    return fourier_features


fourier_columns = ['Avg_Temperature', 'Avg_Irradiation']

fourier_features = create_fourier_features(data, fourier_columns, num_frequencies=3)

data = pd.concat([data, fourier_features], axis=1)

feature_columns = original_feature_columns + list(fourier_features.columns)


X = data[feature_columns]
y = data[['Sowing_dayofyear', 'Harvesting_dayofyear']]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)



model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2)
])


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

model.compile(
    optimizer=optimizer,
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

history = model.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=300,
    validation_split=0.2,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)



test_loss, test_mae = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
print(f'\nTest Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}')


y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)


y_test_sowing = y_test['Sowing_dayofyear'].values
y_pred_sowing = y_pred[:, 0]

y_test_harvesting = y_test['Harvesting_dayofyear'].values
y_pred_harvesting = y_pred[:, 1]


mae_sowing = mean_absolute_error(y_test_sowing, y_pred_sowing)
rmse_sowing = np.sqrt(mean_squared_error(y_test_sowing, y_pred_sowing))
r2_sowing = r2_score(y_test_sowing, y_pred_sowing)

print("\nMetriche per Sowing:")
print(f"MAE: {mae_sowing:.2f} giorni")
print(f"RMSE: {rmse_sowing:.2f} giorni")
print(f"R²: {r2_sowing:.4f}")


mae_harvesting = mean_absolute_error(y_test_harvesting, y_pred_harvesting)
rmse_harvesting = np.sqrt(mean_squared_error(y_test_harvesting, y_pred_harvesting))
r2_harvesting = r2_score(y_test_harvesting, y_pred_harvesting)

print("\nMetriche per Harvesting:")
print(f"MAE: {mae_harvesting:.2f} giorni")
print(f"RMSE: {rmse_harvesting:.2f} giorni")
print(f"R²: {r2_harvesting:.4f}")



plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Perdita di Addestramento')
plt.plot(history.history['val_loss'], label='Perdita di Validazione')
plt.title('Andamento della Perdita durante l\'Addestramento')
plt.xlabel('Epoche')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(y_test_sowing, y_pred_sowing, alpha=0.7)
plt.plot([y_test_sowing.min(), y_test_sowing.max()], [y_test_sowing.min(), y_test_sowing.max()], 'r--')
plt.xlabel('Valori Reali di Sowing (giorno dell\'anno)')
plt.ylabel('Valori Predetti di Sowing (giorno dell\'anno)')
plt.title('Confronto tra Valori Reali e Predetti di Sowing')
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(y_test_harvesting, y_pred_harvesting, alpha=0.7)
plt.plot([y_test_harvesting.min(), y_test_harvesting.max()], [y_test_harvesting.min(), y_test_harvesting.max()], 'r--')
plt.xlabel('Valori Reali di Harvesting (giorno dell\'anno)')
plt.ylabel('Valori Predetti di Harvesting (giorno dell\'anno)')
plt.title('Confronto tra Valori Reali e Predetti di Harvesting')
plt.show()
