import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Crop_recommendation.csv')

print(data.head())
print(data['label'].value_counts())


X = data.drop('label', axis=1).values
y = data['label'].values


label_encoder = LabelEncoder()
y_integer = label_encoder.fit_transform(y)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_integer, test_size=0.2, random_state=42, stratify=y_integer
)

n_features = X.shape[1]
n_classes = len(np.unique(y_integer))

# Creazione del modello
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(n_features,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callback per l'early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)


history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=16,
    callbacks=[early_stopping],
    verbose=1
)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')


y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)


print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))


conf_matrix = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Perdita di addestramento')
plt.plot(history.history['val_loss'], label='Perdita di validazione')
plt.title('Andamento della perdita')
plt.xlabel('Epoca')
plt.ylabel('Perdita')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Accuratezza di addestramento')
plt.plot(history.history['val_accuracy'], label='Accuratezza di validazione')
plt.title('Andamento dell\'accuratezza')
plt.xlabel('Epoca')
plt.ylabel('Accuratezza')
plt.legend()

plt.tight_layout()
plt.show()

numeric_data = data.select_dtypes(include=[np.number])


corr_matrix = numeric_data.corr()


plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Matrice di Correlazione delle Feature Numeriche')
plt.show()

numeric_columns = numeric_data.columns.tolist()

for column in numeric_columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='label', y=column, data=data)
    plt.title(f'Boxplot di {column} per ciascuna coltura')
    plt.xlabel('Coltura')
    plt.ylabel(column)
    plt.xticks(rotation=90)
    plt.show()
