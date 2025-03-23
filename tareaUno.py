import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Generación de datos de prueba
np.random.seed(42)
n_samples = 500

data = {
    'num_creditos_vigentes': np.random.randint(1, 5, n_samples),
    'porcentaje_pago': np.random.uniform(50, 100, n_samples),
    'saldo_mora': np.random.uniform(0, 5000, n_samples),
    'dias_mora': np.random.randint(0, 120, n_samples),
    'antiguedad_credito': np.random.randint(1, 10, n_samples),
    'plazo_credito': np.random.randint(6, 60, n_samples),
    'num_microseguros': np.random.randint(0, 3, n_samples),
    'tasa_interes': np.random.uniform(5, 25, n_samples),
    'estado_ahorro': np.random.choice(["Activo", "Inactivo"], n_samples),
    'saldo_ahorro': np.random.uniform(0, 10000, n_samples),
    'edad': np.random.randint(18, 65, n_samples),
    'estado_civil': np.random.choice(["Soltero", "Casado", "Divorciado"], n_samples),
    'estrato': np.random.randint(1, 6, n_samples),
    'segmento': np.random.choice(["Alto", "Medio", "Bajo"], n_samples),
    'desercion': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
}

df = pd.DataFrame(data)

# Preprocesamiento de datos
le = LabelEncoder()
for col in ['estado_ahorro', 'estado_civil', 'segmento']:
    df[col] = le.fit_transform(df[col])

scaler = StandardScaler()
X = df.drop(columns=['desercion'])
y = df['desercion']
X_scaled = scaler.fit_transform(X)

# División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo de regresión logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluación del modelo
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))

# Visualización de datos
plt.figure(figsize=(10, 5))
sns.countplot(x='desercion', data=df, palette='coolwarm')
plt.title("Distribución de clientes que desertan (1) vs. los que permanecen (0)")
plt.show()

# Mostrar las primeras filas del dataset
df.head()
