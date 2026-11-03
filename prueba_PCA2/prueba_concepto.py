import time
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

# 1. CARGA DE DATOS (UNSW-NB15)
archivos_csv = glob.glob("*UNSW_NB15*.csv")
if not archivos_csv:
    raise FileNotFoundError("Coloca los archivos CSV de UNSW-NB15 en la misma carpeta.")

df_lista = [pd.read_csv(f) for f in archivos_csv]
df = pd.concat(df_lista, ignore_index=True)
print(f"Total de registros cargados: {len(df)}")

# Guardar categoría de ataque para análisis cualitativo antes de limpiar
categorias_ataque = df['attack_cat'].fillna('Normal') if 'attack_cat' in df.columns else None

# Separar la etiqueta binaria (0 = Normal, 1 = Ataque) y eliminar columnas no predictoras
y = df['label'].values
columnas_excluir = [col for col in ['id', 'label', 'attack_cat'] if col in df.columns]
X_df = df.drop(columns=columnas_excluir).copy()

# Codificar variables categóricas (proto, service, state) para conservar los atributos
for col in X_df.select_dtypes(include=['object', 'category']).columns:
    X_df[col] = pd.factorize(X_df[col])[0]

# Limpieza de posibles valores infinitos o nulos
X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0)
nombres_variables = X_df.columns
print(f"Dimensión de la matriz de datos X: {X_df.shape[0]} filas x {X_df.shape[1]} columnas")

# División Entrenamiento (70%) y Prueba (30%)
X_train_raw, X_test_raw, y_train, y_test, cat_train, cat_test = train_test_split(
    X_df.values, y, categorias_ataque, test_size=0.30, random_state=42, stratify=y
)

# 2. PREPROCESAMIENTO: ESTANDARIZACIÓN Z-SCORE (Media = 0, Desviación = 1)
scaler = StandardScaler()
Z_train = scaler.fit_transform(X_train_raw)
Z_test = scaler.transform(X_test_raw)

# 3. APLICACIÓN DE PCA Y SELECCIÓN DE k PARA >90% DE VARIANZA
pca_full = PCA()
pca_full.fit(Z_train)
varianza_ind = pca_full.explained_variance_ratio_ * 100
varianza_acum = np.cumsum(varianza_ind)

# Encontrar k mínimo que supera el 90% de varianza explicada
k_optimo = int(np.argmax(varianza_acum >= 90.0) + 1)
print(f"\n--- RESULTADOS PCA ---")
print(f"Componentes necesarios para >= 90% de varianza (k): {k_optimo} de {X_df.shape[1]}")
print(f"Varianza exacta retenida con k={k_optimo}: {varianza_acum[k_optimo-1]:.2f}%")

# Proyección al subespacio reducido con k_optimo
pca_k = PCA(n_components=k_optimo, random_state=42)
Z_train_pca = pca_k.fit_transform(Z_train)
Z_test_pca = pca_k.transform(Z_test)

# Cálculo del error relativo de reconstrucción (Norma de Frobenius)
Z_test_reconstruido = pca_k.inverse_transform(Z_test_pca)
error_reconstruccion = (np.linalg.norm(Z_test - Z_test_reconstruido, 'fro') / 
                        np.linalg.norm(Z_test, 'fro')) * 100
print(f"Error relativo de reconstrucción (Frobenius): {error_reconstruccion:.2f}%")

# 4. EVALUACIÓN DEL CLASIFICADOR IDS: LÍNEA BASE (SIN PCA) VS CON PCA
def evaluar_modelo(X_tr, X_te, y_tr, y_te, nombre):
    clf = DecisionTreeClassifier(max_depth=15, random_state=42)
    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    t_entrenamiento = time.perf_counter() - t0
    
    t1 = time.perf_counter()
    y_pred = clf.predict(X_te)
    t_inferencia = time.perf_counter() - t1
    
    acc = accuracy_score(y_te, y_pred) * 100
    prec = precision_score(y_te, y_pred) * 100
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    far = (fp / (fp + tn)) * 100  # Tasa de Falsas Alarmas
    
    print(f"\n[{nombre}]")
    print(f"  Exactitud (Accuracy):      {acc:.2f} %")
    print(f"  Precisión (Precision):     {prec:.2f} %")
    print(f"  Falsas Alarmas (FAR):      {far:.2f} %")
    print(f"  Tiempo de entrenamiento:   {t_entrenamiento:.4f} s")
    print(f"  Tiempo de inferencia:      {t_inferencia:.4f} s")
    return y_pred

y_pred_base = evaluar_modelo(Z_train, Z_test, y_train, y_test, f"LÍNEA BASE ({X_df.shape[1]} variables)")
y_pred_pca = evaluar_modelo(Z_train_pca, Z_test_pca, y_train, y_test, f"MODELO CON PCA (k={k_optimo} componentes)")

# Desempeño por tipo de ataque
if cat_test is not None:
    print("\n--- TASA DE DETECCIÓN POR CATEGORÍA DE ATAQUE (CON PCA) ---")
    df_eval = pd.DataFrame({'Categoria': cat_test, 'Real': y_test, 'Pred': y_pred_pca})
    resumen_cat = df_eval.groupby('Categoria').apply(
        lambda g: (g['Real'] == g['Pred']).mean() * 100
    ).sort_values(ascending=False)
    for cat, tasa in resumen_cat.items():
        print(f"  {cat}: {tasa:.2f}% de acierto")

# 5. GENERACIÓN DE GRÁFICAS PARA LAS DIAPOSITIVAS
plt.rcParams.update({'font.size': 11})

# Gráfica 1: Varianza Explicada Acumulada
plt.figure(figsize=(8, 5))
componentes_x = np.arange(1, len(varianza_acum) + 1)
plt.bar(componentes_x, varianza_ind, alpha=0.6, color='#4A90E2', label='Varianza individual (%)')
plt.step(componentes_x, varianza_acum, where='mid', color='#1B365D', linewidth=2.2, label='Varianza acumulada (%)')
plt.axhline(y=90, color='#D9534F', linestyle='--', linewidth=1.5, label='Umbral objetivo (90%)')
plt.axvline(x=k_optimo, color='#2ECC71', linestyle=':', linewidth=1.8, label=f'Selección k = {k_optimo}')
plt.title('Varianza Explicada por Componente Principal en UNSW-NB15')
plt.xlabel('Número de Componentes Principales (k) [adimensional]')
plt.ylabel('Varianza Explicada [%]')
plt.legend(loc='center right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig1_varianza_pca.png', dpi=300)
plt.close()

# Gráfica 2: Proyección 2D (PC1 vs PC2)
plt.figure(figsize=(8, 5.5))
np.random.seed(42)
idx_muestra = np.random.choice(len(Z_test_pca), size=min(6000, len(Z_test_pca)), replace=False)
mask_normal = (y_test[idx_muestra] == 0)
mask_ataque = (y_test[idx_muestra] == 1)

plt.scatter(Z_test_pca[idx_muestra][mask_normal, 0], Z_test_pca[idx_muestra][mask_normal, 1],
            c='#2B6CB0', alpha=0.45, s=14, label='Tráfico Normal (Benigno)')
plt.scatter(Z_test_pca[idx_muestra][mask_ataque, 0], Z_test_pca[idx_muestra][mask_ataque, 1],
            c='#E53E3E', alpha=0.45, s=14, label='Tráfico de Ataque (Anómalo)')
plt.title('Proyección del Tráfico de Red en el Subespacio Reducido (PC1 vs. PC2)')
plt.xlabel(f'Componente Principal 1 ({varianza_ind[0]:.1f}% de varianza) [unidades Z]')
plt.ylabel(f'Componente Principal 2 ({varianza_ind[1]:.1f}% de varianza) [unidades Z]')
plt.xlim(np.percentile(Z_test_pca[:, 0], [1, 99]))
plt.ylim(np.percentile(Z_test_pca[:, 1], [1, 99]))
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig2_proyeccion_2d.png', dpi=300)
plt.close()

print("\n¡Proceso completado! Se guardaron fig1_varianza_pca.png y fig2_proyeccion_2d.png en tu carpeta.")