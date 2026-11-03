```markdown
# Detección Eficiente de Intrusiones en Redes Informáticas Mediante Reducción de Dimensionalidad (PCA) en el Dataset UNSW-NB15


---

##  Integrantes del Grupo
* **Juan David Campos Salcedo** (Código: 20231020007)
* **Cristian David Parroquiano Jiménez** (Código: 20222020192)
* **Ana Karina Roa Mora** (Código: 20232020118)
* **Director / Docente:** Duván Andrés Téllez Castro
* **Institución:** Universidad Distrital Francisco José de Caldas – Facultad de Ingeniería (Ingeniería de Sistemas).

---

## Descripción del Proyecto
Los sistemas modernos de detección de intrusiones (IDS) enfrentan un desafío crítico: el procesamiento de flujos de red de alta dimensionalidad (más de 40 atributos técnicos), lo que genera redundancia de datos, sobrecarga computacional y un aumento en la **Tasa de Falsas Alarmas (FAR)**. 

Este proyecto evalúa empíricamente la efectividad de **PCA** como técnica de preprocesamiento para comprimir el espacio de características del conjunto de datos **UNSW-NB15** ($257.673$ registros), buscando retener al menos el **$90\%$ de la varianza estadística** sin degradar significativamente la capacidad de detección de ataques

---

##  Estructura del Repositorio
```text
├── venv/                           # Entorno virtual de Python (excluido en gitignore)
├── UNSW_NB15_training-set.csv      # Conjunto de datos de entrenamiento
├── UNSW_NB15_testing-set.csv       # Conjunto de datos de prueba
├── descargar.py                    # Script automatizado para la obtención del dataset
├── prueba_concepto.py              # Script principal (Z-Score, PCA, Clasificador y Métricas)[cite: 10]
├── fig1_varianza_pca.png           # Gráfica 1: Varianza explicada acumulada
├── fig2_proyeccion_2d.png          # Gráfica 2: Proyección espacial en 2D (PC1 vs PC2)
├── .gitignore                      # Archivos excluidos del control de versiones
└── README.md                       # Documentación del proyecto

```

---

##  Instrucciones de Reproducibilidad

Para reproducir los experimentos y generar las métricas y gráficas de la prueba de concepto, sigue estos pasos en tu terminal:

1. **Clonar el repositorio:**
```bash
git clone [https://github.com/karinaroa0207/TU-NOMBRE-DE-REPOSITORIO.git](https://github.com/karinaroa0207/TU-NOMBRE-DE-REPOSITORIO.git)
cd TU-NOMBRE-DE-REPOSITORIO

```


2. **Crear y activar el entorno virtual:**
```bash
python -m venv venv
# En Windows (CMD/PowerShell):
venv\Scripts\activate

```


3. **Instalar las dependencias del proyecto:**
```bash
pip install --default-timeout=100 numpy pandas scikit-learn matplotlib kagglehub

```


4. **Descargar los datos y ejecutar el modelo:**
```bash
python descargar.py
python prueba_concepto.py

```



---

## Resultados Principales Obtenidos en la Prueba de Concepto

* **Reducción de Dimensionalidad:** Se comprimió el espacio de **42 a 17 componentes ($k = 17$)**, reteniendo con exactitud el **$90.12\%$ de la varianza total** del tráfico de red.


* **Optimización de Tiempos:** El tiempo de inferencia del clasificador (Árbol de Decisión) se redujo a **$0.012$ segundos**, aliviando la carga operativa del sistema.


* **Detección por Categorías:** Ataques complejos como *Generic* ($99.90\%$ de acierto), *Worms* ($98.31\%$) y *DoS* ($98.02\%$) demostraron una alta efectividad en el subespacio reducido.
