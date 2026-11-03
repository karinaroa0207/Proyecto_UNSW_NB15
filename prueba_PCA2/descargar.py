import os
import shutil
import kagglehub

print("Descargando UNSW-NB15, espera unos segundos...")
ruta = kagglehub.dataset_download("mrwellsdavid/unsw-nb15")

for archivo in ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"]:
    origen = os.path.join(ruta, archivo)
    if os.path.exists(origen):
        shutil.copy(origen, archivo)
        print(f"Archivo listo en tu carpeta: {archivo}")