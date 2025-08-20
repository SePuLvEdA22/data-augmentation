import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Funcion para cargar la configuracion del archivo JSON.
def load_config(config_path):
    try:
        with open(config_path, "r") as file:
            config = json.load(file)
        print("Configuracion cargada correctamente")
        return config
    except FileNotFoundError:
        print(f"No se encontro el archivo de configuracion {config_path}")
        exit(1)
    except json.JSONDecodeError:
        print("Error: El archivo de configuracion no es un archivo JSON valido")
        exit(1)

# Validar la existencia de las rutas configuradas y crear la carpeta de salida si no existe.
def validate_path(config):
    if not os.path.exists(config["folder_path"]):
        print(f"Error: La carpeta {config['folder_path']} no existe")
        exit(1)
    if not os.path.exists(config["output_folder"]):
        os.makedirs(config["output_folder"])
        print(f"La carpeta {config['output_folder']} ha sido creada")

# Función para limpiar la columna 'Normalizado'
def clean_normalized(value):
    if isinstance(value, str):
        parts = value.split('.')
        if len(parts) > 1:
            # Mantener solo el primer punto como separador decimal
            value = parts[0] + '.' + ''.join(parts[1:])
        try:
            return float(value)
        except ValueError:
            print(f"Error converting {value}")
            return np.nan
    return value

# Procesar los datos de entrada segun los codigos de usuario estacional o no estacional.
def process_data(config):
    input_file = os.path.join(config["folder_path"], config["file_path"])
    try:
        data = pd.read_csv(input_file, sep=",")
        print(f"Datos cargados correctamente desde {input_file}")
    except FileNotFoundError:
        print(f"Error: El archivo {input_file} no existe")
        exit(1)

    # Aplicar la limpieza a la columna 'Normalizado'
    data["Normalizado"] = data["Normalizado"].apply(clean_normalized)
    
    # Verificar si hay NaNs después de la conversión
    print(f"Total de NaNs en 'Normalizado' después de la conversión: {data['Normalizado'].isna().sum()}")
    
    # Llenar NaNs con interpolación lineal si es necesario
    data["Normalizado"] = data["Normalizado"].interpolate(method='linear')

    # Filtrar datos segun los codigos positivos y negativos definidos en la configuracion.
    positive_codes = config["positive_codes"]
    negative_codes = config["negative_codes"]
    data["es_estacional"] = np.where(data["Codigo usuario"].isin(positive_codes), 1,
                                     np.where(data["Codigo usuario"].isin(negative_codes), 0, None))
    
    # Identificar usuarios con 'Normalizado' siempre en cero
    def has_non_zero_normalized(group):
        return group['Normalizado'].ne(0).any()

    # Filtrar usuarios cuya 'Normalizado' no es cero en al menos un periodo
    filtered_data = data.groupby('Codigo usuario').filter(has_non_zero_normalized)

    # Eliminar filas con valores nulos en la columna "es_estacional" o "Normalizado".
    filtered_data = filtered_data.dropna(subset=["es_estacional", "Normalizado"])

    # Guardar los datos procesados en un archivo CSV.
    output_file = os.path.join(config["output_folder"], "processed_data.csv")
    filtered_data.to_csv(output_file, sep=";", index=False)
    print(f"Datos procesados guardados en {output_file}")
    return filtered_data


# Agregar ruido porcentual a los datos.
def add_percentage_noise(data, column, min_percent=0.01, max_percent=0.06):
    data = data.copy()
    n = len(data)
    noise_factors = np.random.uniform(min_percent, max_percent, n)
    noise = data[column] * noise_factors
    data[column] += noise
    return data

def multiply_normalized(data, factor):
    data = data.copy()  # Crear una copia para no modificar el original
    data['Normalizado'] = data['Normalizado'] * factor
    return data

# Funciones específicas para cada multiplicador
def multiply_by_0_5(data):
    return multiply_normalized(data, 0.5)

def multiply_by_1_5(data):
    return multiply_normalized(data, 1.5)

def multiply_by_2_5(data):
    return multiply_normalized(data, 2.5)

# Aplicar transformaciones a los datos y guardar resultados.
def apply_transformations_and_save(data, config):
    output_folder = config["output_folder"]
    augmentation_factor = config["augmentation_factor"]
    normalized_column = "Normalizado"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    augmented_data = []

    # Aplicar transformaciones existentes
    original_data = data  # Guardar la versión original para aplicar multiplicaciones

    for i in range(augmentation_factor):
        data_noisy = add_percentage_noise(data, normalized_column, min_percent=0.01, max_percent=0.06)
        augmented_data.append((f"modificacion_{i+1}", data_noisy))
        
        # Aplicar multiplicaciones a cada versión con ruido
        for factor, suffix in [(0.5, "0.5"), (1.5, "1.5"), (2.5, "2.5")]:
            data_multiplied = multiply_normalized(data_noisy, factor)
            augmented_data.append((f"modificacion_{i+1}_multiplicado_{suffix}", data_multiplied))

    # Aplicar multiplicaciones a la versión original
    for factor, suffix in [(0.5, "0.5"), (1.5, "1.5"), (2.5, "2.5")]:
        data_multiplied = multiply_normalized(original_data, factor)
        augmented_data.append((f"original_multiplicado_{suffix}", data_multiplied))

    augmented_file = os.path.join(output_folder, "augmented_data.csv")
    pd.concat([d[1] for d in augmented_data], ignore_index=True).to_csv(augmented_file, sep=';', index=False)
    print(f"Datos aumentados guardados en: {augmented_file}")

    # Guardar graficas originales y transformadas por usuario.
    save_user_graphs(original_data, output_folder, normalized_column, "original")
    for mod_name, mod_data in augmented_data:
        save_user_graphs(mod_data, output_folder, normalized_column, mod_name)

# Guardar graficas por usuario en carpetas separadas.
# Guardar graficas por usuario en carpetas separadas.
def save_user_graphs(data, output_folder, column, graph_label):
    users = data["Codigo usuario"].unique()

    for user in users:
        user_folder = os.path.join(output_folder, f"user_{user}")
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)

        user_data = data[data["Codigo usuario"] == user].sort_values(by="Periodo", ascending=True)
        periods = user_data["Periodo"].astype(str)
        values = user_data[column]  # Aquí obtenemos los valores de 'Normalizado'

        plt.figure(figsize=(15, 10))
        plt.plot(periods, values, label=f"{column} ({graph_label})", marker="o")

        # Añadir anotaciones para cada punto
        for i, (period, value) in enumerate(zip(periods, values)):
            plt.annotate(f'{value:.2f}',  # Formatear el valor a 2 decimales
                         (period, value),
                         textcoords="offset points",  # Ajustar la posición del texto
                         xytext=(0,10),  # Desplazar el texto un poco hacia arriba
                         ha='center')

        plt.xticks(rotation=90)
        plt.ylim(0, max(values) * 1.1)  # Ajustar el eje Y dinámicamente
        plt.title(f"{column} del usuario {user} ({graph_label})")
        plt.xlabel("Periodo")
        plt.ylabel(column)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(user_folder, f"user_{user}_{graph_label}.png"))
        plt.close()

# Funcion principal.
if __name__ == "__main__":
    # Usar la nueva ruta relativa para el archivo de configuración
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.json")
    config = load_config(CONFIG_PATH)
    validate_path(config)
    data = process_data(config)
    apply_transformations_and_save(data, config)