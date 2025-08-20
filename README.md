# Data Augmentation Project

Este proyecto tiene como objetivo aplicar técnicas de aumento de datos (data augmentation) sobre conjuntos de datos tabulares, facilitando la generación de nuevos datos sintéticos para mejorar el rendimiento de modelos de machine learning.

## Estructura del Proyecto

- `data/` — Carpeta donde se almacenan los datos originales (por ejemplo, archivos CSV).
- `results/` — Resultados generados por el proceso de aumento de datos, incluyendo datos procesados y carpetas por usuario.
- `data_augmentation.py` — Script principal que realiza el aumento de datos.
- `config.json` — Archivo de configuración para definir parámetros del proceso.
- `documentation.txt` — Documentación adicional sobre el uso y funcionamiento del proyecto.

## Organización Recomendada de Carpetas

Para una mejor organización, se recomienda la siguiente estructura:

```
project-root/
│
├── data/                # Datos originales
│   └── raw/             # Datos sin procesar
│
├── results/             # Resultados y datos aumentados
│   └── users/           # Resultados por usuario
│
├── src/                 # Código fuente del proyecto
│   └── data_augmentation.py
│
├── config/              # Archivos de configuración
│   └── config.json
│
├── docs/                # Documentación
│   └── documentation.txt
│
└── README.md            # Este archivo
```

## Uso

1. Coloca tus archivos de datos originales en `data/raw/`.
2. Ajusta los parámetros en `config/config.json` según tus necesidades.
3. Ejecuta el script principal desde la carpeta `src/`:
   ```bash
   python src/data_augmentation.py
   ```
4. Los resultados se guardarán en la carpeta `results/`.

## Contribuciones

¡Las contribuciones son bienvenidas! Por favor, abre un issue o pull request para sugerencias o mejoras.

---

Este README proporciona una visión general y una guía para organizar y utilizar el proyecto de manera eficiente.
