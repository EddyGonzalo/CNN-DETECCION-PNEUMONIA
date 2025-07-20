# Usa una imagen oficial de Python
FROM python:3.10-slim

# Instala dependencias del sistema (si usas torch, pillow, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia los archivos del proyecto
WORKDIR /app
COPY . /app

# Instala las dependencias de Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expón el puerto 8080 (Cloud Run lo requiere)
EXPOSE 8080

# Comando para correr la app
CMD exec gunicorn --bind :8080 --workers 1 --threads 8 app:app
