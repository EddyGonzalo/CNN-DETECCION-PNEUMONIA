from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import json
from datetime import datetime

# --- Modelo mejorado (igual que antes) ---
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- Configuración Flask ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Cargar modelo ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedCNN()
model.load_state_dict(torch.load("best_cnn_model_mejorado.pt", map_location=device))
model.eval()
model.to(device)

# --- Transformación de la imagen ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

HISTORIAL_PATH = "historial_pacientes.json"

def guardar_en_historial(data):
    try:
        with open(HISTORIAL_PATH, "r", encoding="utf-8") as f:
            historial = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        historial = []
    historial.append(data)
    with open(HISTORIAL_PATH, "w", encoding="utf-8") as f:
        json.dump(historial, f, ensure_ascii=False, indent=2)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    filename = None
    nombre = ""
    edad = ""
    sexo = ""
    observaciones = ""
    if request.method == 'POST':
        if 'guardar' in request.form:
            # Guardar datos del paciente y predicción
            filename = request.form.get('filename')
            prediction = request.form.get('prediction')
            confidence = float(request.form.get('confidence'))
            nombre = request.form.get('nombre', '')
            edad = request.form.get('edad', '')
            sexo = request.form.get('sexo', '')
            observaciones = request.form.get('observaciones', '')
            data = {
                "nombre": nombre,
                "edad": edad,
                "sexo": sexo,
                "observaciones": observaciones,
                "filename": filename,
                "prediccion": prediction,
                "confianza": confidence,
                "fecha": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            guardar_en_historial(data)
            # Mensaje de éxito (puedes mostrar un flash o similar)
        else:
            # Primer POST: analizar imagen
            file = request.files['file']
            if file:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                image = Image.open(filepath).convert("RGB")
                input_tensor = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1)
                    predicted_index = torch.argmax(probs, dim=1).item()
                    classes = ["NORMAL", "PNEUMONIA"]
                    prediction = classes[predicted_index]
                    confidence = probs[0][predicted_index].item() * 100
    return render_template('index.html', prediction=prediction, confidence=confidence, filename=filename,
                           nombre=nombre, edad=edad, sexo=sexo, observaciones=observaciones, active_page='diagnostico')

@app.route('/pacientes')
def pacientes():
    try:
        with open(HISTORIAL_PATH, "r", encoding="utf-8") as f:
            historial = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        historial = []
    return render_template('pacientes.html', historial=historial, active_page='pacientes')

@app.route('/estadisticas')
def estadisticas():
    try:
        with open(HISTORIAL_PATH, "r", encoding="utf-8") as f:
            historial = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        historial = []
    total = len(historial)
    normales = sum(1 for p in historial if p["prediccion"] == "NORMAL")
    neumonia = sum(1 for p in historial if p["prediccion"] == "PNEUMONIA")
    edades = [int(p["edad"]) for p in historial if p["edad"].isdigit()]
    promedio_edad = sum(edades) / len(edades) if edades else 0
    return render_template('estadisticas.html', total=total, normales=normales, neumonia=neumonia,
                           promedio_edad=promedio_edad, historial=historial, active_page='estadisticas')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
