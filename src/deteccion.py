import torch
import cv2
import numpy as np
from mss import mss

# Cargar el modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True, trust_repo=True)

# Configurar la captura de pantalla
pantalla = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}  # Ajusta la resolución a la de tu pantalla
sct = mss()

# Crear una ventana llamada 'YOLOv5 Real-Time Screen Detection'
cv2.namedWindow("YOLOv5 Real-Time Screen Detection", cv2.WINDOW_NORMAL)

# Lista de clases de veehiculohículos
clase_Vehiculo = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']

while True:
    # Capturar la pantalla
    sct_img = sct.grab(pantalla)
    
    # Convertir la captura a un arreglo numpy y luego a formato BGR para OpenCV
    frame = np.array(sct_img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Realizar inferencia con YOLOv5
    results = model(frame)

    # Filtrar solo detecciones de vehículos
    filtered_boxes = []
    for *box, conf, cls in results.xyxy[0]:  # xyxy contiene las coordenadas, confianza y clase
        label = model.names[int(cls)]
        if label in clase_Vehiculo:  # Filtrar solo vehículos
            filtered_boxes.append((*box, conf, cls))

    # Dibujar las cajas de los vehículos en la imagen
    for *box, conf, cls in filtered_boxes:
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Dibujar la caja en verde
        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar el frame con las detecciones filtradas
    cv2.imshow("YOLOv5 Real-Time Screen Detection", frame)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cerrar la ventana
cv2.destroyAllWindows()
