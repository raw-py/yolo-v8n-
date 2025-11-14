from tkinter import *
from PIL import Image, ImageTk
import cv2 
from ultralytics import YOLO
import numpy as np
import time
import sys
from mss import mss
import threading
import os
import platform
import pyautogui
from pynput import mouse

# --- Configuración global y carga del modelo ---
print("Cargando modelo YOLO con detección normal...")

# Intentar cargar modelos normales en orden de tamaño
model_options = [
    "yolov8n.pt",      # Modelo más pequeño y estable (ya lo tienes)
    "yolov8s.pt",      # Modelo pequeño
    "yolov8m.pt",      # Modelo mediano
    "yolov8l.pt",      # Modelo grande
    "yolov8x.pt"       # Modelo extra grande
]

model = None
model_type = None

# Primero verificar si los archivos existen localmente
available_models = []
for model_path in model_options:
    if os.path.exists(model_path):
        available_models.append(model_path)
        print(f" Modelo encontrado localmente: {model_path}")

if not available_models:
    print(" No se encontraron modelos localmente.")
    print(" Intentando descargar automáticamente...")
    print("   (Si falla, descarga manualmente desde: https://github.com/ultralytics/assets/releases)")
    available_models = model_options  # Intentar todos

# Intentar cargar los modelos disponibles
for model_path in available_models:
    try:
        print(f"Intentando cargar {model_path}...")
        # Si el archivo no existe, YOLO intentará descargarlo automáticamente
        model = YOLO(model_path)
        model_type = model_path.replace('.pt', '')
        print(f" Modelo {model_type} cargado exitosamente!")
        break
    except Exception as e:
        error_msg = str(e)
        if "Download" in error_msg or "download" in error_msg or "curl" in error_msg.lower():
            print(f" Error descargando {model_path}: Problema de conexión o permisos")
            print(f"   Descarga manualmente desde: https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_path}")
        else:
            print(f" Error cargando {model_path}: {e}")
        continue

if model is None:
    print("\n" + "="*60)
    print(" ERROR FATAL: No se pudo cargar ningún modelo")
    print("="*60)
    print("\n SOLUCIÓN: Descarga manualmente un modelo de YOLO 8")
    print("\n Descarga uno de estos modelos desde:")
    print("   https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt")
    print("   https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt")
    print("   https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt")
    print("\n Colócalo en el mismo directorio que este script:")
    print(f"   {os.getcwd()}")
    print("\n Luego ejecuta el script principal nuevamente.")
    print("="*60)
    sys.exit(1)

# Carga los nombres de las clases (personas, coches, etc.) del dataset COCO
classesFile = "coco.names"
try:
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{classesFile}'. Asegúrate de que esté en el mismo directorio.")
    print("Puedes descargarlo desde: https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names")
    sys.exit("Archivo coco.names faltante.")

# Genera colores aleatorios para dibujar los recuadros de las clases
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Configuración de detección normal (sin esqueletos)
# El modelo normal detecta objetos pero no keypoints de pose

# --- Configuración de la captura de pantalla ---
sct = mss()

# Mostrar información de monitores disponibles
print("Monitores disponibles:")
for i, monitor in enumerate(sct.monitors):
    print(f"Monitor {i}: {monitor['width']}x{monitor['height']} en ({monitor['left']}, {monitor['top']})")

# Configurar captura del monitor 2 (pantalla que se revisa por la IA)
if len(sct.monitors) > 2:
    monitor_area = sct.monitors[2]  # Monitor 2 - pantalla que se revisa
    print(f"📹 Capturando desde: Monitor 2 - {monitor_area['width']}x{monitor_area['height']} en ({monitor_area['left']}, {monitor_area['top']})")
else:
    # Si no hay monitor 2, usar monitor 1 como respaldo
    monitor_area = sct.monitors[1]
    print(f" Solo hay {len(sct.monitors)-1} monitor(es). Usando Monitor 1 como respaldo.")
    print(f" Capturando desde: Monitor 1 - {monitor_area['width']}x{monitor_area['height']}")

# Guardar referencia al monitor de captura para calcular coordenadas del mouse
capture_monitor = monitor_area
print(f"El mouse se moverá en las coordenadas del monitor de captura: ({capture_monitor['left']}, {capture_monitor['top']})")

# --- Variables globales para Tkinter ---
root = None
label = None
running = True

# --- Variables globales para detección de mouse ---
left_button_pressed = False
right_button_pressed = False
detected_boxes = []  # Almacenar los cuadrados detectados más recientes
mouse_listener = None
last_mouse_move_time = 0  # Para evitar múltiples movimientos rápidos
MOUSE_MOVE_COOLDOWN = 0.5  # Segundos entre movimientos de mouse

# --- Configuración de precisión ---
# Ajusta estos valores para mejorar la precisión:
CONFIDENCE_THRESHOLD = 0.25  # Umbral de confianza más bajo para detectar más (0.1-1.0)
IOU_THRESHOLD = 0.2          # Umbral de IoU más bajo para detectar más (0.1-0.9)
CLASSES_TO_DETECT = [0]      # 0 = persona, puedes agregar más clases si quieres
MAX_DETECTIONS = 50          # Máximo número de detecciones (aumentado de 20 a 50)

print(f"Configuración de precisión:")
print(f"- Confianza mínima: {CONFIDENCE_THRESHOLD}")
print(f"- IoU threshold: {IOU_THRESHOLD}")
print(f"- Máximo detecciones: {MAX_DETECTIONS}")
print(f"- Clases detectadas: {CLASSES_TO_DETECT}")
print(f"- Modelo usado: {model_type}")

# --- Funciones para detección de mouse ---
def on_mouse_click(x, y, button, pressed):
    """Callback para detectar clicks del mouse"""
    global left_button_pressed, right_button_pressed, last_mouse_move_time
    
    if button == mouse.Button.left:
        left_button_pressed = pressed
    elif button == mouse.Button.right:
        right_button_pressed = pressed
    
    # Si ambos botones están presionados simultáneamente, mover el mouse al cuadrado más cercano
    current_time = time.time()
    if (left_button_pressed and right_button_pressed and 
        len(detected_boxes) > 0 and 
        (current_time - last_mouse_move_time) > MOUSE_MOVE_COOLDOWN):
        move_mouse_to_nearest_box(x, y)
        last_mouse_move_time = current_time
    
    return True  # Continuar escuchando

def move_mouse_to_nearest_box(current_x, current_y):
    """Mueve el mouse al centro del cuadrado detectado más cercano en la pantalla de captura (Monitor 2)"""
    global detected_boxes, capture_monitor
    
    if not detected_boxes:
        return
    
    # Obtener posición actual del mouse
    try:
        current_mouse_x, current_mouse_y = pyautogui.position()
    except:
        current_mouse_x, current_mouse_y = current_x, current_y
    
    # Encontrar el cuadrado más cercano al cursor actual
    min_distance = float('inf')
    nearest_box_center = None
    
    for box in detected_boxes:
        # Calcular centro del bounding box en coordenadas relativas al frame capturado
        center_x_relative = (box['x1'] + box['x2']) // 2
        center_y_relative = (box['y1'] + box['y2']) // 2
        
        # Convertir a coordenadas absolutas de la pantalla de captura (Monitor 2)
        # Las coordenadas relativas del frame se corresponden directamente con las coordenadas del monitor
        screen_x = capture_monitor['left'] + center_x_relative
        screen_y = capture_monitor['top'] + center_y_relative
        
        # Calcular distancia desde la posición actual del mouse (no desde el click)
        distance = ((current_mouse_x - screen_x) ** 2 + (current_mouse_y - screen_y) ** 2) ** 0.5
        
        if distance < min_distance:
            min_distance = distance
            nearest_box_center = (screen_x, screen_y)
    
    if nearest_box_center:
        try:
            # Mover el mouse a las coordenadas absolutas en la pantalla de captura
            pyautogui.moveTo(nearest_box_center[0], nearest_box_center[1], duration=0.1)
            print(f" Mouse movido a: ({nearest_box_center[0]}, {nearest_box_center[1]}) en Monitor de Captura")
            print(f"   Coordenadas relativas en frame: ({(nearest_box_center[0] - capture_monitor['left'])}, {(nearest_box_center[1] - capture_monitor['top'])})")
        except Exception as e:
            print(f" Error moviendo el mouse: {e}")

def start_mouse_listener():
    """Inicia el listener de mouse en un hilo separado"""
    global mouse_listener
    try:
        mouse_listener = mouse.Listener(on_click=on_mouse_click)
        mouse_listener.start()
        print(" Listener de mouse iniciado. Presiona click izquierdo + derecho para mover el mouse al cuadrado detectado.")
    except Exception as e:
        print(f" Error iniciando listener de mouse: {e}")
        print(" La funcionalidad de movimiento de mouse no estará disponible.")

# --- Funciones ---
def onClossing():
    """Función para limpiar y cerrar la aplicación al cerrar la ventana."""
    global running, mouse_listener
    print("Cerrando aplicación...")
    running = False
    
    # Detener el listener de mouse
    if mouse_listener:
        mouse_listener.stop()
    
    # Detener el bucle principal de Tkinter y destruir la ventana
    if root:
        root.quit()
        root.destroy()
    
    # Forzar salida del programa
    os._exit(0)

def process_frame():
    """
    Función que procesa frames en un hilo separado para mejor rendimiento
    """
    global running, label, detected_boxes
    
    # Crear una nueva instancia de mss en este hilo
    thread_sct = mss()
    
    while running:
        try:
            last_time = time.time()  # Para calcular FPS

            # 1. Capturar la pantalla del monitor principal usando la instancia del hilo
            sct_img = thread_sct.grab(monitor_area)
            frame = np.array(sct_img)
            # Convertir de RGBA (mss) a BGR (OpenCV)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            # Opcional: Redimensionar para mejor rendimiento/precisión
            # frame = cv2.resize(frame, (640, 480))  # Descomenta para mejor rendimiento
            # frame = cv2.resize(frame, (1280, 720)) # Descomenta para mejor precisión
            
            frame_for_yolo = frame.copy()  # Usar el frame original para YOLO

            # 2. Realizar la detección con YOLO con parámetros optimizados y manejo de errores
            try:
                results = model.predict(
                    frame_for_yolo,
                    verbose=False,
                    conf=CONFIDENCE_THRESHOLD,
                    classes=CLASSES_TO_DETECT,
                    iou=IOU_THRESHOLD,
                    agnostic_nms=True,  # Mejor para detección de personas
                    max_det=MAX_DETECTIONS  # Máximo número de detecciones
                )
            except Exception as predict_error:
                print(f"Error en predicción: {predict_error}")
                # Continuar con el siguiente frame
                time.sleep(0.01)
                continue
            
            # Crear un frame negro del mismo tamaño
            height, width = frame.shape[:2]
            annotated_frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Contador de objetos detectados
            detection_count = 0
            person_count = 0  # Contador específico para personas
            # Limpiar lista de cuadrados detectados para este frame
            detected_boxes.clear()
            
            # Procesar resultados
            for result in results:
                boxes = result.boxes
                
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        try:
                            # Obtener coordenadas del bounding box
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            
                            # Verificar si la clase está en la lista de clases a detectar
                            if class_id in CLASSES_TO_DETECT:
                                detection_count += 1
                                
                                # Contar personas específicamente
                                if class_id == 0:  # 0 = persona en COCO
                                    person_count += 1
                                
                                # Convertir coordenadas a enteros
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                
                                # Guardar el cuadrado detectado para movimiento de mouse
                                detected_boxes.append({
                                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                    'confidence': confidence
                                })
                                
                                # Convertir color a tupla para evitar problemas
                                color_tuple = tuple(map(int, COLORS[class_id]))
                                
                                # Dibujar el bounding box en el frame negro
                                cv2.rectangle(annotated_frame,
                                            (x1, y1),
                                            (x2, y2),
                                            color_tuple, 3)
                                
                                # Agregar texto con fondo negro para mejor visibilidad
                                class_name = classes[class_id] if class_id < len(classes) else f"Clase {class_id}"
                                label_text = f"{class_name}: {confidence:.2f}"
                                
                                # Obtener el tamaño del texto
                                (text_width, text_height), baseline = cv2.getTextSize(
                                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                                )
                                
                                # Dibujar rectángulo de fondo para el texto
                                cv2.rectangle(annotated_frame,
                                            (x1, y1 - text_height - 10),
                                            (x1 + text_width, y1),
                                            (0, 0, 0), -1)
                                
                                # Agregar texto
                                cv2.putText(annotated_frame, label_text,
                                          (x1, y1 - 5),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_tuple, 2)
                        except Exception as box_error:
                            print(f"Error procesando bounding box: {box_error}")
                            continue
            
            # Calcular FPS
            fps = 1 / (time.time() - last_time)
            
            # Agregar información en la esquina superior izquierda
            info_text = f"FPS: {int(fps)} | Personas: {person_count} | {model_type.upper()}"
            cv2.putText(annotated_frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # 3. Preparar el frame para Tkinter
            img_tk = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img_tk = Image.fromarray(img_tk)
            tkimage = ImageTk.PhotoImage(img_tk)
            
            # Actualizar el Label de Tkinter de forma segura
            if root and running:
                root.after(0, lambda img=tkimage: update_label(img))
            
            # Pequeña pausa para no saturar el CPU
            time.sleep(0.01)

        except Exception as e:
            print(f"Error en el bucle principal: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)  # Pausa más larga en caso de error
            continue

def update_label(tkimage):
    """Función para actualizar el label de forma segura"""
    global label
    if label and running:
        label.configure(image=tkimage)
        label.image = tkimage

# --- Inicialización de la Interfaz Gráfica (Tkinter) ---
root = Tk()
root.protocol("WM_DELETE_WINDOW", onClossing)
root.title(f"Detección de Personas en Pantalla ({model_type.upper()})")

# Configurar la ventana para que aparezca en un monitor diferente al de captura
# La visualización debe estar en otro monitor, no en el monitor de captura (Monitor 2)
display_monitor = None

if len(sct.monitors) > 2:
    # Verificar si estamos capturando del Monitor 2 comparando coordenadas
    is_capturing_monitor_2 = (capture_monitor['left'] == sct.monitors[2]['left'] and 
                               capture_monitor['top'] == sct.monitors[2]['top'] and
                               capture_monitor['width'] == sct.monitors[2]['width'] and
                               capture_monitor['height'] == sct.monitors[2]['height'])
    
    if is_capturing_monitor_2:
        # Si capturamos del Monitor 2, mostrar en Monitor 1
        display_monitor = sct.monitors[1]
        print(f"📺 Visualización en Monitor 1 (diferente al Monitor 2 de captura)")
    else:
        # Si capturamos del Monitor 1, intentar usar Monitor 2 si existe
        if len(sct.monitors) > 2:
            display_monitor = sct.monitors[2]
            print(f"📺 Visualización en Monitor 2 (diferente al Monitor 1 de captura)")
        else:
            display_monitor = sct.monitors[1]
            print(f"📺 Visualización en Monitor 1")
    
    if display_monitor:
        root.geometry(f"+{display_monitor['left']}+{display_monitor['top']}")
        print(f"   Posición: ({display_monitor['left']}, {display_monitor['top']})")
        print(f"   Tamaño: {display_monitor['width']}x{display_monitor['height']}")
    else:
        # Si no hay otro monitor, posicionar al lado del monitor de captura
        root.geometry(f"+{capture_monitor['left'] + capture_monitor['width']}+{capture_monitor['top']}")
        print(f" Visualización al lado del monitor de captura")
else:
    # Si solo hay un monitor, posicionar al lado
    display_monitor = sct.monitors[1] if len(sct.monitors) > 1 else None
    if display_monitor:
        root.geometry(f"+{display_monitor['left']}+{display_monitor['top']}")
    else:
        root.geometry(f"+{capture_monitor['left'] + capture_monitor['width']}+{capture_monitor['top']}")
    print(f" Solo hay un monitor. Visualización al lado del monitor de captura")

# Hacer la ventana de pantalla completa (compatible con Windows y Linux)
system_os = platform.system()
if system_os == 'Windows':
    root.state('zoomed')  # En Windows
elif system_os == 'Linux':
    root.attributes('-zoomed', True)  # En Linux
else:
    # Para macOS u otros sistemas
    root.attributes('-zoomed', True)

# Configurar para que esté siempre al frente
root.attributes('-topmost', True)

label = Label(root)
label.grid(row=0, column=0, padx=10, pady=10)

# --- Inicio del procesamiento ---
try:
    print("\n" + "="*60)
    print("🚀 Iniciando detección de personas")
    print("="*60)
    print(f"📹 Monitor de CAPTURA (revisado por IA):")
    print(f"   - Monitor 2: {capture_monitor['width']}x{capture_monitor['height']}")
    print(f"   - Posición: ({capture_monitor['left']}, {capture_monitor['top']})")
    print(f"\n📺 Monitor de VISUALIZACIÓN (donde se muestran los cuadros):")
    if display_monitor:
        # Determinar qué número de monitor es
        monitor_number = None
        for i, mon in enumerate(sct.monitors):
            if (mon['left'] == display_monitor['left'] and 
                mon['top'] == display_monitor['top'] and
                mon['width'] == display_monitor['width'] and
                mon['height'] == display_monitor['height']):
                monitor_number = i
                break
        
        if monitor_number is not None:
            print(f"   - Monitor {monitor_number}: {display_monitor['width']}x{display_monitor['height']}")
            print(f"   - Posición: ({display_monitor['left']}, {display_monitor['top']})")
        else:
            print(f"   - Tamaño: {display_monitor['width']}x{display_monitor['height']}")
            print(f"   - Posición: ({display_monitor['left']}, {display_monitor['top']})")
    else:
        print(f"   - Mismo monitor de captura")
    print(f"\n🖱️ Movimiento del mouse:")
    print(f"   - El mouse se moverá en el Monitor de CAPTURA (Monitor 2)")
    print(f"   - Las coordenadas son proporcionales al monitor de captura")
    print(f"   - Los cuadros se muestran en otro monitor (visualización)")
    print(f"\n💻 Sistema operativo: {system_os}")
    print("="*60)
    print("\n Instrucciones:")
    print("   1. Los cuadros de detección se muestran en el monitor de visualización")
    print("   2. Presiona click IZQUIERDO + DERECHO simultáneamente")
    print("   3. El mouse se moverá al centro de la persona detectada más cercana")
    print("   4. El movimiento ocurre en el Monitor 2 (monitor de captura)")
    print("   5. Presiona Ctrl+C en la consola para cerrar el programa")
    print("="*60 + "\n")

    # Iniciar el listener de mouse
    start_mouse_listener()

    # Iniciar el procesamiento en un hilo separado
    processing_thread = threading.Thread(target=process_frame, daemon=True)
    processing_thread.start()

    # Iniciar el bucle principal de Tkinter
    root.mainloop()

except Exception as e:
    print(f"Error al iniciar Tkinter: {e}")

    onClossing()
