from ultralytics import YOLO
import cv2

# Загружаем модель YOLOv8n
model = YOLO("yolov8n.pt")

# Загружаем изображение
image = cv2.imread("test_img.jpg")

# Выполняем детекцию
results = model(image)
# Получаем число обнаруженных людей
num_people = len(results[0].boxes.cls)

# Перебираем все объекты Detection в списке
for detection in results:
    # Получаем координаты прямоугольника
    x1, y1, x2, y2 = detection.boxes.xyxy[0].numpy().astype(int)  
    
    # Рисуем прямоугольник на изображении
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) 

# Выводим число людей на изображение
cv2.putText(image, f"People: {num_people}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Отображаем изображение
cv2.imshow("Результат", image)
cv2.waitKey(0)
