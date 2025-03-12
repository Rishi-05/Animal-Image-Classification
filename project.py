import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("C:\\Users\\RAMESH\\Desktop\\codes\\Python Projects\\photo detector\\model.h5") 
categories = ["cats", "dogs", "horses"]
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    image = cv2.resize(frame, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    prediction = model.predict(image)[0]
    class_idx = np.argmax(prediction)
    class_label = categories[class_idx]
    if class_idx < len(categories):
        class_label = categories[class_idx]
    else:
        class_label = "Unknown"
    confidence = prediction[class_idx] * 100

    cv2.putText(frame, f"{class_label} {confidence:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Live Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
