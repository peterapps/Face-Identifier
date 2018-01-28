import numpy as np
import cv2, os, datetime
from Tkinter import *
from PIL import Image
from PIL import ImageTk
from platform import platform

# Create the haar cascade
cascPath = "./cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
recognizer = cv2.face.createLBPHFaceRecognizer()

# Create output window
root = Tk()
root.title("Webcam Capture")
imageLabel = False
learnButton = False

# Video capture
cap = cv2.VideoCapture(0)

def scale(img, s):
    width, height = img.size
    return img.resize( ( int(round(width * s)), int(round(height * s)) ) )

def get_images_and_labels():
    image_paths = []
    image_names = []
    path = "./faces"
    # Loop through faces directory
    for dir in os.listdir(path):
        # Loop through each person's folder
        if not dir.startswith('.'):
            for i in os.listdir(path + "/" + dir):
                # Loop through each image of the person
                if i.endswith(".jpg"):
                    image_paths.append(path + "/" + dir + "/" + i)
                    image_names.append(dir)
    images = []
    labels = []
    for index, image_path in enumerate(image_paths):
        img = cv2.imread(image_path, 0)
        img = cv2.flip(img, 1)
        faces = faceCascade.detectMultiScale(
            img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        for (x, y, w, h) in faces:
            images.append(img[y: y + h, x: x + w])
            labels.append(index)
    return images, labels, image_names

images, labels, image_names = get_images_and_labels()
# Perform the training
recognizer.train(images, np.array(labels))

currentFace = None
def captureLoop():
    global imageLabel, learnButton, currentFace, images, labels, image_names
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    
    # Convert back to RGB
    #rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def save_current_frame():
        cv2.imwrite("./faces/" + currentFace + "/" + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + ".jpg", frame)
        images, labels, image_names = get_images_and_labels()
    
    # Draw a rectangle around Peter's face
    for (x, y, w, h) in faces:
        nbr_predicted, conf = recognizer.predict(gray[y: y + h, x: x + w])
        accuracy = int(round((250 - conf) / 250 * 100))
        minimum = 85
        color = (255, 0, 0)
        currentFace = image_names[nbr_predicted]
        name = str(accuracy) + "% " + image_names[nbr_predicted]
        if accuracy > minimum:
            color = (0, 255, 0)
        cv2.rectangle(rgb, (x, y), (x+w, y+h), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(rgb, name, (x, y - 5), font, 0.9, color ,2, cv2.LINE_AA)
    
    # Display the resulting frame
    img = Image.fromarray(rgb)
    img = scale(img, 0.9)
    img = ImageTk.PhotoImage(img)
    if not imageLabel:
        imageLabel = Label(root, image = img)
        imageLabel.image = img
        imageLabel.grid(row=0, column=0, columnspan=2)
    else:
        imageLabel.configure(image=img)
        imageLabel.image = img
    if not learnButton:
        learnButton = Button(root, text="Add Frame to Database", command=save_current_frame)
        learnButton.grid(row=1,column=0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
    	cap.release()
        return 1
    root.update()
    root.after(10, captureLoop)

root.after(10, captureLoop)
root.mainloop()