import cv2
import pandas as pd

count = 0

def scan_now(n):
    face_cascade = cv2.CascadeClassifier(r'D:\DL projects\project_face_detection\haarcascade_frontalface_alt.xml')
    video = cv2.VideoCapture(0)

    while True:
        success, frame = video.read()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4)

        array1 = [len(faces)]
        df = pd.DataFrame(array1, columns=['values'])

        m = str(n)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=3)
            cv2.putText(frame, text='current attendance=', org=(50, 400), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=1, color=(255, 0, 0), thickness=4)
            cv2.putText(frame, text=m, org=(400, 400), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1,
                        color=(255, 0, 0), thickness=4)
            cv2.putText(frame, text='THANK YOU, press esc (top left)', org=(50, 450),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255, 0, 0), thickness=4)

        cv2.imshow('press esc', frame)
        if cv2.waitKey(8000) & 0xFF == 27:
                break

    video.release()
    cv2.destroyAllWindows()
    x = df['values'].mode()[0]
    return x

x = scan_now(count)
count = count + x
print('current count:', count)
   