import os
import cv2
import numpy as np

#opencv에서 제공하는 미리 학습된 파일
cascade_path = 'C:/workspace/cascade/haarcascade_frontalface_default.xml'

#얼굴 추출해서 저장하기
def faceScrap(uname):
    print("Saving Face Image --Start--")
    face_casecade = cv2.CascadeClassifier(cascade_path)

    def face_extractor(img):
        img
        faces = face_casecade.detectMultiScale(img,1.3,5)

        if faces is():
            return None
        for(x,y,w,h) in faces:
            cropped_face = img[y:y+h, x:x+w]
        return cropped_face

    def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error : Creating directory.' + directory)

    #추출된 얼굴 이미지 저장 장소
    savePath = 'C:/workspace/FaceScrap/'
    createFolder(os.path.join(savePath,str(uname)))
    print("Saving Face Image --Created Folder--")

    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    count = 0

    while True:
        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count+=1
            face = cv2.resize(face_extractor(frame),(200,200))

            file_name_pass = 'C:/workspace/FaceScrap/'+str(uname)+'/'+str(count)+'.jpg'
            print("Saving Face Image --%s--" %count)
            cv2.imwrite(file_name_pass, face)
        else:
            print("Saving Face Image --Error can not found face--")
            pass

        if cv2.waitKey(1) == 13 or count == 20:
            print("Saving Face Image --End--")
            break

#학습 파일 생성
def faceTrain():
    face_casecade = cv2.CascadeClassifier(cascade_path)
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    Face_ID = -1
    pev_person_name = ""
    y_ID = []
    x_train = []

    Face_Image = 'C:/workspace/FaceScrap'
    print(Face_Image)

    for root, dirs, files in os.walk(Face_Image):
        for file in files:
            if file.endswith("jpeg") or file.endswith("jpg") or file.endswith("png"):
                path = os.path.join(root, file)
                person_name = os.path.basename(root)

                if pev_person_name != person_name:
                    Face_ID = Face_ID + 1
                    pev_person_name = person_name

                img = cv2.imread(path)
                gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_casecade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

                print(Face_ID, faces)

                for (x, y, w, h) in faces:
                    roi = gray_image[y:y + h, x:x + w]
                    x_train.append(roi)
                    y_ID.append(Face_ID)

                    recognizer.train(x_train, np.array(y_ID))
                    recognizer.save("C:/workspace/Trained/face_trained.yml")
    print("Train End")
    print("-----------------------")

def faceRecognize():
    face_cascade = cv2.CascadeClassifier(cascade_path)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("C:/workspace/Trained/face_trained.yml")  # 저장된 값 가져오기

    listPath = "C:\workspace\FaceScrap"
    labels = os.listdir(listPath)

    cap = cv2.VideoCapture(0)  # 카메라 실행

    if cap.isOpened() == False:  # 카메라 생성 확인
        exit()

    while(True):
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)  # 얼굴 인식

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]  # 얼굴 부분만 가져오기

            id_, conf = recognizer.predict(roi_gray)  # 얼마나 유사한지 확인

            if conf >= 80:
                font = cv2.FONT_HERSHEY_SIMPLEX  # 폰트 지정
                cv2.putText(img, labels[id_], (x, y - 10), font, 1, (255, 255, 255), 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
                print("Recognize Video --%s has been detected--"%labels[id_])

        cv2.imshow('Preview', img)  # 이미지 보여주기
        if cv2.waitKey(10) >= 0:  # 키 입력 대기, 10ms
            break

    # 전체 종료
    cap.release()
    cv2.destroyAllWindows()


while(True):
    print("--------------------------------------------")
    print("--------------Face ID System----------------")
    print("--------------------------------------------")
    print("-----------1.인식 | 2.학습 | 3.종료------------")
    menu = input("메뉴를 선택해 주세요 : ")

    if menu=='1':
        faceRecognize()
    elif menu=='2':
        uname = input("이름을 입력해 주세요 : ")
        print("화면에 얼굴을 비춰주세요")
        faceScrap(uname)
        faceTrain()
    elif menu=='3':
        break