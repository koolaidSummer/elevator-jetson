import cv2
import paramiko
import pymysql
import numpy as np

dbUrl = "url"
dbPort = 3306
dbId = "Id"
dbPwd = "Pwd"


def sftp():
    sftpURL = 'url'
    sftpUser = 'user'
    sftpPass = 'pwd'

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(sftpURL, username=sftpUser, password=sftpPass)
    ftp = ssh.open_sftp()

    ymlFilepath = 'face_trainner.yml'
    ymlLocalpath = '/home/jetson/Desktop/workspace/sftp/face_trainner.yml'
    npyFilepath = 'floorList.npy'
    npyLocalpath = '/home/jetson/Desktop/workspace/sftp/floorList.npy'
    ftp.get(ymlFilepath, ymlLocalpath)
    ftp.get(npyFilepath, npyLocalpath)

    print("Upgrade complite")


def triggerOff():
    con = pymysql.connect(host=dbUrl, port=dbPort, user=dbId, password=dbPwd, db="elevator")
    cursor = con.cursor()
    sql = "UPDATE TRIG SET FLAG='0' WHERE FLAG='1'"
    cursor.execute(sql)

    cursor.close()
    con.commit()
    con.close()


def triggerCHK():
    con = pymysql.connect(host=dbUrl, port=dbPort, user=dbId, password=dbPwd, db="elevator")
    cursor = con.cursor()
    sql = "SELECT TRIM(FLAG) FROM TRIG"
    cursor.execute(sql)

    for list in cursor:
        chk = list[0]

    cursor.close()
    con.close()

    return chk


def faceRecog():
    count = 0
    face_cascade = cv2.CascadeClassifier(
        '/home/jetson/Desktop/workspace/cascade/haarcascade_frontalface_default.xml') #haar cascade filter 가져오기
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("/home/jetson/Desktop/workspace/sftp/face_trainner.yml")  # 학습된 파일 가져오기

    labels = getFloorList()
    cap = cv2.VideoCapture(0)  # 카메라 실행

    if cap.isOpened() == False:  # 카메라 생성 확인
        exit()

    while (triggerCHK() == '0'):
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)  # 얼굴 인식

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]  # 얼굴 부분만 가져오기

            id_, conf = recognizer.predict(roi_gray)  # 얼마나 유사한지 확인

            if conf >= 50:
                font = cv2.FONT_HERSHEY_SIMPLEX  # 폰트 지정
                cv2.putText(img, labels[id_], (x, y - 10), font, 1, (255, 255, 255), 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
                putFloor(labels[id_])
                print(labels[id_])

        cv2.imshow('Preview', img)  # 이미지 보여주기
        if cv2.waitKey(10) >= 0:  # 키 입력 대기, 10ms
            break

    # 전체 종료
    cap.release()
    cv2.destroyAllWindows()


def getFloorList():
    floorList_ = np.load('/home/jetson/Desktop/workspace/sftp/floorList.npy')
    return floorList_


def putFloor(floor_):
    con = pymysql.connect(host=dbUrl, port=dbPort, user=dbId, password=dbPwd, db="elevator")
    cursor = con.cursor()
    sql = "INSERT INTO FLOOR CTL_FLOOR_JETSON (CTL_FLOOR_JETSON) VALUES (%s)"
    cursor.execute(sql, floor_)

    cursor.close()
    con.commit()
    con.close()

    con = pymysql.connect(host=dbUrl, port=dbPort, user=dbId, password=dbPwd, db="elevator")
    cursor = con.cursor()
    sql = "INSERT INTO FLOOR CTL_FLOOR_WEB (CTL_FLOOR_WEB) VALUES (1)"
    cursor.execute(sql)

    cursor.close()
    con.commit()
    con.close()


while (True):
    print("---------------------------------")
    if triggerCHK() == '1':
        print("start update")
        sftp()
        triggerOff()
    else:
        print("recognize start")
        faceRecog()
