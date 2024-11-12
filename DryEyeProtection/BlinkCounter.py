import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import time

# 눈 감지 실행 여부
eyeDetection = True

def eye_blink_Counter():
    # 웹캠 캡쳐 변수 생성
    cap = cv2.VideoCapture(0)
    # 얼굴 탐지기(최대 얼굴 감지수 = 1)
    detector = FaceMeshDetector(maxFaces=1)
    # 그래프 객체 선언
    plotY = LivePlot(640, 360, [20, 50], invert=True)

    # 눈 메쉬 좌표 리스트
    idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
    # 비율 리스트
    ratioList = []
    # 눈 깜빡임 카운터
    blinkCounter = 0
    pastBlinkCounter = 0
    counter = 0
    color = (255, 0, 255)

    #시간
    timeStart = time.time()

    global eyeDetection
    eyeDetection = True

    while eyeDetection == True:

        # 웹캠으로부터 프레임 읽어오기
        success, img = cap.read()

        # img로 부터 얼굴메쉬 검출(메쉬가 그려진 img와 메쉬 좌표 반환)
        img, faces = detector.findFaceMesh(img, draw=True)

        if faces:
            # 첫번째로 감지된 얼굴 face 변수에 할당
            face = faces[0]
            # idList에 있는 메쉬 포인트 인덱스를 가져와 보라색 원으로 출력
            for id in idList:
                cv2.circle(img, face[id], 5, color, cv2.FILLED)

            # 왼쪽 눈 위,아래,좌,우 좌표 변수 할당
            leftUp = face[159]
            leftDown = face[23]
            leftLeft = face[130]
            leftRight = face[243]
            # 왼쪽 눈 수직거리 & 수평거리 반환
            lenghtVer, _ = detector.findDistance(leftUp,leftDown)
            lenghtHor, _ = detector.findDistance(leftLeft,leftRight)

            cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
            cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)

            # 눈 높이와 너비의 비율(수치상 확인이 쉽게 100곱함)
            ratio = int((lenghtVer/lenghtHor)*100)
            # 눈 비율의 값을 ratioList에 추가
            ratioList.append(ratio)
            # ratioList에 비율 값이 10개만 유지되도록 10개를 넘을시 첫번째 배열 제거
            if len(ratioList) > 3:
                ratioList.pop(0)
            ratioAvg = sum(ratioList)/len(ratioList)

            if ratioAvg < 35 and counter == 0:
                blinkCounter += 1
                color = (0, 200, 0)
                counter = 1
            if counter != 0:
                counter += 1
                if counter > 10 and ratioAvg > 35:
                    counter = 0
                    color = (255, 0, 255)

            cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', (50, 100), colorR=color)

            # 그래프 업데이트(눈 비율 평균값)
            imgPlot = plotY.update(ratioAvg, color)
            img = cv2.resize(img, (640, 360))
            # 웹캠과 눈 깜빡임 비율 그래프를 하나의 이미지 객체로 병합
            imgStack = cvzone.stackImages([img, imgPlot], 2, 1)

            if time.time() - timeStart > 5 and blinkCounter <= pastBlinkCounter + 1:
                print("blink!!")
                timeStart = time.time()
                pastBlinkCounter = blinkCounter
            elif time.time() - timeStart > 5 and blinkCounter > pastBlinkCounter + 1 :
                print("Good")
                timeStart = time.time()
                pastBlinkCounter = blinkCounter


        # 만약 얼굴 인식에 실패한다면
        else:
            img = cv2.resize(img, (640, 360))
            # 웹캠 화면을 두개로 띄우기 위함
            imgStack = cvzone.stackImages([img, img], 2, 1)

        # 웹캡 출력
        cv2.imshow('image',imgStack)
        cv2.waitKey(1)

# 실행을 위한 부분
if __name__ == "__main__":
    eye_blink_Counter()

def stopBlinkEngine():
    global eyeDetection
    eyeDetection = False
