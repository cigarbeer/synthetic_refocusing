import sys
import cv2
import numpy as np 
import matplotlib.pyplot as plt 



def readImg(imgName):
    img = cv2.imread(filename=imgName, flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    img = np.moveaxis(a=img, source=-1, destination=0)
    return img

def readDepthMap(depthMapName):
    img = readImg(depthMapName)
    img = toGray(img)
    return img

def saveImg(img, imgName):
    img = np.moveaxis(a=img, source=0, destination=-1)
    img = np.cvtColor(src=img, code=cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename=imgName, img=img)
    return

def showImg(img):
    img = np.moveaxis(a=img, source=0, destination=-1)
    plt.imshow(X=img)
    plt.show()
    return

def showDepthMap(depthMap):
    plt.imshow(X=depthMap, cmap='gray')
    plt.show()
    return

def toGray(img):
    img = np.moveaxis(a=img, source=0, destination=-1)
    img = cv2.cvtColor(src=img, code=cv2.COLOR_RGB2GRAY)
    return img




if __name__ == '__main__':
    imgName = sys.argv[1]
    depthMapName = sys.argv[2]

    img = readImg(imgName)
    depthMap = readDepthMap(depthMapName)





