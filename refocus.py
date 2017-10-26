import sys
import cv2
import numpy as np 
import matplotlib.pyplot as plt 



def readImg(imgName):
    img = cv2.imread(filename=imgName, flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    img = np.moveaxis(a=img, source=-1, destination=0)
    return img.astype(np.uint64)

def readDepthMap(depthMapName):
    img = readImg(depthMapName)
    img = toGray(img)
    return img.astype(np.uint64)

def saveImg(img, imgName):
    img = np.moveaxis(a=img, source=0, destination=-1)
    img = cv2.cvtColor(src=img, code=cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename=imgName, img=img)
    return

def showImg(img):
    img = np.moveaxis(a=img, source=0, destination=-1)
    plt.imshow(X=img.astype(np.uint8))
    plt.show()
    return

def showDepthMap(depthMap):
    plt.imshow(X=depthMap.astype(np.uint8), cmap='gray')
    plt.show()
    return

def toGray(img):
    img = np.moveaxis(a=img, source=0, destination=-1)
    img = cv2.cvtColor(src=img.astype(np.uint8), code=cv2.COLOR_RGB2GRAY)
    return img.astype(np.uint64)

def depthSegment(depthMap, depthLayer, lowerBound=0, upperBound=255):
    cut = ((upperBound-lowerBound) // depthLayer) + 1 
    l = lowerBound
    u = lowerBound + cut
    seg = []
    for i in range(depthLayer):
        rawSeg = (depthMap >= l) & (depthMap < u)
        dilateSeg = dilate(rawSeg)
        seg.append(dilateSeg)
        l = l + cut
        u = u + cut
    return seg


def depthDependentBlur(img, depthLayer, focusSegmentIndex, dofLevel=1):
    blurredImg = []
    for i in range(depthLayer):
        bImg = None
        if i == focusSegmentIndex:
            bImg = sharpen(img)
        else:
            sigma = calcSigma(i, focusSegmentIndex, dofLevel)
            bImg = blur(img, sigma)
        blurredImg.append(bImg)
    return blurredImg

def calcSigma(segmentIndex, focusSegmentIndex, dofLevel):
    sigma = dofLevel * np.abs(segmentIndex-focusSegmentIndex)
    return sigma

def blur(img, sigma):
    img = np.moveaxis(a=img, source=0, destination=-1)
    img = cv2.GaussianBlur(src=img.astype(np.uint8), ksize=(0, 0), sigmaX=sigma)
    img = np.moveaxis(a=img, source=-1, destination=0)
    return img.astype(np.uint64)

def sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = np.moveaxis(a=img, source=0, destination=-1)
    img = cv2.filter2D(src=img.astype(np.uint8), ddepth=cv2.CV_8U, kernel=kernel)
    img = np.moveaxis(a=img, source=-1, destination=0)
    return img.astype(np.uint64)

def dilate(depthSegmentation):
    # img = np.moveaxis(a=img, source=0, destination=-1)
    kernel = np.ones(shape=(5, 5), dtype=np.uint8)
    depthSegmentation = cv2.dilate(src=depthSegmentation.astype(np.uint8), kernel=kernel)
    # depthSegmentation = np.moveaxis(a=depthSegmentation, source=-1, destination=0)
    return depthSegmentation.astype(np.uint64)

def calcBoundary(depthSegmentation):
    boundary = []
    for i in range(len(depthSegmentation)-1):
        dilatedForeground = dilate(depthSegmentation[i])
        dilatedBackground = dilate(depthSegmentation[i+1])
        b = dilatedForeground & dilatedBackground
        boundary.append(b.astype(np.bool))
    return boundary

def render(blurredImg, depthSegmentation, boundary, alpha=2):
    result = blurredImg[0] * depthSegmentation[0]
    for i in range(1, len(blurredImg)):
        result += blurredImg[i] * depthSegmentation[i]
    temp = np.array(result)
    result = np.moveaxis(a=result, source=0, destination=-1)
    for b in boundary:
        result[b] //= alpha
    result = np.moveaxis(a=result, source=-1, destination=0)
    return result.astype(np.uint8), temp


# if __name__ == '__main__':
#     imgName = sys.argv[1]
#     depthMapName = sys.argv[2]

#     img = readImg(imgName)
#     depthMap = readDepthMap(depthMapName)





