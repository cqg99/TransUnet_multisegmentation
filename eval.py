import numpy as np
import glob
import tqdm
from PIL import Image
import cv2 as cv
import os
from sklearn.metrics import confusion_matrix,cohen_kappa_score
from skimage import io
from skimage import measure
from scipy import ndimage
from sklearn.metrics import f1_score
from PIL import Image

def ConfusionMatrix(numClass, imgPredict, Label):  
    #  返回混淆矩阵
    mask = (Label >= 0) & (Label < numClass)  
    label = numClass * Label[mask] + imgPredict[mask]  
    count = np.bincount(label, minlength = numClass**2)  
    confusionMatrix = count.reshape(numClass, numClass)  
    return confusionMatrix

def OverallAccuracy(confusionMatrix):  
    #  返回所有类的整体像素精度OA
    # acc = (TP + TN) / (TP + TN + FP + TN)  
    OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()  
    return OA
  
def Precision(confusionMatrix):  
    #  返回所有类别的精确率precision  
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 1)
    return precision  

def Recall(confusionMatrix):
    #  返回所有类别的召回率recall
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 0)
    return recall
  
def F1Score(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 1)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 0)
    f1score = 2 * precision * recall / (precision + recall)
    return f1score
def IntersectionOverUnion(confusionMatrix):  
    #  返回交并比IoU
    intersection = np.diag(confusionMatrix)  
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) - np.diag(confusionMatrix)  
    IoU = intersection / union
    return IoU

def MeanIntersectionOverUnion(confusionMatrix):  
    #  返回平均交并比mIoU
    intersection = np.diag(confusionMatrix)  
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) - np.diag(confusionMatrix)  
    IoU = intersection / union
    mIoU = np.nanmean(IoU)  
    return mIoU
  
def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
    #  返回频权交并比FWIoU
    freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)  
    iu = np.diag(confusionMatrix) / (
            np.sum(confusionMatrix, axis = 1) +
            np.sum(confusionMatrix, axis = 0) -
            np.diag(confusionMatrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU


def read_img(filename):
    with Image.open(filename) as img:
        im_width, im_height = img.size
        im_data = np.array(img)
        return im_width, im_height, im_data

def write_img(filename, im_width, im_height, im_data):
    # 确保im_data是3通道（RGB）或者1通道（灰度）的数据
    if len(im_data.shape) == 2:  # 灰度图
        im_data = im_data.reshape((im_height, im_width, 1))
    elif len(im_data.shape) != 3 or im_data.shape[2] not in (1, 3):  
        raise ValueError("Invalid image data shape. It should be (height, width, 1) for grayscale or (height, width, 3) for RGB.")
    
    img = Image.fromarray(im_data.astype('uint8'), 'RGB' if im_data.shape[2] == 3 else 'L')
    img.save(filename)

def eval_new(label_all,predict_all,classNum):
    label_all = label_all.flatten()
    predict_all = predict_all.flatten()

    confusionMatrix = ConfusionMatrix(classNum, predict_all, label_all)
    precision = Precision(confusionMatrix)
    recall = Recall(confusionMatrix)
    OA = OverallAccuracy(confusionMatrix)
    IoU = IntersectionOverUnion(confusionMatrix)
    FWIOU = Frequency_Weighted_Intersection_over_Union(confusionMatrix)
    mIOU = MeanIntersectionOverUnion(confusionMatrix)
    f1ccore = F1Score(confusionMatrix)

    print("")
    print("混淆矩阵:")
    print(confusionMatrix)
    print("精确度:")
    print(precision)
    print("召回率:")
    print(recall)
    print("F1-Score:")
    print(f1ccore)
    print("整体精度:")
    print(OA)
    print("IoU:")
    print(IoU)
    print("mIoU:")
    print(mIOU)
    print("FWIoU:")
    print(FWIOU)

if __name__ == "__main__":
    #################################################################
    #  标签图像文件夹
    LabelPath = "dataset/CamVid/test/labels"
    #  预测图像文件夹
    PredictPath = "dataset/CamVid/test/pre"
    #  类别数目(包括背景)
    classNum = 12

    #  获取文件夹内所有图像
    labelList = os.listdir(LabelPath)
    PredictList = os.listdir(PredictPath)

    #  读取第一个图像，后面要用到它的shape
    # Label0 = cv2.imread(LabelPath + "//" + labelList[0], 0)
    im_proj,im_geotrans,im_width, im_height, Label0 = read_img(LabelPath + "//" + labelList[0])

    #  图像数目
    label_num = len(labelList)


    #  把所有图像放在一个数组里
    label_all = np.zeros((label_num, ) + Label0.shape, np.uint8)
    predict_all = np.zeros((label_num, ) + Label0.shape, np.uint8)
    for i in range(label_num):
        im_proj,im_geotrans,im_width, im_height, Label = read_img(LabelPath + "//" + labelList[i])
        label_all[i] = Label

        im_proj,im_geotrans,im_width, im_height, Predict = read_img(PredictPath + "//" + PredictList[i])
        predict_all[i] = Predict


    eval_new(label_all,predict_all,classNum)