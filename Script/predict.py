import cv2
import numpy as np
from deepgaze.color_classification import HistogramColorClassifier
from pathlib import Path
from matplotlib import pyplot as plt

def showResultGraph(amount, label1, label2, result):
    labels = (label1, label2)
    font_size = 10
    width = 0.5 
    plt.barh(np.arange(amount), result, width, color='r')
    plt.yticks(np.arange(amount) + width/2.,labels , rotation=0, size=font_size)
    plt.xlim(0.0, 10.0)
    plt.ylim(-0.5, 8.0)
    plt.xlabel('Probability', size=font_size)
    plt.show()

def resize(str):
    image = cv2.imread(str)
    resize_image = cv2.resize(image, (800,600), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str, resize_image)

def determineFanType():
    classifier = HistogramColorClassifier(channels=[0, 1, 2], hist_size=[128, 128, 128], hist_range=[0, 256, 0, 256, 0, 256], hist_type='BGR')

    bif_model = cv2.imread('./data/bif/REFERENCE.jpg')
    classifier.addModelHistogram(bif_model)

    fck_model = cv2.imread('./data/fck/REFERENCE.jpg')
    classifier.addModelHistogram(fck_model)

    path = input("Please enter the path of the image you want to check: ")

    filecheck = Path(path)
    if filecheck.is_file():  

        resize(path)
        image = cv2.imread(path)
        result = classifier.returnHistogramComparisonArray(image, method="intersection")

        if result[0] > result[1]:
            print("Picture is most likely of a BIF fan.")

        elif result[0] < result[1]:
            print("Picture is most likely of an FCK fan.")

        else:
            print("Cant determine fan type. Try a different picture")

        showResultGraph(2, 'BIF', 'FCK', result)
    else:
        print("File dosent exist. Run the file again and try a different path.")

determineFanType()