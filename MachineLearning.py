import matplotlib.pyplot as plt

from sklearn import datasets #get data to label
from sklearn import svm # suport vector machine

digits = datasets.load_digits() #SVM is going to bring number (0 to 9) --> number recognation

clf = svm.SVC(gamma=0.0001, C=100) #more gamma is little, more precise is the prediction. gamma=1: image is a 9 and prediction is a 3; gamma=0.0001 image is a 9. prediction is a 9

print(len(digits.data)) #1797 examples of digits

x,y = digits.data[:-10], digits.target[:-10] #going to data to target. Data : last element - target : is it right?

clf.fit(x,y)#fit is like the SVM (see image.png). 

print("Prediction:",clf.predict(digits.data[[-2]])) #predict what is the negative first element
plt.imshow(digits.images[-2], cmap=plt.cm.gray_r,interpolation="nearest")#show the image
plt.show()
