# This is a Normal Bayesian classifier capable of classifying 10 letters based on feature data

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


#Read the test and training data
Train_Data = np.loadtxt("ocr10-train-win.txt",np.float32,delimiter=",")
Test_Data =  np.loadtxt("ocr10-test-win.txt",np.float32,delimiter=",")

#print(Train_Data.shape)#(15455, 129)
#
#print(Test_Data.shape)#(7730, 129)

####Split the features from the labels
train_features = Train_Data[:,:128]
Labels=np.array(Train_Data[:,128],np.int32)

test_features = Test_Data[:,:128]
test_Labels = Test_Data[:,128]

###Create a normal baye's classifier and train it on training data features
Bayes= cv.ml.NormalBayesClassifier_create()
Bayes.train(train_features,cv.ml.ROW_SAMPLE,Labels)

ret, predict_labels, predictProba = Bayes.predictProb(test_features)

###Get the number of misclassifications using list comprehension
Error = len([i for i, (test_Labels,predict_labels) in enumerate(zip(test_Labels,predict_labels))
             if test_Labels != predict_labels])
###Print the number of misclassifications and percentage error
print("Number of Errors is {}\nPercentage Error is {:.3f}".format(Error,(Error/test_Labels.shape[0])))


######To output a confusion matrix##############

###Initialize list to store predicted class score
A_Class=[0,0,0,0,0,0,0,0,0,0]
B_Class=[0,0,0,0,0,0,0,0,0,0]
C_Class=[0,0,0,0,0,0,0,0,0,0]
D_Class=[0,0,0,0,0,0,0,0,0,0]
E_Class=[0,0,0,0,0,0,0,0,0,0]
F_Class=[0,0,0,0,0,0,0,0,0,0]
G_Class=[0,0,0,0,0,0,0,0,0,0]
H_Class=[0,0,0,0,0,0,0,0,0,0]
I_Class=[0,0,0,0,0,0,0,0,0,0]
J_Class=[0,0,0,0,0,0,0,0,0,0]

###Store the list of scores for each class in another list
Classes= [A_Class,B_Class,C_Class,D_Class,E_Class,F_Class,G_Class,H_Class,I_Class,J_Class]


letter = 0

for z in Classes:
        for i in range(len(test_Labels)): ###Go through test label rows
            if test_Labels[i] == letter: ###If original label is letter A,B,C etc
                if predict_labels[i]==0: ###Increase matching score if it is A in predicted and same for all
                    z[0] +=1
                elif predict_labels[i]==1:
                    z[1]+=1
                elif predict_labels[i]==2:
                    z[2] += 1
                elif predict_labels[i]==3:
                    z[3]+=1
                elif predict_labels[i]==4:
                    z[4] += 1
                elif predict_labels[i]==5:
                    z[5]+=1
                elif predict_labels[i]==6:
                    z[6] += 1
                elif predict_labels[i]==7:
                    z[7]+=1
                elif predict_labels[i]==8:
                    z[8] += 1
                elif predict_labels[i]==9:
                    z[9]+=1
        letter +=1  ###Move to the next letter


letters= ["a","b","c","d","e","f","g","h","i","j"]
format_row = "{:>8}" * (len(letters) + 1)

print(format_row.format("", *letters))

###Print the confusion matrix
for classification, row in zip(letters, Classes):
    print(format_row.format(classification, *row))


###Now to display the letters using matplotlib
plt.figure()
plt.suptitle("Test Data Sample Letters")
val = 1 #Subplot number
for letter in range(10):
    ####Pick the first index where that letter shows up in the labels class and output the reshaped features
    letter = train_features[list(Labels).index(letter),:].reshape([16,8])
    plt.subplot(5,2,val)
    plt.imshow(letter)
    #plt.grid = False
    plt.xticks([])
    plt.yticks([])
    val+=1
plt.show()