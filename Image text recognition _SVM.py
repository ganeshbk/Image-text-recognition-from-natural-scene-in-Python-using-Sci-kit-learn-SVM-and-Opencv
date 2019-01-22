# GitHub -----  https://github.com/pidwid/

import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# directory of the images folder  **** Create individual folder for individual image type***
fn_dir = "C:/Users/lucky/Desktop/data/"

c = 0
(images, lables, names, id) = ([], [], {}, 0)
for (subdir, dirs, files) in os.walk(fn_dir):
    for subdir in dirs:
        names[id] = subdir
        mypath = os.path.join(fn_dir, subdir)
        for item in os.listdir(mypath):
            if ".jpg" in item:  # define the format of the image *** make sure to have all images in same format***
                label = id
                image = cv2.imread(os.path.join(mypath, item), 0)
                r_image = cv2.resize(image, (30, 30)).flatten()
                # print(r_image,"r_images")
                if image is not None:
                    images.append(r_image)
                    lables.append(names[id])
        id += 1

# print ( images )

(images, lables) = [np.array(lis) for lis in [images, lables]]

# print ( lables ,"images")
# print ( images ,"labels")

nf = 10
pca = PCA(n_components=nf)
# print ( images.shape,"images.shape" )
pca.fit(images)
img_feature = pca.transform(images)

# print ( img_feature )

classifier = SVC(verbose=0, kernel="poly", degree=3)
classifier.fit(img_feature, lables)

path = "C:/Users/lucky/Desktop/3/"  # path of the test image  ***store one or multiple***
result = []
for item in os.listdir(path):
    if ".jpg" in item:
        label = id
        test_image = cv2.imread(os.path.join(path, item), 0)
        test_arr_img = cv2.resize(test_image, (30, 30))
        test_arr_img = np.array(test_arr_img).flatten()
        temp = []
        temp.append(test_arr_img)
        im_test = pca.transform(temp)
        pred = classifier.predict(im_test)
        result.append(pred.item())

print("Predicted values are :- ", result)

