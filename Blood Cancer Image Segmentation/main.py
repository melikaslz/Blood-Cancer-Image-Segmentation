import os
from functions import *
path = os.getcwd()+'/img'
images = os.listdir(path)
mean_fit = 0
for i in images:
    image_path = path+'/'+i
    image = cv2.imread(image_path)
    color = GA(image,20,100,4,0.4)
    mean_fit += fitness(image,4,color)
    out_img = Recolored(image,color)
    print('PSNR :{}'.format(psnr(image,out_img)))