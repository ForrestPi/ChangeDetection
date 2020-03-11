import os
import time
import scipy
import sys
import cv2
import numpy
sys.path.append("/home/forrest/caffe/python")
import caffe
import random
import math

IMG_MEAN = np.array([104.008, 116.669, 122.675]).reshape((1, 1, 3))
def prepare_img(img):
    return (img - IMG_MEAN)



model_folder='small'
model_file = './'+model_folder+'/recon__iter_85000.caffemodel'
deploy_file = './'+model_folder+'/deploy.prototxt'


h=1024
w=2048

# feat_layer = "prob"

caffe.set_device(0)
caffe.set_mode_gpu()

net = caffe.Net(deploy_file, model_file, caffe.TEST)
for fi in range(1,18):          
	name1="p"+str(fi)+"_1.jpg"
	name2="p"+str(fi)+"_2.jpg"
	print name1,name2
	test_img1 = "/home/forrest/0426/"+name1
	test_img2 = "/home/forrest/0426/"+name2
	kernel_size = (3, 3)
	sigma = 5
	face_img_rgb1 = cv2.imread(test_img1)
	B1 = cv2.cvtColor(face_img_rgb1,cv2.COLOR_RGB2GRAY)
			B1 = cv2.resize(B1,(w,h)) 
	B1= cv2.GaussianBlur(B1, kernel_size, sigma);

	face_img_rgb2 = cv2.imread(test_img2)
	B2 = cv2.cvtColor(face_img_rgb2,cv2.COLOR_RGB2GRAY) 
			B2 = cv2.resize(B2,(w,h)) 
	B2= cv2.GaussianBlur(B2, kernel_size, sigma);



	name_id=random.randint(0, 20000)
	cv2.imwrite('./results_'+model_folder+'/'+name1,B1)
	cv2.imwrite('./results_'+model_folder+'/'+name2,B2)



	net.blobs['data'].reshape(1, 2, h, w)
	blob_data = net.blobs['data'].data
	B1 = B1.astype(numpy.float32, copy = False)
	B2 = B2.astype(numpy.float32, copy = False)
	blob_data[0][0][:][:] = B1
	blob_data[0][1][:][:] = B2
			start=time.clock()
	out11 = net.forward(blobs=['data'])
			end=time.clock()
			print end-start
	out1=out11['seg-score'][0].copy()
	out1=out1[0,:,:]*255
	cv2.imwrite('./results_'+model_folder+'/'+name2[:-4]+'_'+name1[:-4]+'_maps.jpg',out1)






