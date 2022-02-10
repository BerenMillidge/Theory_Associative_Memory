# quick script to generate example images for figures
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from functions import *
from data import *
from copy import deepcopy


    

if __name__ == '__main__':
    trainset_cifar, testset_cifar = get_cifar10(10000)
    imgs = trainset_cifar[0][0]
    print(imgs.shape)
    for i in range(5):
        fig = plt.figure()
        plt.imshow(imgs[i].reshape(3,32,32).permute(1,2,0))
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig("example_images/img_3_" + str(i) + ".jpg")
        plt.show()
    # query img
    img = imgs[0]
    print(img.shape)
    img = img.reshape(32 * 32 * 3)
    halved = halve_continuous_img(img)
    print(halved.shape)
    fig = plt.figure()
    plt.imshow(halved.permute(1,2,0))
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("example_images/query_img_3" + str(i) + ".jpg")
    plt.show()
    
    