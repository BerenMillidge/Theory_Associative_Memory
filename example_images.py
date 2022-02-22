# quick scripts to generate example images for figures
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from functions import *
from data import *
from copy import deepcopy
import pickle

def plot_threshold_value_examples(savename):
    with open(savename, 'rb') as handle:
        data = pickle.load(handle)
    print("LOADED")
    images = data["images"]
    reconstructions = data["reconstructions"]
    sqdiffs = data["sqdiffs"]
    print(images.shape)
    print(reconstructions.shape)
    print(sqdiffs.shape)
    # get buckets
    bucket_edges = [1,10,25,50,75,100,200,500]
    bucket_elements = [[] for i in range(len(bucket_edges))]
    for i,sqdiff in enumerate(sqdiffs):
        # first 0th bucket
        if sqdiff >=bucket_edges[0] and sqdiff <= bucket_edges[1]:
            bucket_elements[0].append(i)
        for j in range(len(bucket_edges)-2):
            j = j+1
            if sqdiff >= bucket_edges[j] and sqdiff <= bucket_edges[j+1]:
                bucket_elements[j].append(i)
        # final bucket
        if sqdiff > bucket_edges[-1]:
            bucket_elements[-1].append(i)
    for b in bucket_elements:
        print(len(b))
    first_indices = [bucket_elements[i][0] for i in range(len(bucket_elements))]
    #print(first_indices)
    #setup figure
    nrow = 2
    ncol = len(bucket_elements)
    fig, ax_array = plt.subplots(nrow, ncol, figsize=(ncol+1,nrow+1), gridspec_kw = {'wspace':0, 'hspace':0, 'top':1.-0.5/(nrow+1), 'bottom': 0.5/(nrow+1), 'left': 0.5/(ncol+1), 'right' :1-0.5/(ncol+1)})
    for i,ax_row in enumerate(ax_array):
        for j,axes in enumerate(ax_row):
            idx = first_indices[j]
            if i == 0:
                axes.imshow(images[idx].transpose(1,2,0))
            if i == 1:
                axes.imshow(reconstructions[idx])
            #axes.set_aspect("auto")
            axes.set_yticklabels([])
            axes.set_xticklabels([])
            axes.set_xticks([])
            axes.set_yticks([])
    #fig.suptitle("Cifar10 Fraction Masked")
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.subplots_adjust(wspace=0, hspace=0)
    #plt.tight_layout()

    plt.savefig("example_images/threshold_examples_cifar10.jpg", format="jpeg",bbox_inches = "tight", pad_inches = 0)
    plt.show()
    

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
    
    plot_threshold_value_examples("example_reconstructions_thresholds_saved_3")
    