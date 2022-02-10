# functions for plotting graphs given the associative memory networks
from tokenize import PlainToken
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from functions import *
from data import *
from copy import deepcopy

def plot_capacity_graphs(Ns, imgs, beta,fs, labels, image_perturb_fn = halve_continuous_img,sep_fn = separation_max, sep_param=1000,sigma=0.1,plot_results = False):
    corrects_list = [[] for i in range(len(fs))]
    for i,(f, label) in enumerate(zip(fs, labels)):
        print(label.upper())
        for N in Ns:
            print(N)
            N_correct = PC_retrieve_store_continuous(imgs,N,beta=beta,num_plot=0,f = f, image_perturb_fn = gaussian_perturb_image,sigma = sigma,sep_fn = sep_fn, sep_param=sep_param)
            corrects_list[i].append(N_correct)


    #if plot_results:
    #    plt.title("Memory Capacity of MCHN and PC associative networks")
    #    for i in range(len(fs)):
    #        plt.plot(Ns, corrects_list[i], label = labels[i])##

        #plt.xlabel("Images Stored")
        #plt.ylabel("Fraction Correctly Retrieved")
        #plt.legend()
        #plt.show()
    return np.array(corrects_list).reshape(len(fs),len(Ns))

def N_runs_capacity_graphs(N_runs, Ns, imgs, beta,fs,fn_labels, image_perturb_fn = halve_continuous_img, sep_fn = separation_max, sep_param = 1000, sigma=0.1,sname = "tiny_N_capacity_results.npy", figname = "tiny_N_runs_capacity_graph.jpg", load_data = False, plot_results = True, save_continuously=True):
    if not load_data:
        N_corrects = []
        max_N = Ns[-1]
        for n in range(N_runs):
            X = imgs[(max_N*n):(max_N * (n+1))]
            corrects_list = plot_capacity_graphs(Ns, X, beta, fs, fn_labels, image_perturb_fn=image_perturb_fn, sep_fn = sep_fn, sep_param = sep_param, sigma=sigma)
            N_corrects.append(corrects_list)
            if save_continuously:
                prelim_N_corrects = np.array(deepcopy(N_corrects))
                np.save(sname, prelim_N_corrects)
        N_corrects = np.array(N_corrects)
        np.save(sname, N_corrects)
    else:
        N_corrects = np.load(sname)
    # begin plot
    if plot_results:
        mean_corrects = np.mean(N_corrects,axis=0)
        std_corrects = np.std(N_corrects, axis=0)
        fig = plt.figure(figsize=(12,10))
        sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
        sns.despine(left=False,top=True, right=True, bottom=False)
        plt.title("Memory Capacity by Similarity Function",fontsize=30)
        for i in range(len(fs)):
            plt.plot(Ns, mean_corrects[i,:],label=fn_labels[i])
            plt.fill_between(Ns, mean_corrects[i,:] - std_corrects[i,:], mean_corrects[i,:]+std_corrects[i,:],alpha=0.5)
        plt.xlabel("Images Stored",fontsize=25)
        plt.ylabel("Fraction Correctly Retrieved",fontsize=25)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        plt.legend(fontsize=25)
        plt.ylim(bottom=0)
        fig.tight_layout()
        plt.savefig(figname, format="jpeg")
        plt.show()
    return N_corrects

def plot_noise_level_graphs(N, imgs, beta, fs, labels, sigmas,use_norm = True,sep_fn = separation_max, sep_param=1000):
    corrects_list = [[] for i in range(len(sigmas))]
    for i,sigma in enumerate(sigmas):
        print("SIGMA: ", sigma)
        corrects = [[] for i in range(len(fs))]
        for j, (f,label) in enumerate(zip(fs,labels)):
            print(label)
            N_correct = PC_retrieve_store_continuous(imgs, N, beta=beta, num_plot=0,f=f,sigma=sigma,image_perturb_fn=gaussian_perturb_image,use_norm = use_norm,sep_fn = sep_fn, sep_param = sep_param)
            corrects[j].append(deepcopy(N_correct))
        corrects_list[i].append(np.array(corrects))
    corrects_list = np.array(corrects_list)
    return corrects_list.reshape(len(sigmas), len(fs))

def N_runs_noise_level_graphs(N_runs, N, imgs, beta,fs,fn_labels, sigmas, sep_fn = separation_max, sep_param = 1000, load_data = False,sname = "tiny_N_noise_level_results.npy", figname = "tiny_N_runs_noise_levels.jpg", plot_results = True):
    if not load_data:
        N_corrects = []
        for n in range(N_runs):
            X = imgs[(N*n):(N * (n+1))]
            corrects_list = plot_noise_level_graphs(N, X, beta, fs, fn_labels, sigmas, sep_fn = sep_fn, sep_param = sep_param)
            N_corrects.append(corrects_list)
        N_corrects = np.array(N_corrects)
        np.save(sname, N_corrects)
    else:
        N_corrects = np.load(sname)

    if plot_results:
        mean_corrects = np.mean(N_corrects,axis=0)
        std_corrects = np.std(N_corrects, axis=0)
        # begin plot
        fig = plt.figure(figsize=(12,10))
        sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
        sns.despine(left=False,top=True, right=True, bottom=False)
        plt.title("Memory Capacity by Noise Level",fontsize=30)
        for i in range(len(fs)):
            plt.plot(sigmas, mean_corrects[:,i],label=fn_labels[i])
            plt.fill_between(sigmas, mean_corrects[:,i] - std_corrects[:,i], mean_corrects[:,i]+std_corrects[:,i],alpha=0.5)
        plt.xlabel("Noise variance (sigma)",fontsize=25)
        plt.ylabel("Fraction Correctly Retrieved",fontsize=25)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        plt.legend(fontsize=25)
        plt.ylim(bottom=0)
        fig.tight_layout()
        plt.savefig(figname, format="jpeg")
        plt.show()
    return N_corrects

def plot_mask_frac_graphs(N, imgs, beta, fs, labels, mask_fracs,use_norm = True,sep_fn = separation_max, sep_param=1000):
    corrects_list = [[] for i in range(len(mask_fracs))]
    for i,mask_frac in enumerate(mask_fracs):
        print("MASK FRAC: ", mask_frac)
        corrects = [[] for i in range(len(fs))]
        for j, (f,label) in enumerate(zip(fs,labels)):
            print(label)
            N_correct = PC_retrieve_store_continuous(imgs, N, beta=beta, num_plot=0,f=f,sigma=mask_frac,image_perturb_fn=mask_continuous_img,use_norm = use_norm,sep_fn = sep_fn, sep_param = sep_param)
            corrects[j].append(deepcopy(N_correct))
        corrects_list[i].append(np.array(corrects))
    corrects_list = np.array(corrects_list)
    return corrects_list.reshape(len(mask_fracs),len(fs))

def N_runs_mask_frac_graphs(N_runs, N, imgs, beta,fs,fn_labels, mask_fracs, sep_fn = separation_max, sep_param = 1000, load_data = False,sname = "tiny_N_mask_frac_results.npy", figname = "tiny_N_runs_mask_fracs.jpg", plot_results = True):
    if not load_data:
        N_corrects = []
        for n in range(N_runs):
            X = imgs[(N*n):(N * (n+1))]
            corrects_list = plot_mask_frac_graphs(N, X, beta, fs, fn_labels, mask_fracs, sep_fn = sep_fn, sep_param = sep_param)
            N_corrects.append(corrects_list)
        N_corrects = np.array(N_corrects)
        np.save(sname, N_corrects)
    else:
        N_corrects = np.load(sname)

    if plot_results:
        mean_corrects = np.mean(N_corrects,axis=0)
        std_corrects = np.std(N_corrects, axis=0)
        # begin plot
        fig = plt.figure(figsize=(12,10))
        sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
        sns.despine(left=False,top=True, right=True, bottom=False)
        plt.title("Memory Capacity by Fraction Masked",fontsize=30)
        for i in range(len(fs)):
            plt.plot(mask_fracs, mean_corrects[:,i],label=fn_labels[i])
            plt.fill_between(mask_fracs, mean_corrects[:,i] - std_corrects[:,i], mean_corrects[:,i]+std_corrects[:,i],alpha=0.5)
        plt.xlabel("Fraction Masked",fontsize=25)
        plt.ylabel("Fraction Correctly Retrieved",fontsize=25)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        plt.legend(fontsize=25)
        plt.ylim(bottom=0)
        fig.tight_layout()
        plt.savefig(figname, format="jpeg")
        plt.show()
    return N_corrects

def plot_separation_function_graph(Ns, imgs, beta,sep_fns, labels, image_perturb_fn = halve_continuous_img,sigma=1,f=manhatten_distance,use_norm = True,sep_param = 1000, plot_results = True):
    corrects_list = [[] for i in range(len(sep_fns))]
    for i,(sep_fn, label) in enumerate(zip(sep_fns, labels)):
        print(label.upper())
        for N in Ns:
            print(N)
            N_correct = PC_retrieve_store_continuous(imgs,N,beta=beta,num_plot=0,image_perturb_fn = image_perturb_fn,sigma = sigma,sep_fn = sep_fn,f=f,use_norm = use_norm,sep_param = sep_param)
            corrects_list[i].append(N_correct)

    #if plot_results:
    #    plt.title("Memory Capacity by separation function")
    #    for i in range(len(sep_fns)):
    #        plt.plot(Ns, corrects_list[i], label = labels[i])

        #plt.xlabel("Images Stored")
        #plt.ylabel("Fraction Correctly Retrieved")
        #plt.legend()
        #plt.show()
    return np.array(corrects_list).reshape(len(sep_fns), len(Ns))

def N_runs_separation_function_graphs(N_runs, Ns, imgs, beta,sep_fns,fn_labels, f = manhatten_distance, sep_fn = separation_max, sep_param = 1000, load_data = False,sname = "tiny_N_runs_separation_function_results_2.npy", figname = "tiny_N_runs_separation_functions_2.jpg", plot_results = True):
    if not load_data:
        N_corrects = []
        max_N = Ns[-1]
        for n in range(N_runs):
            X = imgs[(max_N*n):(max_N * (n+1))]
            corrects_list = plot_separation_function_graph(Ns, X, beta, sep_fns, fn_labels, f=f,sep_param=sep_param)
            N_corrects.append(corrects_list)
        N_corrects = np.array(N_corrects)
        np.save(sname, N_corrects)
    else:
        N_corrects = np.load(sname)
    if plot_results:
        mean_corrects = np.mean(N_corrects,axis=0)
        std_corrects = np.std(N_corrects, axis=0)
        # begin plot
        fig = plt.figure(figsize=(12,10))
        sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
        sns.despine(left=False,top=True, right=True, bottom=False)
        plt.title("Memory Capacity by Separation Functions",fontsize=30)
        xs = np.arange(0, len(Ns))
        for i in range(len(sep_fns)):
            plt.plot(Ns, mean_corrects[i,:],label=fn_labels[i])
            plt.fill_between(Ns, mean_corrects[i,:] - std_corrects[i,:], mean_corrects[i,:]+std_corrects[i,:],alpha=0.5)
        plt.xlabel("Number of Images",fontsize=25)
        plt.ylabel("Fraction Correctly Retrieved",fontsize=25)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        plt.legend(fontsize=25)
        plt.ylim(bottom=0)
        fig.tight_layout()
        plt.savefig(figname, format="jpeg")
        plt.show()
    return N_corrects

def generate_demonstration_reconstructions(imgs, N, f=manhatten_distance, image_perturb_fn = mask_continuous_img, perturb_vals =[], sep_fn = separation_max, sep_param=1000, use_norm=True):
    X = imgs[0:N,:]
    img_shape = X[0].shape
    img_len = np.prod(np.array(img_shape))
    if len(img_shape) != 1:
        X = reshape_img_list(X, img_len)
    img_idx = int(np.random.choice(N))
    init_img = deepcopy(X[img_idx,:])
    show_init_img = deepcopy(init_img).reshape(img_shape).permute(1,2,0)
    perturbed_imgs = []
    reconstructed_imgs = []
    beta = 1
    for val in perturb_vals:
        query_img = image_perturb_fn(init_img, val).reshape(1, img_len)
        perturbed_imgs.append(deepcopy(query_img.reshape(img_shape).permute(1,2,0)))
        out = general_update_rule(X,query_img,beta, f,sep=sep_fn, sep_param=sep_param,norm=use_norm).reshape(img_len)
        reconstructed_imgs.append(deepcopy(out).reshape(img_shape).permute(1,2,0))
    N_vals = len(perturb_vals)
    ncol = N_vals
    nrow = 3
    fig, ax_array = plt.subplots(nrow, ncol, figsize=(ncol+1,nrow+1), gridspec_kw = {'wspace':0, 'hspace':0, 'top':1.-0.5/(nrow+1), 'bottom': 0.5/(nrow+1), 'left': 0.5/(ncol+1), 'right' :1-0.5/(ncol+1)})
    for i,ax_row in enumerate(ax_array):
        for j,axes in enumerate(ax_row):
            if i == 0:
                axes.imshow(show_init_img)
            if i == 1:
                axes.imshow(perturbed_imgs[j])
                #axes.set_title("Fraction Masked " + str(perturb_vals[i]))
            if i == 2:
                axes.imshow(reconstructed_imgs[j])
            #axes.set_aspect("auto")
            axes.set_yticklabels([])
            axes.set_xticklabels([])
            axes.set_xticks([])
            axes.set_yticks([])
    #fig.suptitle("Cifar10 Fraction Masked")
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.subplots_adjust(wspace=0, hspace=0)
    #plt.tight_layout()

    plt.savefig("figures/cifar10_reconstruction_examples_masked_3.jpg", format="jpeg",bbox_inches = "tight", pad_inches = 0)
    plt.show()

def visualize_heteroassociative_MCHN(imgs, N,N_imgs=5, f=manhatten_distance, sep_fn = separation_softmax, sep_param =100000, use_norm=True):
    X = imgs[0:N,:]
    img_shape = X[0].shape
    img_len = np.prod(np.array(img_shape))

    if len(img_shape) != 1:
        X = reshape_img_list(X, img_len)

    M = torch.zeros_like(X)
    # setup heteroassociative memory
    for i,img in enumerate(X):
        halved = halve_continuous_img(img, reversed=True).reshape(img_len)
        M[i,:] = halved
    beta = 1000
    query_imgs = []
    reconstructed_imgs = []
    for i in range(N_imgs):
        query_img = halve_continuous_img(X[i,:], None).reshape(1, img_len)
        query_imgs.append(deepcopy(query_img).reshape(img_shape).permute(1,2,0))
        out = heteroassociative_update_rule(X,M,query_img,beta, f,sep=sep_fn, sep_param=sep_param,norm=use_norm).reshape(img_len)
        reconstructed_imgs.append(deepcopy(out).reshape(img_shape).permute(1,2,0))
    ncol = 4
    nrow = N_imgs
    fig, ax_array = plt.subplots(nrow, ncol, figsize=(ncol+1,nrow+1), gridspec_kw = {'wspace':0.1, 'hspace':0.1, 'top':1.-0.5/(nrow+1), 'bottom': 0.5/(nrow+1), 'left': 0.5/(ncol+1), 'right' :1-0.5/(ncol+1)})
    for i,ax_row in enumerate(ax_array):
        for j,axes in enumerate(ax_row):
            if j == 0:
                axes.imshow(X[i,:].reshape(img_shape).permute(1,2,0))
            if j == 1:
                axes.imshow(query_imgs[i])
                #axes.set_title("Example " + str(i))
            if j == 2:
                axes.imshow(M[i,:].reshape(img_shape).permute(1,2,0))
            if j == 3:
                axes.imshow(reconstructed_imgs[i])
            axes.set_yticklabels([])
            axes.set_xticklabels([])
            axes.set_xticks([])
            axes.set_yticks([])
    #fig.suptitle("Example MCHN Heteroassociation")
    #plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("figures/MCHN_cifar10_heteroassociation_4.jpg", format="jpeg",bbox_inches="tight",pad_inches = 0)
    plt.show()

def visualize_heteroassociative_hopfield(imgs, N,N_imgs=5, f=normalized_dot_product, sep_fn = separation_softmax, sep_param =100000, use_norm=True):
    X = torch.sign(imgs[0:N,:])
    print("X ", imgs.shape)
    img_shape = deepcopy(X[0].shape)
    print("IMG SHAPE ", img_shape)
    img_len = np.prod(np.array(img_shape))
    print(img_len)

    M = torch.zeros((N,784))
    # setup heteroassociative memory
    for i,img in enumerate(X):
        print(img.shape)
        img = img.reshape(784)
        halved = halve_continuous_img(img, reversed=True)#.reshape(imglen)
        print(halved.shape)
        halved = halved.reshape(img_len)
        print(halved.shape)
        halved = halved.reshape(784)
        M[i,:] = binary_to_bipolar(halved)


    if len(img_shape) != 1:
        X = reshape_img_list(X, img_len, opt_fn = binary_to_bipolar)

    beta = 1
    query_imgs = []
    reconstructed_imgs = []
    for i in range(N_imgs):
        query_img = binary_to_bipolar(halve_continuous_img(X[i,:], None)).reshape(1, img_len)
        query_imgs.append(deepcopy(query_img).reshape(img_shape))
        out = binary_to_bipolar(torch.sign(heteroassociative_update_rule(X,M,query_img,beta, f,sep=sep_fn, sep_param=sep_param,norm=use_norm).reshape(img_len)))
        reconstructed_imgs.append(deepcopy(out).reshape(img_shape))
    ncol = 4
    nrow = N_imgs
    fig, ax_array = plt.subplots(nrow, ncol, figsize=(ncol+1,nrow+1), gridspec_kw = {'wspace':0.1, 'hspace':0.1, 'top':1.-0.5/(nrow+1), 'bottom': 0.5/(nrow+1), 'left': 0.5/(ncol+1), 'right' :1-0.5/(ncol+1)})
    for i,ax_row in enumerate(ax_array):
        for j,axes in enumerate(ax_row):
            if j == 0:
                #print(img_shape)
                im_shape = (28,28)
                #print(X[j,:].reshape(im_shape).shape)
                axes.imshow(X[i,:].reshape(im_shape))
            if j == 1:
                axes.imshow(query_imgs[i].reshape(im_shape))
                #axes.set_title("Example " + str(i))
            if j == 2:
                axes.imshow(binary_to_bipolar(M[i,:]).reshape(im_shape))
            if j == 3:
                axes.imshow(reconstructed_imgs[i].reshape(im_shape))
            axes.set_yticklabels([])
            axes.set_xticklabels([])
            axes.set_xticks([])
            axes.set_yticks([])
    #fig.suptitle("Example MCHN Heteroassociation")
    #plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("figures/classical_HN_MNIST_heteroassociation_2.jpg", format="jpeg",bbox_inches="tight", pad_inches=0)
    plt.show()
    

def N_runs_auto_vs_heteroassociative(N_runs, Ns, imgs, beta,f = normalized_dot_product, image_perturb_fn = halve_continuous_img, sep_fn = separation_identity, sep_param = 1000, sigma=0.1,sname = "tiny_N_capacity_results.npy", figname = "tiny_N_runs_capacity_graph.jpg", load_data = False, plot_results = True, save_continuously=True, network_type=""):
    if not load_data:
        N_corrects_autoassociative = []
        N_corrects_heteroassociative = []
        max_N = Ns[-1]

        for n in range(N_runs):
            # autoassociative 
            n_corrects_auto = []
            X = imgs[(max_N*n):(max_N * (n+1))]
            for N in Ns:
                N_correct = PC_retrieve_store_continuous(X, N, beta=beta, num_plot=0, f=f, image_perturb_fn=image_perturb_fn, sigma=sigma, sep_fn = sep_fn, sep_param = sep_param, network_type = network_type)
                n_corrects_auto.append(N_correct)
            n_corrects_auto = np.array(n_corrects_auto)
            N_corrects_autoassociative.append(n_corrects_auto)
            
            #heteroassociative
            n_corrects_hetero = []
            #X = imgs[(max_N*n):(max_N * (n+1))]
            #setup heteroassociaitve memory as bottom half
            print("X: ", X.shape)
            P = torch.zeros((max_N, np.prod(np.array(X[0].shape))))
            print(P.shape)
            for i,img in enumerate(X):
                img = img.reshape(np.prod(np.array(img.shape)))
                #print("IMG: " + str(i) + "  " + str(img.shape))
                out_img = halve_continuous_img(img, reversed = True)
                #print(out_img.shape)
                out_img = out_img.reshape(np.prod(np.array(X[0].shape)))
                if network_type == "classical_hopfield":
                    out_img = binary_to_bipolar(out_img)
                P[i,:] = out_img
                
            print("P: ", P.shape)
            for N in Ns:
                N_correct = PC_retrieve_store_continuous(X, N,P=P, beta=beta, num_plot=0, f=f, image_perturb_fn=image_perturb_fn, sigma=sigma, sep_fn = sep_fn, sep_param = sep_param,network_type=network_type)
                n_corrects_hetero.append(N_correct)
            n_corrects_hetero = np.array(n_corrects_hetero)
            N_corrects_heteroassociative.append(n_corrects_hetero)

            if save_continuously:
                prelim_corrects_auto = np.array(deepcopy(N_corrects_autoassociative))
                prelim_corrects_hetero = np.array(deepcopy(N_corrects_heteroassociative))
                np.save(sname + "_auto.npy", prelim_corrects_auto)
                np.save(sname + "_hetero.npy", prelim_corrects_hetero)
                
        N_corrects_autoassociative = np.array(N_corrects_autoassociative)
        N_corrects_heteroassociative = np.array(N_corrects_heteroassociative)
        np.save(sname + "_auto.npy", N_corrects_autoassociative)
        np.save(sname + "_hetero.npy", N_corrects_heteroassociative)
    else:
        N_corrects_autoassociative = np.load(sname + "_auto.npy")
        N_corrects_heteroassociative = np.load(sname + "_hetero.npy")
    # begin plot
    if plot_results:
        mean_auto = np.mean(N_corrects_autoassociative,axis=0)
        mean_hetero = np.mean(N_corrects_heteroassociative, axis=0)
        std_auto = np.std(N_corrects_autoassociative, axis=0)
        std_hetero = np.std(N_corrects_heteroassociative,axis=0)
        print("SHAPES")
        print(mean_auto.shape)
        print(std_auto.shape)
        fig = plt.figure(figsize=(12,10))
        sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
        sns.despine(left=False,top=True, right=True, bottom=False)
        plt.title("Memory Capacity for Auto vs Heteroassociative",fontsize=30)
        # auto
        plt.plot(Ns, mean_auto,label="Autoassociative")
        plt.fill_between(Ns, mean_auto - std_auto, mean_auto+std_auto,alpha=0.5)
        #hetero
        plt.plot(Ns, mean_hetero,label="Heteroassociative")
        plt.fill_between(Ns, mean_hetero - std_hetero, mean_hetero+std_hetero,alpha=0.5)
        plt.xlabel("Images Stored",fontsize=25)
        plt.ylabel("Fraction Correctly Retrieved",fontsize=25)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        plt.legend(fontsize=25)
        plt.ylim(bottom=0)
        fig.tight_layout()
        plt.savefig(figname, format="jpeg")
        plt.show()
        
    return N_corrects_autoassociative, N_corrects_heteroassociative

if __name__ == '__main__':
    #trainset_cifar, testset_cifar = get_cifar10(10000)
    #imgs = trainset_cifar[0][0]
    trainset_mnist, testset_mnist = load_mnist(60000)
    imgs = trainset_mnist[0][0]
    #imgs = load_tiny_imagenet(N_imgs=10000)
    # separation functions
    
    PLOT_RESULTS = True
    LOAD_DATA = True
    #dataset_str = "mnist_longer_capacity_"
    dataset_str = "mnist_"
    
    
    sep_fns = [separation_log, separation_identity, separation_softmax, separation_square, separation_cube, separation_sqrt, separation_quartic, separation_ten, separation_max]
    sep_labels = ["Log", "Identity", "Softmax","Square","Cube","Sqrt", "Quartic","10th Order Polynomial","Max"]
    # do with fewer labels just to see if it works
    
    #Ns = [2,3,5,7,10,20,50,100,200,300]
    Ns = [5,7,10,20,50,100,200,300,500,700,1000]
    beta = 1
    sep_param = 1000
    N_runs = 5
    N_runs_separation_function_graphs(N_runs, Ns, imgs,  beta, sep_fns, sep_labels, load_data = LOAD_DATA, plot_results = PLOT_RESULTS,sname=dataset_str + "N_runs_separation_function_results_2.npy", figname= dataset_str+ "N_runs_separation_functions_2.jpg")
    
    
    # noise levels

    sigmas = [0.05,0.1,0.2,0.3,0.5,0.8,1,1.5]#,2]
    N = 100
    N_runs = 5
    fs = [euclidean_distance, manhatten_distance,normalized_dot_product,KL_divergence,reverse_KL_divergence,Jensen_Shannon_divergence]
    labels = ["Euclidean Distance","Manhattan Distance", "Dot Product", "KL divergence","Reverse KL","Jensen-Shannon"]
    beta = 1
    corrects_list = N_runs_noise_level_graphs(N_runs, N,imgs,beta,fs,labels,sigmas, load_data=LOAD_DATA,plot_results = PLOT_RESULTS,sname=dataset_str + "N_noise_level_results.npy", figname = dataset_str + "N_runs_noise_levels.jpg")
    print(corrects_list.shape)

    # frac masking

    mask_fracs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    N = 50
    N_runs =5
    fs = [euclidean_distance, manhatten_distance,normalized_dot_product,KL_divergence,reverse_KL_divergence,Jensen_Shannon_divergence]
    labels = ["Euclidean Distance","Manhattan Distance", "Dot Product", "KL Divergence","Reverse KL","Jensen-Shannon"]
    beta = 1
    mask_frac_corrects = N_runs_mask_frac_graphs(N_runs,N,imgs,beta,fs,labels,mask_fracs,load_data = LOAD_DATA,plot_results=PLOT_RESULTS,sname = dataset_str + "N_mask_frac_results.npy", figname = dataset_str + "N_runs_mask_fracs.jpg")
    
    
    # similarity functions
    #Ns = [2,5,10,20,50,100,200,300,500,700,1000]
    #longer mnist run
    #Ns = [1500,2000,2500,3000]
    # even longer mnist run
    Ns = [4000,5000,6000,7000,8000,9000,100000]
    N_runs = 5
    beta = 1000
    #fs = [euclidean_distance, manhatten_distance,normalized_dot_product]#,KL_divergence,reverse_KL_divergence,Jensen_Shannon_divergence]#,cosine_similarity]

    fs = [euclidean_distance, manhatten_distance,normalized_dot_product,KL_divergence,reverse_KL_divergence,Jensen_Shannon_divergence]#,cosine_similarity]
    labels = ["Euclidean Distance","Manhatten Distance", "Dot Product","KL Divergence","Reverse KL","Jensen-Shannon"]
    corrects_list2 = N_runs_capacity_graphs(N_runs, Ns, imgs, beta,fs,labels,image_perturb_fn = gaussian_perturb_image,sigma=0.5,load_data = LOAD_DATA,plot_results=PLOT_RESULTS,sname = dataset_str + "N_capacity_results.npy", figname = dataset_str + "N_runs_capacity_graph.jpg")
    


    sigmas = [0.05,0.1,0.2,0.3,0.5,0.8,1,1.5]
    mask_fracs  = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    generate_demonstration_reconstructions(imgs, 100, perturb_vals= sigmas)
    generate_demonstration_reconstructions(imgs, 50, perturb_vals= mask_fracs)

    visualize_heteroassociative_MCHN(imgs, 15,6)
    mnist_imgs = load_mnist(1000)[0][0]
    m_imgs = mnist_imgs[0].reshape(1000,28,28)
    print(m_imgs.shape)
    visualize_heteroassociative_hopfield(m_imgs, 5,5)
    
    
    # run the heteroassociative for MCHN
    
    N_runs = 5
    Ns = [2,5,10,20,50,100,200,300,500,700,1000]
    #Ns = [2,5,10]
    beta = 1000
    f = normalized_dot_product
    sep_fn = separation_max
    
    N_runs_auto_vs_heteroassociative(N_runs, Ns, imgs, beta,f, image_perturb_fn = gaussian_perturb_image, sep_fn = separation_max, sep_param = 1000, sigma=0.3,sname = dataset_str + "capacity_comparison_gaussian", figname = dataset_str + "auto_hetero_comparison_gaussian.jpg", load_data = False, plot_results = True, save_continuously=True)
    

    # heteroassociative for mnist classical hopfield
    
    N_runs = 5
    Ns = [2,5,10,20,30,50]#,70,100,150,200]
    #Ns = [2,5,10]
    beta = 1
    f = normalized_dot_product
    sep_fn = separation_identity
    
    N_runs_auto_vs_heteroassociative(N_runs, Ns, imgs, beta,f, image_perturb_fn = gaussian_perturb_image, sep_fn = sep_fn, sep_param = 1000, sigma=0.0000001,sname = dataset_str + "capacity_comparison_gaussian", figname = dataset_str + "auto_hetero_comparison_gaussian.jpg", load_data = False, plot_results = True, save_continuously=True,network_type = "classical_hopfield")
    

