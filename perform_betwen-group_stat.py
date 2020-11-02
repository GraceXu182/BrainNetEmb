'''
Code for statistical analysis for between-group MEG Gaussian embedding results
(i.e., NC/sMCI and NC/pMCI--new add on 10/26/2020)
@ MJX
'''
import random
import scipy.io as sio
import numpy as np
import time

def W2_distance_2(mu0, mu3, sig0, sig3, L = 8):
    '''
    Compute W2 distance for two ROIs' gaussian embeddings
    Input:
        L  —— embedding size
        mu0,mu3 —— embedding mean array for each ROI with embedding size L, (L is embedding size)
        sig0,sig3 —— embedding variance array-like of N*L (~)
    Output:
        w2_distance —— 2nd wasserstein distance

    '''
    cov0 = np.eye(L)*sig0
    cov3 = np.eye(L)*sig3
    mean_L2_norm = np.linalg.norm(np.subtract(mu3,mu0),2)
    sigma_F_norm = np.linalg.norm(np.sqrt(cov0)+np.sqrt(cov3))
    distance = np.sqrt(np.square(mean_L2_norm)+np.square(sigma_F_norm))
    return distance

def get_inter_W2(con,sig,con3,sig3,roi):
    # 'between-pair W2 distance computation'

    inter_W2 =np.zeros([len(con),len(con3)])
    for i in range(0,len(con)):
        for j in range(0, len(con3)):             
            was_2 = W2_distance_2(con[i][roi,:],con3[j][roi,:],sig[i][roi,:],sig3[j][roi,:])
            inter_W2[i,j] = round(was_2,3)   
    return np.array(inter_W2)

def get_triangle(w2,c1,c2):
    'get lower triangle elements (discard the diagonal elements) from the complete W2 distance array'
    l_t = np.tril(w2)
    np.fill_diagonal(l_t, 0)
    NN = l_t[:c1,:c1]
    SN = l_t[c1:,0:c1]
    SS = l_t[-c2:,-c2:]
    return NN,SS,SN

def run_stat_MEG(mu_var,sig_var,c1,c2,s,t,n_perm=1000):
    # compute GCI value for ROIs in the index range [s,t], which is designed for efficient MPI running
    # to improve the computational efficiency
    
    GCI = np.zeros([t-s,n_perm])  # d value 2D array: n_roi* n_perm
    sub_index = list(np.arange(c1 + c2))
    for roi in range(s,t): 
        print('\n--<ROI %d>--'%roi)
        roi_dist=[]
        rd_append = roi_dist.append
        t1= time.time()
        for i in range(n_perm):
            print('permute time: %d'%i)
            if i == 0:
                multi_mu=mu_var
                multi_sig=sig_var

            else:
                # shuffle all classes' indices
                shuffle_index = random.sample(sub_index,len(sub_index))

                # take the first c1 subjects as class 1; remaining ones as class 2.
                g1_ind = shuffle_index[:c1]
                g2_ind = shuffle_index[c1:] 
                mu1 = [mu_var[i] for i in g1_ind]
                sig1 = [sig_var[i] for i in g1_ind]
                mu2 = [mu_var[i] for i in g2_ind]
                sig2 = [sig_var[i] for i in g2_ind]

                multi_mu = mu1+mu2
                multi_sig = sig1+sig2

            w2 = get_inter_W2(multi_mu,multi_sig,multi_mu,multi_sig, roi)
            #     plt.figure(dpi=100)
            #     plt.imshow(w2,cmap='jet')
            #     plt.title('Permutation %d'%i)
            #     plt.colorbar()
            NN,SS,SN = get_triangle(w2,c1,c2)

            # compute averaged d value for each ROI in each permutation;
            d = np.mean(SN)-(np.mean(NN)/2+np.mean(SS)/2)
            rd_append(round(d,3))
        np.save('output/dvalues_NP_pmt1000/total_dis_roi%d.npy'%roi,roi_dist)
        t3 = time.time()
        print('------cost time : %.3f'%(t3-t1))
        
        DD[roi,:]=roi_dist
    return DD

###############################################################
# load embedding results for control, smci and pmci groups
###############################################################
emb3C = sio.loadmat('output/total_emb_3c.mat')

c1 = 53 # normal quantity
c2 = 48 # smci quantity
c3 = 28 # pmci quantity

# mean vectors
nc_mu = list(emb3C['mu'])[:c1]
smci_mu = list(emb3C['mu'])[c1:-c3]
pmci_mu = list(emb3C['mu'])[-c3:]

# variance vectors
nc_sig = list(emb3C['sig'])[:c1]
smci_sig = list(emb3C['sig'])[c1:-c3]
pmci_sig = list(emb3C['sig'])[-c3:]

###############################################################
# Compute permutated W2 between class 'normal' and class 'smci'
###############################################################
mu_var = nc_mu + smci_mu
sig_var = nc_sig + smci_sig
run_stat_MEG(mu_var,sig_var,c1,c2,0,1,n_perm=1000)

###############################################################
# Compute permutated W2 between class 'smci' and class 'pmci'
###############################################################
mu_var2 = smci_mu + pmci_mu
sig_var2 = smci_sig + pmci_sig
run_stat_MEG(mu_var2,sig_var2,c2,c3,1,2,n_perm = 1000)
