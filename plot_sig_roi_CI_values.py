"""
visualization 
@2019-11-16
@ XMJ

"""
import matplotlib.pyplot as plt

import numpy as np
import os
from scipy import stats
import pdb
import scipy.io as sio
import matplotlib.patches as patches
from g2g.utils2 import *
from g2g.meg_utils import *
from nilearn import datasets
from nilearn import  plotting
from visbrain.objects import (BrainObj, ColorbarObj, SceneObj, SourceObj, ConnectObj)
from visbrain.io import download_file

############################################
# *** Display brain network pairwisely *** 
############################################

def vis_brainNet(i, xyz, s_con, t_con):
    '''
    Visualization of meg brain network in 3D space

    @param i : subject index
    @param xyz: coordinate list of all brain regions in the atlas
    @param total_con: connectivity list of all subjects' brain connectivity matrices

    ## date: 2020-2-11 by Mengjia Xu

    '''
    sc = SceneObj(bgcolor='black')
    n_sources = xyz.shape[0]
    data = np.random.rand(n_sources)

    cnn_1 = np.triu(s_con[i])
    cnn_2 = np.triu(t_con[i])
    cnn_select_1 = (-.010 < cnn_1) & (cnn_1 < .010)
    cnn_select_2 = (-.010 < cnn_2) & (cnn_2 < .010)


    "normal control"
    
    tt = 'Subject %d (Month 0)'%i
    b_obj_1 = BrainObj('white')
    b_obj_1.animate(iterations=10, step=30, interval=1)
    s_obj_1 = SourceObj('s1', xyz, color = 'red', data=data, symbol='disc', radius_max=20)
    c_obj_1 = ConnectObj('c1', xyz, cnn_1, select=cnn_select_1, dynamic=(0., 1.),
                         dynamic_orientation='center', dynamic_order=3, cmap='bwr',
                         antialias=True)
    sc.add_to_subplot(b_obj_1, rotate = 'front', title= tt)   
    sc.add_to_subplot(s_obj_1) 
    # sc.add_to_subplot(c_obj_1)  

    "stable mci"
    
    tt = 'Subject %d (Month 3)'%i

    b_obj_2 = BrainObj('white')
    b_obj_2.animate(iterations=10, step=30, interval=1)
    s_obj_2 = SourceObj('s2', xyz, color = 'red', data=data, symbol='disc', radius_max=20)
    c_obj_2 = ConnectObj('c2', xyz, cnn_2, select=cnn_select_2, dynamic=(0., 1.),
                         dynamic_orientation='center', dynamic_order=3, cmap='bwr',
                         antialias=True)

    sc.add_to_subplot(b_obj_2, col = 1,use_this_cam = True,title= tt)
    sc.add_to_subplot(s_obj_2, col = 1)
    sc.add_to_subplot(c_obj_2, col = 1)
    # sc.record_animation('animate_example_sub%d.gif'%i, n_pic=10)

    sc.preview()

def plot_meg_connectome():
    '''
    2020-2-11 by mengjia xu
    Plot the MEG brain connectome for the master figure in MEG paper

    '''
    megdata= sio.loadmat('MEG_data_info/total_connecivtiy_coord_label.mat')
    total_con = megdata['connectivity']
    xyz = megdata['ROI_coords']
    normal_con = total_con[:53]
    smci_con = total_con[53:-28]
    pmci_con = total_con[-28:]
    edges = smci_con[8,:,:]

    sc = SceneObj(bgcolor='black')
    c_obj = ConnectObj('default', xyz, edges, select= edges>.026, line_width=3.,dynamic=(0., 1.),
                            dynamic_orientation='center', cmap='bwr', color_by='strength')
    s_obj = SourceObj('sources', xyz, color='red', radius_min=15.)
    cb_obj = ColorbarObj(c_obj, cblabel='Edge strength')

    sc.add_to_subplot(c_obj, title='MEG brain network')
    sc.add_to_subplot(s_obj)
    sc.add_to_subplot(BrainObj('B3'),use_this_cam=True)
    sc.add_to_subplot(cb_obj, col=1, width_max=200)
    sc.preview()

    # vis_brainNet(0, xyz, normal_con, smci_con)
#############################################
# 2019-11-16 by Mengjia Xu
# visualization regional variables on 3D plot
# (p value, node degree)
#############################################

def get_log_pval(padj, basis=None):
    '''
    compute the corresponding log values of padj values

    '''
    if basis == None:
        log_p = np.log(padj)
    elif basis == 2:
        log_p = np.log2(padj)
    else:
        log_p = np.log10(padj)
    max_p = np.max(log_p) 
    min_p = np.min(log_p) 
    return log_p, max_p, min_p

def vis_sources_by_pval(padj,all_coord,short_label):
    ''' plot the source objects along with the brain object'''
    # Define the default camera state used for each subplot
    CAM_STATE = dict(azimuth=0,        # azimuth angle
                     elevation=90,     # elevation angle
                     scale_factor=180  # distance to the camera
                     )
    S_KW = dict(camera_state=CAM_STATE)
    # Create the scene
    CBAR_STATE = dict(cbtxtsz=12, txtsz=13., width=.5, cbtxtsh=2.,
                      rect=(1., -2., 1., 4.))
    sc = SceneObj(bgcolor='black', size=(1600, 1000))

    # mask = padj<=0.05  # significant regions with true label
    
    data0, max_p, min_p=get_log_pval(padj)

    b_obj = BrainObj('white', hemisphere='both', translucent=True)
    b_obj.animate(iterations=10, step=30, interval=1)
    s_obj = SourceObj('s1', all_coord,   #all_coord, 
                        # mask = mask,
                        # mask_color='white',
                        # mask_radius = 15,
                        color = padj*100, 
                        data=data0, 
                        text= short_label,text_size=10, text_bold=True, text_color='yellow',
                        symbol='disc', 
                        visible = True, 
                        radius_min = 10,
                        radius_max = 40,
                        alpha=0.65
                        )

    s_obj_1.set_visible_sources('left')

    s_obj.color_sources(data=data0, cmap='jet',clim=(min_p,max_p))
    cb_data = ColorbarObj(s_obj, cblabel='Log FDR-adjusted p value', border=False, **CBAR_STATE)

    sc.add_to_subplot(b_obj, row=0, col=0, rotate = 'front')   
    sc.add_to_subplot(s_obj,row=0, col=0)
    sc.add_to_subplot(cb_data,row=0, col=1, width_max=90)
    
    # sc.record_animation('output/animate_pvalue_projection.gif', n_pic=10)
    sc.preview()

def vis_brainSurface_by_padj(padj_parcellates, gci_parcellates, tt, ptype='significant',cmap='jet',use_log = False, use_1_p=False, use_p=True):

        # Define the default camera state used for each subplot
        CAM_STATE = dict(azimuth=0,        # azimuth angle
                         elevation=90,     # elevation angle
                         scale_factor=180  # distance to the camera
                         )
        S_KW = dict(camera_state=CAM_STATE)
        # Create the scene
        CBAR_STATE = dict(cbtxtsz=12, txtsz=13., width=.5, cbtxtsh=2.,
                          rect=(1., -2., 1., 4.))
        sc = SceneObj(bgcolor='black', size=(1300, 1000))
        # n_sources = all_coord.shape[0]  #sig_roi_coord.shape[0]
        # data = np.random.rand(n_sources)
        
        # 1. significant regions
        b_obj_1 = BrainObj('white', translucent=False)

        # parcellize brain based on desikan atlas
        path_to_file1 = download_file('lh.aparc.annot', astype='example_data')
        path_to_file2 = download_file('rh.aparc.annot', astype='example_data')

        #  dataframe type varaible inlcuding index, label, color for each brain region in DKT atlas
        lh_df = b_obj_1.get_parcellates(path_to_file1)
        rh_df = b_obj_1.get_parcellates(path_to_file2)
        # lh_df.to_excel('output/lh.aparc.xlsx')


        def select_rois(df,row=1, col=1):
            select_val = list(np.array(df.iloc[row:,col]))
            select_val.pop(3)
            return select_val

        select_roi1 = select_rois(lh_df)
        select_roi2 = select_rois(rh_df)

        #  get log FDR-adjusted p values
        # log_p_parce, l_p, s_p = get_log_pval(padj_parcellates, basis=2)

        l_p = np.max(gci_parcellates)
        s_p = np.min(gci_parcellates)
        print('-----#####',l_p,s_p)



        if ptype == 'significant' or ptype == 'unsignificant' :

            def get_hemisphere_rois(lh_padj, lh_gci, lh_rois,select='significant',cal_log=True):
                if select =='significant':
                    lh_sig_ind = np.where(np.array(lh_padj)<=0.05)[0]
                else:
                    lh_sig_ind = np.where(np.array(lh_padj)>0.05)[0]

                lh_sig_rois = [lh_rois[i] for i in lh_sig_ind]

                if cal_log:
                    log_p,_,_ = get_log_pval(lh_padj,basis=2)   # calculate "log2(padj)"
                    lh_sig_padj = list(np.array(log_p)[lh_sig_ind])
                else: 
                    # lh_sig_padj = list(np.array(lh_padj)[lh_sig_ind])  
                    lh_sig_gci = list(np.array(lh_gci)[lh_sig_ind])  

                
                # max_p= np.max(np.array(lh_sig_padj))
                # min_p= np.min(np.array(lh_sig_padj))
                # return lh_sig_rois,lh_sig_padj,max_p,min_p

                max_gci= np.max(np.array(lh_sig_gci))
                min_gci= np.min(np.array(lh_sig_gci))
                
                return lh_sig_rois,lh_sig_gci,max_gci,min_gci

            # (1). set (log-padj) as values for color mapping

            # select_regions_L,lh_padj,_,_ = get_hemisphere_rois(padj_parcellates[:34], gci_parcellates[:34], select_roi1, select = ptype, cal_log=use_log)
            # select_regions_R,rh_padj,_,_ = get_hemisphere_rois(padj_parcellates[34:], gci_parcellates[34:], select_roi2, select = ptype, cal_log=use_log)
            
            # clab = 'Log FDR-adjusted p value'
            # b_obj_1.parcellize(path_to_file1, select= select_regions_L,data=lh_padj,cmap=cmap,clim=[s_p,l_p])
            # b_obj_1.parcellize(path_to_file2, select= select_regions_R,data=rh_padj,cmap=cmap,clim=[s_p,l_p])
            # cb_1 = ColorbarObj(b_obj_1, clim= [s_p,l_p], cblabel=clab, border=False, **CBAR_STATE)
            
            # plot GCI value-4/23/2020

            select_regions_L,lh_gci,_,_ = get_hemisphere_rois(padj_parcellates[:34], gci_parcellates[:34], select_roi1, select = ptype, cal_log=use_log)
            select_regions_R,rh_gci,_,_ = get_hemisphere_rois(padj_parcellates[34:], gci_parcellates[34:], select_roi2, select = ptype, cal_log=use_log)
            
            clab = 'GCI value'
            b_obj_1.parcellize(path_to_file1, select= select_regions_L,data=lh_gci)#clim=[s_p,l_p]
            b_obj_1.parcellize(path_to_file2, select= select_regions_R,data=rh_gci,cmap=cmap)#, clim=[s_p,l_p])
            cb_1 = ColorbarObj(b_obj_1, clim= [1.76,1.80], cblabel=clab, border=False, **CBAR_STATE)
         
        elif ptype == 'together':
            select_regions_L = select_roi1
            select_regions_R = select_roi2
            if use_log:
                # (1). set (log-padj) as values for color mapping
                clab = 'Log FDR-adjusted p value'
                lh_padj = log_p_parce[:34]
                rh_padj = log_p_parce[34:]
                b_obj_1.parcellize(path_to_file1, select= select_regions_L,data=lh_padj,cmap=cmap,clim=[s_p,l_p])
                b_obj_1.parcellize(path_to_file2, select= select_regions_R,data=rh_padj,cmap=cmap,clim=[s_p,l_p])
                cb_1 = ColorbarObj(b_obj_1, clim= [s_p,l_p], cblabel=clab, border=False, **CBAR_STATE)
            if use_1_p:

                # (2). set (1-padj) as values for color mapping
                clab = tt #'1-FDR-adjusted p value'
                # clab = '1-FDR-adjusted p value'
                padj_0 = [1-i for i in padj_parcellates]
                b_obj_1.parcellize(path_to_file1,select= select_regions_L,data=padj_0[:34],cmap=cmap)
                b_obj_1.parcellize(path_to_file2,select= select_regions_R,data=padj_0[34:],cmap=cmap)
                cb_1 = ColorbarObj(b_obj_1, cblabel=clab, border=False, **CBAR_STATE)  #Log FDR-adjusted p value

            if use_p:
                # (2). set (1-padj) as values for color mapping
                print('--------use p-------')
                clab = tt #'1-FDR-adjusted p value'
                mx =  np.array(gci_parcellates).max()
                mi =  np.array(gci_parcellates).min()
                b_obj_1.parcellize(path_to_file1,select= select_regions_L,data=gci_parcellates[:34],cmap=cmap,clim=[mi,mx])
                b_obj_1.parcellize(path_to_file2,select= select_regions_R,data=gci_parcellates[34:],cmap=cmap,clim=[mi,mx])
                cb_1 = ColorbarObj(b_obj_1, cblabel=clab,  border=False, **CBAR_STATE)  #Log FDR-adjusted p value
        
        b_obj_1.animate(iterations=10, step=30, interval=1.2)
        sc.add_to_subplot(b_obj_1, row=0, col=0,rotate='front')
        sc.add_to_subplot(cb_1, row=0, col=1, width_max=90) 
        # sc.record_animation('output/%s_pvalue_projection.gif'%ptype, n_pic=10)
        # sc.record_animation('output/pmci_degree.gif', n_pic=8)


        sc.preview()

def main():

    # data = sio.loadmat('output/meg_atlas_info_SP_20200110_v2.mat')
    # data = sio.loadmat('output/meg_atlas_info_NS_v2.mat')

    # add on 10/26/2020 for TBME revision
    data = sio.loadmat('output/meg_atlas_info_NP_v2.mat')


    # all 68 brain regions' information (coord, full name, short name)
    # full_label = data['full_label']
    full_label = load_MEG_info(os.path.abspath('MEG_data_info/ROI_label_2.xlsx'))
    short_label = data['short_label']
    all_coord = data['coord']
    padj = data['padj_u'][0]
    padj_parcellates = data['padj_parcellates'][0]
    gci_parcellates = data['gci_parcellates'][0] #  GCI value---4/23/2020


    meg_lobe = data['lobe']

    h_label,_,_,_,_ = get_visbrain_label_color_index(full_label)


    # # 1. significant brain region information

    sig_roi_coord = data['sig_roi_coord']
    sig_roi_llabel = data['sig_roi_llabel']
    sig_roi_slabel = data['sig_roi_slabel']
    sig_roi_ind = data['sig_roi_index'][0]
    sig_padj = data['sig_padj'] 

    all_roi_ind =[i for i in range(68)]

    #2. unsignificant brain region information
    unsig_roi_ind = np.array(list(set(all_roi_ind)-set(list(sig_roi_ind))))
    unsig_roi_coord = all_coord[unsig_roi_ind]
    unsig_roi_llabel = np.array(full_label)[unsig_roi_ind]
    unsig_roi_slabel = short_label[unsig_roi_ind]

    '1 display brain surface based on padj value'
    # vis_brainSurface_by_padj(padj_parcellates, 'Log FDR-adjusted p value', ptype='significant',cmap='jet',use_log=True, use_1_p=False,use_p=False)
    # vis_brainSurface_by_padj(padj_parcellates, 'Log FDR-adjusted p value', ptype='unsignificant',cmap='jet',use_log=True, use_1_p=False,use_p=False)
    
    '2 display brain surface based on GCI value'
    vis_brainSurface_by_padj(padj_parcellates, gci_parcellates,'Log FDR-adjusted p value', ptype='significant',cmap='jet',use_log=False, use_1_p=False,use_p=True)
    # vis_brainSurface_by_padj(padj_parcellates, gci_parcellates,'GCI value', ptype='together',cmap='jet',use_log=False, use_1_p=False,use_p=True)
    
    '3. display brain ROIs based on padj value'

    # vis_sources_by_pval(padj,all_coord,short_label) 
    # vis_sources_by_pval(sig_padj,sig_roi_coord,sig_roi_slabel)  



    #-----visualize node feature on Brain surface----#       

    # feat_name = ['Degree','Ecentrality','Bcentrality']

    # feature = sio.loadmat('output/3c_mean_%s.mat'%feat_name[0])
    # nfea = feature['nc_mean_degree'][0]
    # sfea = feature['smci_mean_degree'][0]
    # pfea = feature['pmci_mean_degree'][0]

    # n_dg_parcellates = get_parcellates_padj(h_label, full_label, nfea)
    # s_dg_parcellates = get_parcellates_padj(h_label, full_label, sfea)
    # p_dg_parcellates = get_parcellates_padj(h_label, full_label, pfea)

    # diff = [s_dg_parcellates[i]-n_dg_parcellates[i] for i in range(len(s_dg_parcellates))]

    # vis_brainSurface_by_padj(n_dg_parcellates, feat_name[0], ptype='together',cmap='jet')
    # vis_brainSurface_by_padj(diff, feat_name[0],  ptype='together',cmap='jet')
    # vis_brainSurface_by_padj(p_dg_parcellates, feat_name[0], ptype='together',cmap='jet')
    
    # node_EC = sio.loadmat('output/3c_mean_%s.mat'%feat_name[1])
    # node_BC = sio.loadmat('output/3c_mean_%s.mat'%feat_name[2])
    # pdb.set_trace()

    





############################################
# " *** Display Brain regions *** "
############################################
if __name__ == '__main__':
    main()
    # plot_meg_connectome()





