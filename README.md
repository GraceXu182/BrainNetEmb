# MEG_AD_project (AD progression prediction based on MEG brain networks)
Characterizing the subtle changes of functional
brain networks associated with the pathological cascade of
Alzheimer’s disease (AD) is important for early diagnosis and
prediction of disease progression prior to clinical symptoms. We
developed a new deep learning method, termed multiple graph
Gaussian embedding model, which can learn highly
informative network features by mapping high-dimensional
resting-state brain networks into a low-dimensional latent space.
These latent distribution-based embeddings enable a quantitative
characterization of subtle and heterogeneous brain connectivity
patterns at different regions, and can be used as input to
traditional classifiers for various downstream graph analytic
tasks, such as AD early stage prediction, and statistical evaluation
of between-group significant alterations across brain regions. We
used our proposed method to detect the intrinsic latent dimensionality of MEG
brain networks, predict the progression of patients with mild
cognitive impairment (MCI) to AD, and identify brain regions
with network alterations related to MCI.

![main workflow](Fig1.png)
Format: ![main workflow](https://github.com/GraceXu182/BrainNetEmb/Fig1.png)

## Experimental results
### AD progression prediction based on graph embedding features
![Supervised learning for AD progression prediction](Fig3.png)
Format: ![Supervised learning for AD progression prediction](https://github.com/GraceXu182/BrainNetEmb/Fig3.png)

### Brain regions with significant AD-related effects
![Brain regions with AD-related effects for NC vs. sMCI and sMCI vs. pMCI cases](Fig4.png)
Format: ![Brain regions with AD-related effects for NC vs. sMCI and sMCI vs. pMCI cases](https://github.com/GraceXu182/BrainNetEmb/Fig4.png)
![Brain regions with AD-related effects for NC vs. pMCI comparison](Suppl.Fig1.png)
Format: ![Brain regions with AD-related effects for NC vs. pMCI case](https://github.com/GraceXu182/BrainNetEmb/Suppl.Fig1.png)   
      

# Reference
If you use this repository, please cite our paper in the below.

>@article{xu2021graph,
  title={[A Graph Gaussian Embedding Method for Predicting Alzheimer's Disease Progression With MEG Brain Networks](https://ieeexplore.ieee.org/abstract/document/9314203)},
  author={Xu, Mengjia and Sanz, David Lopez and Garces, Pilar and Maestu, Fernando and Li, Quanzheng and Pantazis, Dimitrios},
  journal={IEEE Transactions on Biomedical Engineering},
  volume={68},
  number={5},
  pages={1579--1588},
  year={2021},
  publisher={IEEE}
}
