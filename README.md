# SSNet

> **Abstract:** Spacecraft Inverse Synthetic Aperture Radar (ISAR) imaging super-resolution aims to enhance the resolution of low-resolution images to produce high-resolution images. However, spacecraft ISAR imaging presents challenges such as sparse, fuzzy boundaries, and the intricate differentiation between background and spacecraft, rendering current methods less effective in achieving satisfactory super-resolution results. In this letter, we propose a sparse and selective feature fusion network for super-resolving spacecraft ISAR images. At the heart of our approach lies a multi-scale residual block featuring the following essential components: (a) parallel multi-resolution streams to extract multi-scale features, (b) trainable top-k selection operator that intelligently retains the most critical attention scores from the keys for each query, enhancing the distinction between background and spacecraft information within the local region, and (c) selective cross attention to discriminatively determine which low and high-frequency information to retain when aggregating multi-scale features. The resulting tightly interlinked architecture, named as SSNet, learns a set of more accurate features. Extensive experiments conducted on actual ISAR images of  spacecraft  clearly demonstrate that the proposed method outperforms state-of-the-art approaches, delivering impressive performance.


### Download the [model](https://drive.google.com/file/d/1mNsNor3rb7JuQIzJPKUuRL7ubW6OVeAS/view?usp=sharing) 

 ## Citations
If our code helps your research or work, please consider citing our paper.
The following is a BibTeX reference:

```
@ARTICLE{10536725,
  author={Gao, Hu and Yang, Jingfan and Wang, Ning and Yang, Jing and Zhang, Ying and Dang, Depeng},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Learning Accurate Features for Super-Resolution Spacecraft ISAR Imaging}, 
  year={2024},
}

'''

## Contact
Should you have any question, please contact two_bits@163.com
