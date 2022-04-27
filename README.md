# CGL Detection


<p align="justify"> Most of the visual scene understanding tasks in the field of computer vision involve identification of the objects present in the scene. Image regions like hideouts, turns, and other obscured regions of the scene also contain crucial information, for specific surveillance tasks. In this work, we propose an intelligent visual aid for identification of such locations in an image, which has either the potential to create an imminent threat from an adversary or appear as the target zones needing further investigation to identify concealed objects. Covert places (CGL) for hiding behind an occluding object are concealed 3D locations, not usually detectable from the viewpoint (camera). Hence this involves delineating specific image regions around the outer boundary of the projections of the occluding objects, as places to be accessed around the potential hideouts.  CGL detection finds applications in military counter-insurgency operations, surveillance with path planning for an exploratory robot. Given an RGB image, the goal is to identify all CGLs in the 2D scene. Identification of such regions would require knowledge about the 3D boundaries of obscuring items (pillars, furniture), their spatial location with respect to the background, and other neighboring regions of the scene. We propose this as a novel task, termed Covert Geo-Location (CGL) Detection. Classification of any region of an image as a CGL (as boundary sub-segments of an occluding object, concealing the hideout) requires examining the relation with its surroundings. CGL detection would thus require understanding of the 3D spatial relationships between boundaries of occluding objects and their neighborhoods. Our method successfully extracts relevant depth features from a single RGB image and quantitatively yields significant improvement over existing object detection and segmentation models adapted and trained for CGL detection. We also introduce a novel hand-annotated CGL detection dataset containing ~1.5K real-world images for experimentation. </p>

Link to [ArXiv version of our CGL Detection Paper](https://arxiv.org/abs/2202.02567)

# Installation & Preparation

## Prerequisites
- Python 3.6
- Pytorch 1.10.2
- Cuda 10.2

## Installation

Clone the repo and make new virtual enviornment by using [environment.yml][1]

[1]:https://github.com/VPLabIMPRINT/CGL_DETECTION/blob/main/environment.yml

```sh
# Clone this repository
git clone https://github.com/VPLabIMPRINT/CGL_DETECTION.git

# Go into the repository
cd CGL_DETECTION

# Deactivating previous environment
conda deactivate

# Creating environment from environment.yml file
conda env create --file environment.yml 

# Activating new enviornment
conda activate CGL 

```

## Running Demo
To run the real-time demo use the following command:
```sh
python3 -u CGL_test_realtime.py --cfg config/cglncgl_ade20k-hrnetv2_medium.yaml
```

## Note
For any installation issue, refer:
https://github.com/CSAILVision/semantic-segmentation-pytorch

## References

If you find the code or pre-trained models useful, please cite the following papers:

Saha, Binoy, and Sukhendu Das. "Catch Me if You Can: A Novel Task for Detection of Covert Geo-Locations (CGL)." arXiv preprint arXiv:2202.02567 (2022). [paper has been accepted for publication at ICVGIP'21 WCVA]

```
@article{saha2022catch,
  title={Catch Me if You Can: A Novel Task for Detection of Covert Geo-Locations (CGL)},
  author={Saha, Binoy and Das, Sukhendu},
  journal={arXiv preprint arXiv:2202.02567},
  year={2022}
}
```

Semantic Understanding of Scenes through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, T. Xiao, S. Fidler, A. Barriuso and A. Torralba. International Journal on Computer Vision (IJCV), 2018. (https://arxiv.org/pdf/1608.05442.pdf)

```
@article{zhou2018semantic,
  title={Semantic understanding of scenes through the ade20k dataset},
  author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Xiao, Tete and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
  journal={International Journal on Computer Vision},
  year={2018}
}
```
Scene Parsing through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso and A. Torralba. Computer Vision and Pattern Recognition (CVPR), 2017. (http://people.csail.mit.edu/bzhou/publication/scene-parse-camera-ready.pdf)

```
@inproceedings{zhou2017scene,
    title={Scene Parsing through ADE20K Dataset},
    author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    year={2017}
}
```












[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
