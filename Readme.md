# Single-Image SVBRDF Capture with a Rendering-Aware Deep Network
![teaser](https://team.inria.fr/graphdeco/files/2018/08/teaser_v0.png)
This repository contains the code for our paper "Single-Image SVBRDF Capture with a Rendering-Aware Deep Network, Valentin Deschaintre, Miika Aittala, Fredo Durand, George Drettakis, Adrien Bousseau. Transactions on graphics (Siggraph Conference Proceedings),  aug 2018".

The project webpage can be found here: https://team.inria.fr/graphdeco/projects/deep-materials/

**The data for the pre-training can be found on the project webpage.**

## Paper abstract

Texture, highlights, and shading are some of many visual cues that allow humans to perceive material appearance in single pictures. Yet, recovering spatially-varying bi-directional reflectance distribution functions (SVBRDFs) from a single image based on such cues has challenged researchers in computer graphics for decades.
We tackle lightweight appearance capture by training a deep neural network to automatically extract and make sense of these visual cues. Once trained, our network is capable of recovering per-pixel normal, diffuse albedo, specular albedo and specular roughness from a single picture of a flat surface lit by a hand-held flash. We achieve this goal by introducing several innovations on training data acquisition and network design.
For training, we leverage a large dataset of artist-created, procedural SVBRDFs which we sample and render under multiple lighting directions. We further amplify the data by material mixing to cover a wide diversity of shading effects, which allows our network to work across many material classes. Motivated by the observation that distant regions of a material sample often offer complementary visual cues, we design a network that combines an encoder-decoder convolutional track for local feature extraction with a fully-connected track for global feature extraction and propagation.
Many important material effects are view-dependent, and as such ambiguous when observed in a single image. We tackle this challenge by defining the loss as a differentiable SVBRDF similarity metric that compares the renderings of the predicted maps against renderings of the ground truth from several lighting and viewing directions. Combined together, these novel ingredients bring clear improvement over state of the art methods for single-shot capture of spatially varying BRDFs.

## Software requirements
This code relies on Tensorflow 1.X but can be adapted to TF 2.X with the following compatibility code:

    Replace tensorflow import everywhere by:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

It is based on python 3.X, numpy, imageio and opencv for python.

## /!\Material model
This network is trained to use 256x256 linear input pictures (please use ---correctGamma if your input still has gamma correction, this option assumes gamma 2.2) and output linear parameters. Higher resolution tend to work less well despite the convolutional nature of the network (see https://team.inria.fr/graphdeco/projects/large-scale-materials/ supplemental materials). 

The model used is the one described in this paper (similar to Adobe Substance), **changing the rendering model implementation to render the results will cause strong appearance difference** as different implementations use the parameters differently (despite sharing their names, for example diffuse and specular will be controled for light conservation or roughness will be squared)! 

This method is based purely on the rendering loss, so you should be able to retrain the network using the same dataset but a different rendering loss material model implementation if you need.

## Re-training the network
To retrain the network, the basic version is: python3 material_net.py --mode train --output_dir $outputDir --input_dir $inputDir/trainBlended --batch_size 8 --loss render --useLog
You can find the training data (85GB) here: https://repo-sam.inria.fr/fungraph/deep-materials/DeepMaterialsData.zip

There are a lot of options to explore in the code if you are curious.

## Running the network inference
**First download the trained weights here: https://repo-sam.inria.fr/fungraph/deep-materials/InferenceCode_DeepMaterials.zip**

INPUT-OUTPUTS:
This code takes pictures of material taken with a cellphone (Fov ~ 45 °) and flash approximately in the middle (be careful not to entirerly burn the picture on very specular materials).
It outputs a set of 4 maps (diffuse, specular, roughness and normal) corresponding to a Cook-Torrance GGX implementation described in the paper (similar to the one of Adobe Substance for coherence).

PRE-REQUISITE(install):

Python (including numpy)
Tensorflow 1.X or 2.X (tested on 1.4, 1.12.1, 2.1.0)
/!\ Uncomment 2 lines at the very top of the code file to run this with TF 2.0+ /!\

Run the test folder:
python3 material_net.py --input_dir inputExamples/ --mode eval --output_dir examples_outputs --checkpoint . --imageFormat png --scale_size 256 --batch_size 1

HOW TO USE:

Run it as a python script in a command invite:

python material_net.py --input_dir $INPUT_DIR --mode eval --output_dir $OUTPUT_DIR --checkpoint $CHECKPOINT_LOCATION --imageFormat $YOURIMAGEFORMAT --scale_size $SIZEOFIMAGESIDE

These are the most interesting parameters. 


Here is a description of all useful parameters for inference :

--input_dir	help="path to xml file, folder or image (defined by --imageFormat) containing information images"

--mode	 required=True  choices=["test", "eval"])   help="Defines the mode of inference (test expect inputs with ground truth, eval expect single pictures )

--output_dir	 required=True, help="where to put output files"

--checkpoint	 required=True, default=None, help="directory with checkpoint to use for testing "

--testMode	 type=str, default="auto"	 choices=["auto", "xml", "folder", "image"], help="What kind of input should be used (auto should automatically determine)"

--imageFormat	 type=str, default="png	 choices=["jpg", "png", "jpeg", "JPG", "JPEG", "PNG"], help="Which format have the input files"

--batch_size	 type=int, default=1, help="number of images in batch to process parallely"

--scale_size	 type=int, default=288, help="scale images to this size before cropping to 256x256. Should be used carefully, it's best to use the actual size of your images here"

--logOutputAlbedos	"Log the output albedos diffuse and specular default is false, to use just add "--logOutputAlbedos""

## Bibtex
If you use our code, please cite our paper:

@Article{DADDB18,
  author       = "Deschaintre, Valentin and Aittala, Miika and Durand, Fr\'edo and Drettakis, George and Bousseau, Adrien",
  title        = "Single-Image SVBRDF Capture with a Rendering-Aware Deep Network",
  journal      = "ACM Transactions on Graphics (SIGGRAPH Conference Proceedings)",
  number       = "128",
  volume       = "37",
  pages        = "15",
  month        = "aug",
  year         = "2018",
  keywords     = "material capture, appearance capture, SVBRDF, deep learning",
  url          = "http://www-sop.inria.fr/reves/Basilic/2018/DADDB18"
}


