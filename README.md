<h1 align="center"  >Face Mask Detection Using Machine Learning</h1>

<p align="center">
  <img src="https://5.imimg.com/data5/PI/FD/NK/SELLER-5866466/images-500x500.jpg" alt="Sublime's custom image"/>
</p>


 ## TABLE OF CONTENTS
 
- Abstract 
- About Project
- Research Questions
- Prerequisites
- Dataset
- Roadmap
- Preprocessing
- Results and Discussion
- Refrences

 ## ABSTRACT
As we know that Covid-19 is almost ended and majority of the countries have removed the restriction to wear mask at every places.After ending the pandemic what is the use of this model everyone think that. Nowadays ,indusrilisation have changed the world at global level in each and every field such as healthcare , chemical ,technology , and many more. Visit to the madicap company ,which is making various medicines.The idea came from that visit that surgical mask is mandatory at that place becuase their employers deal with various substance which will affect their body in bad way if that substance directly goes to their body.What is more, they have to keep security person to check each and every person that wearing mask or not? Creating the autonomus model using the various machine learning techniques will reduce the burden of doing manual work and considering the health of their workers will prove best solution for them.


 ## RESEARCH QUESTIONS
 - Person is wearing mask or not?
<img width="234" alt="image" src="https://static.vecteezy.com/system/resources/thumbnails/002/311/476/small/no-entry-without-a-face-mask-wear-face-mask-right-and-wrong-wearing-a-mask-illustration-in-a-flat-style-vector.jpg">

 - Which kind of mask is person is wearing from 5 categories?
 <img width="234" alt="image" src="https://user-images.githubusercontent.com/83024113/181513129-a6b4546f-6709-4ff2-a37d-41e3308fb334.png">

 
 ## ABOUT PROJECT
This project that I created will present dataset to detect the face masks.Collected the data from various sources with different mask catergories.Dataset is used to train and test the machine learning model.there are certain images in our dataset from various backgorund , image from front face and side by side.



 ## PREREQUISITES
 >Software
  #### You can use any of this to run the code:

- Pycharm
- Anaconda
- Visual Studio Code
- You can run on your google collaboratory

>Libraries
#### All listed library must be installed using -> ***!pip install (library name)***  in command line.
Jupyter notebook
- torch
- numpy
- matplotlib
- pandas

## DATASET

Dataset is maily collected from google images and kaggle dataset according to requriment of my model.
>Their are five categories in dataset.
- Cloth mask
- Without mask
- Surgical mask
- N95 mask
- N95 mask with valve

<img width="300" alt="image" src="https://user-images.githubusercontent.com/83024113/181511386-58fed343-cb19-4141-9074-7e913461c756.png">

I have used ImageLoader for file and image reading, so if you want to run the notebook you have to change the file location accordingly. You can do same on google colab to by uploading and mounting files on drive.

>Dataset is divided in to 2 parts Test and Train:

![image](https://user-images.githubusercontent.com/83024113/181541123-7c63d956-d4e5-49a2-8f3e-71ba945b8846.png)



## ROADMAP
```mermaid
flowchart TD
    A[IMPORTING_LIBRARIES]-->B[TRANSFORMATION];
    B[TRANSFORMATION]-->C[LOAD DATASET];
    C[LOAD_DATASET]-->D[DISPLAY_A_BATCH_OF_IMAGES];
    D[DISPLAY_A_BATCH_OF_IMAGES]-->E[DEFINE_MODEL];
    E[DEFINE_MODEL]-->F[TRAIN_MODEL];
    F[TRAIN_MODEL]-->G[TEST_ACCURACY];
    G[TEST_ACCURACY]-->H[DOWNLOAD_MODEL];
```
## PREPROCESSING
I have used convolution neural netowork to define model and test the model.
A CNN is also known as a Convolution Neural Network(CNN), consisting of several different network architectures such as convolution, pooling, normalization, and Perceptron. Each one has different characteristics. 
1. Convolution Layer
2. Normlization Layer
3. Perceptron Network
4. Loss function
5. Optimizer

>[HOW CONVOLUTION NEURAL NETWORK WORKS](https://github.com/gatisonani/FaceMasKDetection/blob/master/How_Convolution_Neural_Network_works.pdf).

## RESULT AND DISCUSSION

<img width="314" alt="image" src="https://user-images.githubusercontent.com/83024113/181536789-30f0cd50-a5bc-453b-8030-fc610f218e95.png">
<img width="172" alt="image" src="https://user-images.githubusercontent.com/83024113/181537128-5346cb85-26ec-4a99-abc3-d7722cae40b2.png">

	Predicted Label







       
       
       
 
