# Streamlit Facial Attribute Modifier App

Interactive Web App that performs facial attributes modifications in front face images.

![demo_gif](./resources/gifs/demo.gif)

## Description

This app makes use of [STGAN](https://arxiv.org/abs/1904.09709v1) which is based in [AttGAN](https://arxiv.org/pdf/1711.10678v1.pdf)
        

## Dependencies

If you're a conda user, you can create an environment from the ```environment.yml``` file using the Terminal or an Anaconda Prompt for the following steps:

1. Create the environment from the ```environment.yml``` file:

    ```conda env create -f environment.yml```
2. Activate the new environment:
    * Windows: ```activate stgan```
    * macOS and Linux: ```source activate stgan``` 

3. Verify that the new environment was installed correctly:

    ```conda list```
    
You can also clone the environment through the environment manager of Anaconda Navigator.

It's mandatory to download the pretrained model from [Google Drive](https://drive.google.com/open?id=1329IbLE6877DcDUut1reKxckijBJye7N) or [Baidu Cloud (4qeu)](https://pan.baidu.com/s/1D43d_8oER8_Xm4P9SovvuQ) and unzip the files to the *model/* directory. 

## Use

Within the virtual environment:

```streamlit run app.py```

A web application will open in the prompted URL. The *Options* panel will appear at the left sidebar. First of all, you'll need to specify which of the images located in *input_images/* is going to be processed. The model is fed with 128x128 px images, so select images that already have this kind of aspect ratio. Furthermore, the better ilumination, centered and visible the face is within the picture, the better results will the model output. Several images from the [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) are provided to quicly see some results.  
A *Save button* is also available to store the output image in the *output_images/* folder. 

## Acknowledgments

* [STGAN](https://github.com/csmliu/STGAN)
* [AttGAN](https://github.com/LynnHo/AttGAN-Tensorflow)
* [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
