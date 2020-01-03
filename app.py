# -----------------------------------------------------------------------------
# Streamlit app that selects front face images and allows the user to modify
# them with interactive widgets. The GUI is displayed in a web browser. 
#
# Model details: STGAN (https://arxiv.org/abs/1904.09709v1)
# -----------------------------------------------------------------------------
import argparse
from functools import partial
import json
import numpy as np
import tensorflow as tf
import tflib as tl
import models
from pathlib import Path
import streamlit as st
import sys
import os
import cv2

IMG_DISPLAY_HEIGHT = 360
IMG_DISPLAY_WIDTH = 360

ATTRIBUTES = ['Bald', 'Bangs', 'Black Hair', 'Blond Hair', 'Brown Hair',
    'Bushy Eyebrows', 'Eyeglasses', 'Male', 'Mouth Slightly Open',
    'Mustache', 'No Beard', 'Pale Skin', 'Young']


def model_initialization():
    """Model creation and weight load.
    
    Load of several parameters found in the pretrained STGAN model: 
    https://drive.google.com/open?id=1329IbLE6877DcDUut1reKxckijBJye7N.

    Returns:
        sess (TF Session): Current session for inference.
        x_sample (tfTensor): Tensor of shape (n_img, 128, 128, 3).
        xa_sample (tfTensor): Input tensor of shape (n_img, 128, 128, 3).
        _b_sample (tfTensor): Label tensor of shape (n_img, 13).
        raw_b_sample (tfTensor): Label tensor of shape (n_img, 13).

    """
    with open('./model/setting.txt') as f:
        args = json.load(f)

    atts = args['atts']
    n_atts = len(atts)
    img_size = args['img_size']
    shortcut_layers = args['shortcut_layers']
    inject_layers = args['inject_layers']
    enc_dim = args['enc_dim']
    dec_dim = args['dec_dim']
    dis_dim = args['dis_dim']
    dis_fc_dim = args['dis_fc_dim']
    enc_layers = args['enc_layers']
    dec_layers = args['dec_layers']
    dis_layers = args['dis_layers']

    label = args['label']
    use_stu = args['use_stu']
    stu_dim = args['stu_dim']
    stu_layers = args['stu_layers']
    stu_inject_layers = args['stu_inject_layers']
    stu_kernel_size = args['stu_kernel_size']
    stu_norm = args['stu_norm']
    stu_state = args['stu_state']
    multi_inputs = args['multi_inputs']
    rec_loss_weight = args['rec_loss_weight']
    one_more_conv = args['one_more_conv']

    sess = tl.session()
    # Models
    Genc = partial(models.Genc, dim=enc_dim, n_layers=enc_layers, 
        multi_inputs=multi_inputs)
    Gdec = partial(models.Gdec, dim=dec_dim, n_layers=dec_layers, 
        shortcut_layers=shortcut_layers, inject_layers=inject_layers, 
        one_more_conv=one_more_conv)
    Gstu = partial(models.Gstu, dim=stu_dim, n_layers=stu_layers, 
        inject_layers=stu_inject_layers, kernel_size=stu_kernel_size, 
        norm=stu_norm, pass_state=stu_state)

    # Inputs
    xa_sample = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3])
    _b_sample = tf.placeholder(tf.float32, shape=[None, n_atts])
    raw_b_sample = tf.placeholder(tf.float32, shape=[None, n_atts])

    # Sample
    test_label = _b_sample - raw_b_sample if label == 'diff' else _b_sample
    if use_stu:
        x_sample = Gdec(Gstu(Genc(xa_sample, is_training=False),
            test_label, is_training=False), test_label, is_training=False)
    else:
        x_sample = Gdec(Genc(xa_sample, is_training=False), 
            test_label, is_training=False)

    # Initialization
    ckpt_dir = './model/checkpoints'
    tl.load_checkpoint(ckpt_dir, sess)

    return sess, x_sample, xa_sample, _b_sample, raw_b_sample


def inference(sess, x_sample, xa_sample, _b_sample, raw_b_sample, 
        image_path, attributes):
    """Inference function for the STGAN model.
    
    Given a TF model (STGAN), an input image with the centered face of a person 
    and some facial attributes (13), produces a replica where the facial 
    attributes of the initial person've been changed. 

    Args:
        sess (TF Session): Current session for inference.
        x_sample (tfTensor): Tensor of shape (n_img, 128, 128, 3).
        xa_sample (tfTensor): Input tensor of shape (n_img, 128, 128, 3).
        _b_sample (tfTensor): Label tensor of shape (n_img, 13).
        raw_b_sample (tfTensor): Label tensor of shape (n_img, 13).

    Returns:
        output_image (ndarray): Output of the STGAN in RGB format

    """
    # Image preprocessing
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))

    zero_att_list = np.zeros((1, 13))

    image = np.expand_dims(image.astype(np.float32) / 127.5 - 1, 0)

    output_image = sess.run(x_sample, feed_dict={xa_sample: image[..., ::-1],
        _b_sample: np.array([attributes]),
        raw_b_sample: zero_att_list})[0]

    output_image = (cv2.resize(output_image, 
        (IMG_DISPLAY_HEIGHT, IMG_DISPLAY_WIDTH)) + 1)* 127.5
    output_image = output_image.clip(0, 255).astype(np.uint8)

    return output_image


def final_image(input_image_path, output_image):
    """Concatenating images for visual purposes.

    Reshape all images and displays a concatenation of them in the following 
    fashion: original image, an arrow image, and the image with the
    modifications.

    Args:
        input_image_path (str): Path to the input image.
        output_image (ndarray): Output image in RGB format.

    Returns:
        final_image (ndarray): Horizontal concatenation of the images.

    """
    input_image = cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(input_image, 
        (IMG_DISPLAY_HEIGHT, IMG_DISPLAY_WIDTH))
    arrow = cv2.resize(cv2.cvtColor(cv2.imread('resources/images/arrow.jpg'), 
        cv2.COLOR_BGR2RGB), (IMG_DISPLAY_HEIGHT, IMG_DISPLAY_WIDTH))
    output_image = cv2.resize(output_image, 
        (IMG_DISPLAY_HEIGHT, IMG_DISPLAY_WIDTH))

    result_image = np.concatenate((input_image, arrow, output_image), axis=1)

    return result_image

def main():
    # Streamlit initialization
    st.title("Facial Attributes Modifier")
    st.markdown("""Welcome to the Facial Attributes Modifier app. Use the 
        *Options* panel on the left to choose an image located in the 
        *input_images/* folder and to modify up to 13 facial attributes. Try to 
        use images with well centered and aligned faces for better results. 
        Special thanks to the [STGAN](https://arxiv.org/abs/1904.09709v1) team 
        for their work""")
    ## Define holder for the processed image
    img_placeholder = st.empty()
    ## Sidebar
    st.sidebar.title("Options")
    ## Select image from the 'input_image/' folder
    Path('input_images/').mkdir(parents=True, exist_ok=True)
    image_paths = [path.resolve() for path in Path('input_images').glob("**/*") 
        if path.suffix in [".jpg", ".jpeg", ".png"]]
    image_path = st.sidebar.selectbox(
        "Select an image from the 'input_images/' folder", image_paths)
    #img_placeholder.image(cv2.imread(image_path))
    ## Tune sliders for each attribute
    attributes = []
    for att in ATTRIBUTES:
        value = st.sidebar.slider(att, 0, 100, 50)
        # Change the value acquired (between 0 to 100) to a new range (-1 to 1)
        value = (((value - 0) * (1 - (-1))) / (100 - 0)) + (-1)
        attributes.append(value)

    input_image_path = str(image_path)
    sess, x_sample, xa_sample, _b_sample, raw_b_sample = model_initialization()
    output_image = inference(sess, x_sample, xa_sample, _b_sample, 
        raw_b_sample, input_image_path, attributes)
    img_to_display = final_image(input_image_path, output_image)
    img_placeholder.image(img_to_display)
    
    ## Button to save output image
    if st.button('Save modification'):
        Path('output_images/').mkdir(parents=True, exist_ok=True)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'output_images/{image_path.name}.jpg', output_image)
        st.text(f"Image saved in output_images/{image_path.name}.jpg")

if __name__ == "__main__":
    main()

    