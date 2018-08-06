'''
based on    https://arxiv.org/abs/1508.06576
and         https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398
'''

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import functools

import tensorflow as tf

import tensorflow.contrib.eager as tfe
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

def resize_image(file, size):
    '''
    resizes image at file location and returns path to resized Image
    '''
    im = Image.open(file)
    im = im.resize(size, Image.ANTIALIAS)

    outfile = "resized_{}".format(file)
    im.save(outfile)

    return outfile

def load_image(file):
    '''
    takes path to image file, loads it, then returns an array of its values
    expands the dimensions to (1,n_values) so returned array plays well with
    batch sizes
    '''
    im = image.load_img(file, target_size = (in_shape[0], in_shape[1]))
    im = image.img_to_array(im)

    # image needs to have a 'batch  size' for tf
    # expand_dims: [1,2] -> [[1,2]] batch size = 1
    im = np.expand_dims(im, axis=0)
    return im

def load_and_process_img(file):
    '''
    takes an image and performs preprocessing expected by VGG.
    because were using the VGG pretrained network, we want to preprocess the
    images the way that net expects.
    '''
    im = load_image(file)
    im = tf.keras.applications.vgg19.preprocess_input(im)
    return im

# NOTE: i dont pretend to understand this step. see below link
# https://colab.research.google.com/github/tensorflow/models/blob/master/research/nst_blogpost/4_Neural_Style_Transfer_with_Eager_Execution.ipynb#scrollTo=mjzlKRQRs_y2&line=9&uniqifier=1
def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = x.reshape(in_shape)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                             "dimension [batch, height, width, channel] or [height_width_channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # perform the inverse of the preprocessiing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x

def show_image(im, title = None):
    # load_images adds batch dimension, get rid of that
    out_im = np.squeeze(im, axis=0)
    # normalize for display
    out_im = out_im.astype('uint8')

    # create plot
    plt.imshow(out_im)

    if title is not None:
        plt.title(title)

    plt.imshow(out_im)

def get_model():
    '''
    the paper recomends using the VGG19 pretrained model to do style transfer.
    this is in part becasue it is simple, and in part because it just works well
    this loads the model and then makes a new net that just returns the output
    from the internediate layers that we are interested in.
    '''
    # load model
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    # get output were interested in
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs

    # return a keras model
    return models.Model(vgg.input, model_outputs)

def get_content_loss(actual, target):
    return tf.reduce_mean(tf.square(actual-target))

def gram_matrix(input_tensor):
    # We make the image channels first
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

def get_style_loss(base_style, gram_target):
    """Expects two images of dimension h, w, c"""
    # height, width, num filters of each layer
    # We scale the loss at a given layer by the size of the feature map and the number of filters
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)

    return tf.reduce_mean(tf.square(gram_style - gram_target))# / (4. * (channels ** 2) * (width * height) ** 2)

def get_feature_representations(model, content_path, style_path):
    """Helper function to compute our content and style feature representations.

    This function will simply load and preprocess both the content and style
    images from their path. Then it will feed them through the network to obtain
    the outputs of the intermediate layers.

    Arguments:
    model: The model that we are using.
    content_path: The path to the content image.
    style_path: The path to the style image

    Returns:
    returns the style features and the content features.
    """
    # Load our images in
    content_image = load_and_process_img(content_path)
    style_image = load_and_process_img(style_path)

    # batch compute content and style features
    stack_images = np.concatenate([style_image, content_image], axis=0)
    model_outputs = model(stack_images)

    # Get the style and content feature representations from our model
    style_features = [style_layer[0] for style_layer in model_outputs[:num_style_layers]]
    content_features = [content_layer[1] for content_layer in model_outputs[num_style_layers:]]
    return style_features, content_features

def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    """This function will compute the loss total loss.

    Arguments:
    model: The model that will give us access to the intermediate layers
    loss_weights: The weights of each contribution of each loss function.
      (style weight, content weight, and total variation weight)
    init_image: Our initial base image. This image is what we are updating with
      our optimization process. We apply the gradients wrt the loss we are
      calculating to this image.
    gram_style_features: Precomputed gram matrices corresponding to the
      defined style layers of interest.
    content_features: Precomputed outputs from defined content layers of
      interest.

    Returns:
    returns the total loss, style loss, content loss, and total variational loss
    """
    style_weight, content_weight = loss_weights

    # Feed our init image through our model. This will give us the content and
    # style representations at our desired layers. Since we're using eager
    # our model is callable just like any other function!
    model_outputs = model(init_image)

    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    # Accumulate style losses from all layers
    # Here, we equally weight each contribution of each loss layer
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

    # Accumulate content losses from all layers
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer* get_content_loss(comb_content[0], target_content)

    style_score *= style_weight
    content_score *= content_weight

    # Get total loss
    loss = style_score + content_score
    return loss, style_score, content_score

def compute_grads(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    # Compute gradients wrt input image
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_image']), all_loss

def run_style_transfer(content_path,
                       style_path,
                       num_iterations=1000,
                       content_weight=1e3,
                       style_weight=1e-2):
    display_num = 100
    # We don't need to (or want to) train any layers of our model, so we set their
    # trainable to false.
    model = get_model()
    for layer in model.layers:
        layer.trainable = False

    # Get the style and content feature representations (from our specified intermediate layers)
    style_features, content_features = get_feature_representations(model, content_path, style_path)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    # Set initial image
    init_image = load_and_process_img(content_path)
    init_image = tfe.Variable(init_image, dtype=tf.float32)
    # Create our optimizer
    opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

    # For displaying intermediate images
    iter_count = 1

    # Store our best result
    best_loss, best_img = float('inf'), None

    # Create a nice config
    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }

    # For displaying
    plt.figure(figsize=(14, 7))
    num_rows = (num_iterations / display_num) // 5
    start_time = time.time()
    global_start = time.time()

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means
    for i in range(num_iterations):
        grads, all_loss = compute_grads(cfg)
        loss, style_score, content_score = all_loss
        # grads, _ = tf.clip_by_global_norm(grads, 5.0)
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)
        end_time = time.time()

        if loss < best_loss:
            # Update best loss and best image from total loss.
            best_loss = loss
            best_img = init_image.numpy()

        if i % display_num == 0:
            print('Iteration: {}'.format(i))
            print('Total loss: {:.4e}, '
                    'style loss: {:.4e}, '
                    'content loss: {:.4e}, '
                    'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))
            start_time = time.time()

        # Display intermediate images
        if iter_count > num_rows * 5: continue
        plt.subplot(num_rows, 5, iter_count)
        # Use the .numpy() method to get the concrete numpy array
        plot_img = init_image.numpy()
        plot_img = deprocess_img(plot_img)
        plt.imshow(plot_img)
        plt.title('Iteration {}'.format(i + 1))
        iter_count += 1
    print('Total time: {:.4f}s'.format(time.time() - global_start))

    return best_img, best_loss


def  show_results (best_img, content_path, style_path, show_large_final=True):
    plt.figure(figsize=(15, 15))
    x = deprocess_img(best_img)
    content = load_image(content_path)
    style = load_image(style_path)

    plt.subplot(1, 3, 1)
    show_image(content, 'Content Image')

    plt.subplot(1, 3, 2)
    show_image(style, 'Style Image')

    plt.subplot(1, 3, 3)
    plt.imshow(x)
    plt.title('Output Image')
    plt.show()

    if show_large_final:
        plt.figure(figsize=(10, 10))

        plt.imshow(x)
        plt.title('Output Image')
        plt.show()

in_shape = (500,500,3)
content = 'zach.jpg'
style = 'frida2.jpg'

content = resize_image(content, (in_shape[0], in_shape[1]))
style = resize_image(style, (in_shape[0], in_shape[1]))

tf.enable_eager_execution()
print("executing eagarly: {}".format(tf.executing_eagerly()))

content_im = load_image(content)
style_im = load_image(style)

plt.figure(figsize = (10,10))
plt.subplot(1,2,1)
show_image(content_im, 'content')
plt.subplot(1,2,2)
show_image(style_im, 'style')
plt.show()

input('press enter to continue ...')

plt.close()

'''
were using a pretrained image classification network for this project .
part of the reason were doing that is becasue it already has an internal
representation of what an image is. that internal representation occurs in
the middle layers. in order to do our style transfer we are going to use the
output from some of those layers. the layers we are using are shown below
'''

# content layer where will pull our feature maps
content_layers = ['block5_conv2']

# style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
               ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

best, best_loss = run_style_transfer(content,
                                     style, num_iterations = 2500)

show_results(best, content, style)
