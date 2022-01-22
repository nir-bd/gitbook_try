# Tensorleap Guide



## Tensorleap Guide

Tensorleap’s platform offers unique tools for debugging, observability, and explainability within the development of deep-learning models. In order to make those deep analyses, our platform tracks each sample, feature, layer and collects a large number of indicators. For integration to begin, the model needs to be exported, along with a dataset, and a script to read the dataset. The purpose of this guide is to describe how to convert a model defined in PyTorch/Tensorflow into a Tensorleap-compatible file format, and the script that is used to read and preprocess data from your dataset.

### Model Import

A deep learning model consists of multiple components:

* Model architecture - the layers and their connections
* Weights values - the state of the model after training
* A set of loss functions, an optimizer and a set of metrics

Tensorflow or Pytorch models can be saved to a serialization format for trained models, which stores the model's weights and details on its architecture and how it was trained. The saved model can then be used in Tensorleap independently of the code that created it. Tensorleap reads this serialization file, loads it, and displays it in the platform. For your convenience, below are a few references to code one-liners.

#### Tensorflow 2 (Keras) - Save Model

The following command generates a folder with the serialized model data, that contains the model architecture and weights.

```python
model.save('path/to/location')
```

#### Tensorflow 2 (Keras) - H5 format

Keras also supports saving the model's architecture and weights in a single HDF5 file. This is essentially a light-weight alternative to the “Save Model” option described above.

```python
model.save("my_h5_model.h5")
```

More info can be found here: https://www.tensorflow.org/guide/keras/save\_and\_serialize

#### PyTorch - ONNX Format

Tensorleap supports PyTorch and requires the model to first be exported to an .onnx file format in order to read it.

```python
dummy_input = torch.randn(10, 3, 224, 224, device="cuda")
model = torchvision.models.alexnet(pretrained=True).cuda()

input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)
```

More info can be found here: https://pytorch.org/docs/stable/onnx.html

#### Leap Save Model

To manage the adaptation and saving of a model import prior to its import, we recommend using the `leap_save_model` function. The code for this function can be found in `model.py` within the provided repository. The function saves the model to a specified path and can be employed to adapt models for use within the Tensorleap platform. This function is also used by Tensorleap's `leap-cli` for importing models. Sample code:

```python
from src.model import MyModel

def leap_save_model(target_file_path: Path):
    my_model = MyModel()
    model.save(target_file_path)
```

### Dataset Integration

Dataset preprocessing scripts are used by Tensorleap to encode data for the network. The script includes the preprocessing function that prepares the data state for fetching into the neural network. Providing encoding functions for each input, which reads them and prepares them for neural networks. Additionally, a ground truth encoding function that is correlated with each output.

#### Preprocessing Function

The `preprocessing` function is called once, just before the training/evaluating process. It prepares the training data and validation data (`train_data` and `val_data` in the sample code below). In the sample code below, the function downloads and reads a `TFRecord` file of a pandas dataframe, splits it into `train` and `validation`, and finally returns `train_data` and `validation_data`.

```python
from sklearn.model_selection import train_test_split
from src.etl.datasetintegration.datasetbinder import SubsetResponse

def extract_fn(tfrecord):
    # Extract features using the keys set during creation
    features = {
        'image_fpath': tf.FixedLenFeature([], tf.string),
        'target': tf.FixedLenFeature([], tf.int64)
    }

    # Extract the data record
    sample = tf.parse_single_example(tfrecord, features)
    return sample

def preprocessing():
    # arrange the data
    train_path = "Tensorleap/train.tfrecord"
    validation_path = "Tensorleap/validation.tfrecord"
    
    train_dataset = tf.data.TFRecordDataset([train_path]).map(extract_fn)
    validation_dataset = tf.data.TFRecordDataset([validation_path]).map(extract_fn)
    
    train_dataset = list(train_dataset.as_numpy_iterator())
    validation_dataset = list(validation_dataset.as_numpy_iterator())
    
    train = SubsetResponse(length=len(train_dataset), data=train_dataset)
    val = SubsetResponse(length=len(validation_dataset), data=validation_dataset)
    
    response = [train, val]
    return response
```

#### Batch Generation Functions

During the training or evaluation process, the samples are fetched to the neural network in batches.

This section describes functions that are called during the batch generation process, for every sample within the batch. As an example, a training set of 10K samples would result in 10K calls for each function per epoch. Consequently, it is recommended to avoid long processes in those functions.

**Input Encoder Function(s)**

The input encoder functions receive data (`train_data` / `validation_data` according to the state) as an argument, as well as idx that represents the index of the sample. For each model input, there should be an encoding function that extracts and generates the input data per one sample. In order to facilitate tracking and analysis, Tensorleap requires samples to be fetched by index. Sample code:

```python
def image_input_encoder(self,idx,subset_response):
    image_fpath = subset_response.data[idx]["image_fpath"]
    img = imread(image_fpath)
    return img
```

#### Ground Truth Encoder Function(s)

Similar to the input encoder functions, there are also ground truth encoder functions correlated to each output of the neural network. Sample code:

```python
def ground_truth_encoder(self,idx,subset_response):
    return subset_response.data[idx]["target"]
```

#### Metadata Encoder Function(s)

Furthermore, it is optional to add metadata functions. Those functions return additional data about the sample. This enables querying by each value and detect various related correlations. Sample code:

```python
def color_metadata(self,idx,subset_response):
    gender = subset_response.data[idx]["color"] #black, blue, brown, gray, green, orange, pink, red, violet, white, yellow
    return color

def shape_metadata(self,idx,subset_response):
    gender = subset_response.data[idx]["shape"] #long, round, rectangular, square
    return gender
```

#### Test

In order to test the code, the following scripts use the functions above as they will be used within the Tensorleap platform.

The script reads the preprocessed data, and fetches a sample from the training set, and a sample from the validation set. Finally, it prints the two sample inputs along with the ground truths. Note - the function is presented here for clarification purposes only, and is not required by Tensorleap.

```python
train_data, validation_data = preprocessing()
fetch_idx = 0 # or any other index.

# for testing the training set
input_feature_1 = image_input_encoder(fetch_idx, train_data)
ground_truth_1 = ground_truth_encoder(fetch_idx, train_data)

# print the training sample
print(input_feature_1)
print(ground_truth_1)

# for testing the validation set
input_feature_1 = image_input_encoder(fetch_idx, validation_data)
ground_truth_1 = ground_truth_encoder(fetch_idx, validation_data)

# print the training sample
print(input_feature_1)
print(ground_truth_1)
```

#### Leap Binding Functions

The `leap` is an instance pre-set globally by Tensorleap's engine. Its purpose is to represent the dataset by introducing all the functions above. In the following sample code, we describe how the attributes and functions are bound:

```python
from src.contract.common.enums import DatasetInputType, DatasetOutputType, DatasetMetadataType

leap.set_subset(ratio=1, function=preprocessing, name='ImageClassificationSubset')

leap.set_input(function=image_input_encoder, subset='ImageClassificationSubset', input_type=DatasetInputType.Image, name='Image')

leap.set_ground_truth(function=ground_truth_encoder, subset='ImageClassificationSubset',
                      ground_truth_type=DatasetOutputType.Classes,
                      name='ground_truth', labels=['Dog', 'Cat', 'Mouse'], masked_input=None)

leap.set_metadata(function=color_metadata, subset='ImageClassificationSubset', metadata_type=DatasetMetadataType.string,
                  name='color')
leap.set_metadata(function=shape_metadata, subset='ImageClassificationSubset', metadata_type=DatasetMetadataType.string,
                  name='shape')
```

#### Additional Example

Below is another example of dataset integration. Sample code:

```python
from typing import List
from src.contract.common.enums import DatasetInputType, DatasetOutputType, DatasetMetadataType
from src.etl.datasetintegration.datasetbinder import SubsetResponse

import tensorflow as tf
from keras.datasets import mnist
import numpy as np

def subset_subset0() -> List[SubsetResponse]:
    (trainX, trainy), (testX, testy) = mnist.load_data()
    return [SubsetResponse(length=10000, data={"data": trainX, "label": trainy}),
            SubsetResponse(length=10000, data={"data": testX, "label": testy})]


def input_image(idx, samples):
    img = samples.data["data"][idx]
    return np.array(img)[..., np.newaxis]


def ground_truth_num(idx, samples):
    label = samples.data["label"][idx]
    return np.eye(10)[label]


def metadata_label(idx, samples):
    label = samples.data["label"][idx]
    return str(label)


leap.set_subset(1.0, subset_subset0, 'subset0')

leap.set_input(input_image, 'subset0', DatasetInputType.Image, 'image')

leap.set_ground_truth(ground_truth_num, 'subset0', DatasetOutputType.Classes, 'num',
                      labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], masked_input=None)

leap.set_metadata(metadata_label, 'subset0', DatasetMetadataType.int, 'label')
```

