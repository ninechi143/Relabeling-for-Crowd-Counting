# Relabeling-Crowd-Counting

This is the official Tensorflow implementation of the paper submission: Relabeling for Small-Data Crowd Count Estimation.

## Code

### Install Dependencies

The code is used with Python 3.6, requiring the packages listed below.

```
tensoflow==1.14.0
opencv-python
pillow
scipy
numpy
```
The packages can be easily installed by pip install.

### Train

1. Download the ShanghaiTech Dataset and the initial weight for the VGG backbone. [Google Drive Link](https://drive.google.com/file/d/1-okjXwqTAprjuHTzUmEh3vpoGzZ4xYR6/view?usp=sharing)

2. Unzip the downloaded file and modify the path to the same directory of this repository.

3. Run the python file for the data preprocessing.

  `python preprocess_ShanghaiTech.py`

4. Run the python file for the first training stage with relabeling.

  `python Self_Recalibration_First_Stage.py`

5. After the first training, run the python file for the scale re-estimation.

  `python Inference_for_re-estimate.py`

6. Run the python file for the second training stage.

  `python Self_Recalibration_Second_Stage.py`


### The other details will be updated soon.

to be continued.
