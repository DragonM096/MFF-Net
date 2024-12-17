# 1、Introduction
This project currently has a paper under submission, titled "**MFF-Net: A Multi-Scale Feature Fusion Network for Generalized Forgery Image Detection**". 
We will gradually upload the remaining code in the future.

# 2、Setup
Install packages: `pip install -r requirements.txt`

# 3、Dataset
**The training and validation dataset** we use comes from [this link](https://github.com/peterwang512/CNNDetection), which was collected and organized by the paper 'CNN generated images are surprisingly easy to spot... for now'.
The file organization order for training and validation datasets is as follows：
```
Training and validation dataset
	|- train(90% of downloaded dataset)
		|_ 0_real
			xxxx.png
			...
		|_ 1_fake
			yyyy.png
			...
	|- val(10% of downloaded dataset)
		|_ 0_real
			xxxx.png
			...
		|_ 1_fake
			yyyy.png
			...
```
**test dataset** will be provided in the near future. Coming soon~~~

# 4、How to train our model
We provide a sample script to train our model by executing `bash train.sh`, where you can adjust the following settings:
```
--name: Specify the directory where you want to store the checkpoints.
--blur_prob: Set the probability of applying Gaussian blur to the image.
--blur_sig: Define the σ parameter for Gaussian blur.
--jpg_prob: Set the probability of applying JPEG compression to the image.
--jpg_method: Choose the compression method, either cv2 or pil.
--jpg_qual: Adjust the JPEG compression quality.
--dataroot: Provide the path to the training and validation datasets.
```

