# Ad^2Attack：Adaptive Adversarial Attack on Real-Time UAV Tracking

## Demo video

- :video_camera: Our video on [bilibili](https://www.bilibili.com/video/BV1S44y1b7xC?spm_id_from=333.999.0.0) demonstrates the test results of Ad^2Attack on several sequences.

![Ad^2Attack](Fig/Attack.gif)

## Environment setup
This code has been tested on Ubuntu 18.04, Python 3.8.3, Pytorch 0.7.0/1.6.0, CUDA 10.2.
Please install related libraries before running this code: 
```bash
pip install -r requirements.txt
```

## Attack on Trackers

## [SiamAPN] 
The pre-trained model of SiamAPN can be found at (epoch=37) : [general_model](https://pan.baidu.com/s/1GSgj3UwObcUKyT8TFSJ5qA)(code:w3u5) 
and the pre-trained model of Ad^2Attack can be found at /checkpoints/AdATTACK/model.pth

Ad^2Attack on other trackers, e.g., SiamCAR, SiamGAT, HiFT, SiamAPN++ will be released soon.

## Datasets Setting
We evaluate our attack method on 3 well-known UAV tracking benchmark, i.e., UAV123, UAV112 and UAVDT
You can download them and put them in /pysot/test_dataset
remember change the path in Setting.py

## Test Attack
```
vim ~/.bashrc
export PYTHONPATH=/home/user/Ad^2Attack:$PYTHONPATH
export PYTHONPATH=/home/user/Ad^2Attack/pysot:$PYTHONPATH
export PYTHONPATH=/home/user/Ad^2Attack/pix2pix:$PYTHONPATH
source ~/.bashrc
```

```
python pysot/tools/test.py 	        \
	--trackername SiamAPN           \ # tracker_name
	--dataset V4RFlight112          \ # dataset_name
	--snapshot snapshot/general_model.pth   # model_path
```

The testing result will be saved in the `results/dataset_name/tracker_name` directory.


## Evaluation 
If you want to evaluate the Ad^2Attack on trackers, please put those results into  `results` directory.
```
python pysot/tools/eval.py 	                          \
	--tracker_path ./results          \ # result path
	--dataset V4RFlight112            \ # dataset_name
	--tracker_prefix 'general_model'  \ # tracker_name
```


## Contact
If you have any questions, please contact me.

Sihang Li

Email: [sihangli990704@outlook.com](sihangli990704@outlook.com)


## Acknowledgement
The code is implemented based on [pysot](https://github.com/STVIR/pysot), [SiamAPN](https://github.com/vision4robotics/SiamAPN) and [CSA](https://github.com/MasterBin-IIAU/CSA). We would like to express our sincere thanks to the contributors.
