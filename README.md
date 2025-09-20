# Requirements
* Python 3.10.12
* PyTorch 2.0.1
* Torchvision 0.15.2
* Others specified in [requirements.txt](requirements.txt)

# Data Preparation
1. Download ShanghaiTech and UCF-QNRF datasets from official sites and unzip them.
2. Run the following commands to preprocess the datasets:
    ```
    python utils/preprocess_data.py --origin-dir [path_to_ShanghaiTech]/part_A --data-dir data/sta
    python utils/preprocess_data.py --origin-dir [path_to_ShanghaiTech]/part_B --data-dir data/stb
    python utils/preprocess_data.py --origin-dir [path_to_UCF-QNRF] --data-dir data/qnrf
    ```
3. Run the following commands to generate GT density maps:
    ```
    python dmap_gen.py --path data/sta
    python dmap_gen.py --path data/stb
    python dmap_gen.py --path data/qnrf


# Training
Run the following command:
```
python main.py --task train --config configs/sta_train.yml
```
You may edit the `.yml` config file as you like.
