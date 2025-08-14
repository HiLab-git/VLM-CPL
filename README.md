# VLM-CPL

### Overall Framework
There are three steps to train the classification network.
![overall](https://github.com/lanfz2000/VLM-CPL/blob/main/fig1.png)

### Data prepare
Download the HPH dataset [here](https://data.mendeley.com/datasets/h8bdwrtnr5/1)  
Download the LC25K dataset [here](https://huggingface.co/datasets/1aurent/LC25000)  
Download the CRC100K dataset [here](https://zenodo.org/records/1214456)  
Download the DigestPath dataset [here](https://digestpath2019.grand-challenge.org/)  

Using a 4:1 split for training and testing.

### Training process

First, use the on-the-shelf VLM for zero-shot inference with our proposed method to filter out noisy samples on the training set.  
In the vlm_cpl_LC25K.py file, there are two main functions, ```MVC```and```Prompt_feature_consensus```.  
You can use the combination of ```MVC```and```Prompt_feature_consensus``` or either one alone. You can also adjust the order of these two filters.
```
python vlm_cpl_LC25K.py --gpu 0
```
Second, after obtaining high-quality pseudo-labels, you can train a classification network.
```
python train_pseudo.py --gpu 0 --pseudo_csv <your_csv>
```
