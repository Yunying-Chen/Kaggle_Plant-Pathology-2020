# Kaggle_Plant-Pathology-2020
An image classifier by using MobileNet based on Tensorflow Keras to recognize the condition of plant.




## Requirements
Tensorflow >=2.0            
Pandas            
Sklearn            
Opencv              

## Data       
Download Kaggle's dataset. The dataset has 1638 sample. The dataset is split into training set and validation set. Opencv is used to read the images and resize the images into (224 * 224 * 3) to fit in the Network input. In order to avoid the overfitting, data augmentation is applied to all trainning data by using rotation, flipping, shiftting etc.

## Network
This is a 35-layer CNN model with 224 * 224 * 3 as input and the output is 4 as the images are in 4 categories. 

## Train
```
python PlantPathology.py -train /PATH/TO/CSV -test /PATH/TO/CSV
```             
Example
```
python PlantPathology.py -train /train.csv -test /test.csv -lr 0.001 -output result.csv -e 100
```



## Eval
With Kaggle's raw dataset, it can reach 0.953 score.
![Image](https://github.com/Yunying-Chen/Kaggle_Plant-Pathology-2020/blob/master/IMG/score.png)
