# Kaggle_Plant-Pathology-2020
An image classifier by using MobileNet based on Tensorflow Keras to recognize the condition of plant.




## Requirements
Tensorflow >=2.0
Pandas
Sklearn
Opencv

## Data       
The dataset has 1638 sample. The dataset is split into training set and validation set. Opencv is used to read the images and resize the images into (224 * 224 * 3) to fit in the Network input. In order to avoid the overfitting, data augmentation is applied to all trainning data by using rotation, flipping, shiftting etc.

## Network
MobileNet is called from Keras and the output is changed to 4 as the images are in 4 categories. 

## Train
```
python PlantPathology.py -train /PATH/TO/CSV -test /PATH/TO/CSV
```             
Example
```
python mnist.py -train Dataset/train.csv -test Dataset/test.csv -lr 0.001 -output result.csv -e 100
```



## Eval
With Kaggle's raw dataset, it can reach 0.953 score.
![Image](https://github.com/Yunying-Chen/Kaggle_Plant-Pathology-2020/blob/master/IMG/score.png)
