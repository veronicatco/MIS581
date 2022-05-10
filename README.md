# Sorghum Cultivar Identification

* Veronica Thompson
* Colorado State University Global
* MIS 581: Capstone
* Dr. Orenthio Goodwin
* 5/15/2022

#### Links:

###### [Neural Network Code](https://github.com/veronicatco/MIS581/blob/main/sorghum%20nn%20training.ipynb)
###### [Accuracy Analysis of Neural Network Models](https://github.com/veronicatco/MIS581/blob/main/training%20data%20analysis/training%20data%20analysis.md)
###### [Dataset on Kaggle](https://www.kaggle.com/competitions/sorghum-id-fgvc-9)
###### [Project Paper](https://github.com/veronicatco/MIS581/blob/main/ThompsonVeronica-MIS581-CTA7.pdf)

#### Abstract:

Large-scale plant breeding programs for sorghum, an important cereal crop, will enable increased production of both food and bioenergy. Analysis of growing plants is an important component of a breeding program. When the breeding program is conducted on a large scale such analysis can best be accomplished using automated processing of sensing data. An example of automated analysis is cultivar identification from photos, which can be used to verify the integrity of propagation experiments. This project evaluates sorghum cultivar identification employing deep learning methods of computer vision. Candidate models include models completely trained using the sorghum images and pretrained models employing transfer learning. All models were compared to a baseline model consisting of a simple convolution neural network, using classification accuracy as the metric. For each candidate model variations including larger input images and data augmentation were examined. All models exhibited significant overfitting when trained without data augmentation. Most models showed increased accuracy when trained with larger images. The most successful model contains three VGG-inspired convolution blocks and employs data augmentation.



