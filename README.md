
# Advanced LSTM Model for Multi-class Text Classification

![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)



## Summary
This project implement advanced NLP using Long short-term memory (LSTM) for Multi-class text classification in python.
## Abstract
Text documents are essential as they are one of the richest sources of data for 
businesses. Text documents often contain crucial information which might shape 
the market trends or influence the investment flows. Therefore, companies often 
hire analysts to monitor the trend via articles posted online, tweets on social media 
platforms such as Twitter or articles from newspaper. However, some companies 
may wish to only focus on articles related to technologies and politics. Thus, 
filtering of the articles into different categories is required. 

Often the categorization of the articles is conduced manually and retrospectively; 
thus, causing the waste of time and resources due to this arduous task.
## Data Set
Dataset used for training and validation can be obtain from [Susan Li's Github page](https://github.com/susanli2016) @ https://github.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/blob/master/bbc-text.csv
## Run Locally

Clone the project

```bash
  git clone https://github.com/liewwy19/Advanced-LSTM-Model-for-Multi-class-Text-Classification.git
```



## Methodology
+ Data Preparation
    + Data Loading
    + Data Inspection (EDA)
    + Data Cleaning
        + remove duplicates 
        + remove monetory value (e.g. £35m, $180bn)
        + remove abbreviation inside parenthesis (e.g. (IOC))
        + remove puntuations
        + remove single/double letters residual 
+ Data Preprocessing
    + Feature Selection
    + Data Tranform
        + train the tokenizer
        + transform the text using trained tokenizer
        + padding the text
        + apply label encoding (OneHotEncoding)
    + Train-Test-Split
+ Model Development
    + construct embedding layer
    + construct LSTM algorithm using Sequntial API from Tensorflow Keras library
    + compile the model
+ Model Training
    + Define Callback Functions 
    + Fit the model
+ Model Analysis
    + Plot Loss and Accuracy chart for Analysis
    + construct confusion matrix with actual and predicted values
+ Model Deployment  
    + model saving
        + trained model
        + one hot encoder model
        + tokenizer
    + perform prediction with unseen data
## The Model

![](https://github.com/liewwy19/Advanced-LSTM-Model-for-Multi-class-Text-Classification/blob/main/model.png?raw=True)

## Analysis
The model able to achieve accuracy of more than 90% and average f1-score of 0.91 across 5 categories.

![](https://github.com/liewwy19/Advanced-LSTM-Model-for-Multi-class-Text-Classification/blob/main/confusion_matrix.png?raw=True)
## Results

#### Unseen Text:

    'Messi returns to the French club at an important time, with the team having lost for the first time this season against Lens last weekend. PSG's lead at the top of the Ligue 1 table has been cut to just four points as a result. The team's next game is against Chatearoux in the French Cup on Friday  a game that may come too soon for Messi.'

|  |Business | Entertainment | Politics | Sport | Tech |
| --- | --- | --- | --- | --- | --- |
| new_text | 0.018292    |    0.02178 | 0.004566 | 0.954268 | 0.001094 |

#### Predicted Category:

    'Sport'




## Contributing

This project welcomes contributions and suggestions. 

    1. Open issues to discuss proposed changes 
    2. Fork the repo and test local changes
    3. Create pull request against staging branch


## Acknowledgements

 - [GitHub - susanli2016 / PyCon-Canada-2019-NLP-Tutorial](https://github.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial)
 - [LSTM for Text Classification in Python](https://www.analyticsvidhya.com/blog/2021/06/lstm-for-text-classification/)
 - [Introduction to Keras pad_sequences](https://www.educba.com/keras-pad_sequences/)
 - [Selangor Human Resource Development Centre (SHRDC)](https://www.shrdc.org.my/)

