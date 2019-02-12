# 聊天机器人
一个使用Tensorflow框架通过Sequence To Sequence模型训练出的简单的聊天机器人。  

![效果截图](https://raw.githubusercontent.com/luojiangtao/ChatBot_web/master/1.png)  


*   在线体验 http://ai.luojiangtao.com:8000
## 2.项目结构介绍  
 - config.py  整个项目的配置文件，如语料库的存放位置，模型的参数等  
 - data_unit.py 处理语料库的类，对原始语料进行清洗，并生成批训练数据。  
 - seq2seq.py 构建了一个Sequence To Sequence模型，包含编码器、解码器、优化器、训练过程、预测过程等部分。
 - train.py 用于模型的训练。  
 - predict.py 用于模型的测试。  
 - data 该文件夹用于保存语料文件。  
 - model 该文件夹用于保存训练好的模型。

## 3.项目的使用  
*   运行train.py训练自己的模型。 
*   运行predict.py 测试模型的效果 
*   运行predict_web.py 使用网页界面，运行成功后访问 http://localhost:8000/ 

## 作者使用版本
*   python 3.6
*   tensorflow 1.12.0
 
 
