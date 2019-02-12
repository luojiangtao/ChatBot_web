"""
    对训练好的模型进行测试
    运行此文件开始与机器人聊天
    注意只能进行中文聊天
"""

from flask import Flask, jsonify, render_template, request
import data_unit
import os
import tensorflow as tf
from seq2seq import Seq2Seq
import numpy as np
from config import BASE_MODEL_DIR, MODEL_NAME, data_config, model_config

class LoadModel(object):
    def __init__(self):
        self.du = data_unit.DataUnit(**data_config)
        self.save_path = os.path.join(BASE_MODEL_DIR, MODEL_NAME)
        self.batch_size = 1
        tf.reset_default_graph()
        self.model = Seq2Seq(batch_size=self.batch_size,
                        encoder_vocab_size=self.du.vocab_size,
                        decoder_vocab_size=self.du.vocab_size,
                        mode='decode',
                        **model_config)

        self.sess = tf.InteractiveSession()
        self.init = tf.global_variables_initializer()

        self.sess.run(self.init)
        self.model.load(self.sess, self.save_path)



    def predict(self, input_string=""):
        """
        针对用户输入的聊天内容给出回复
        :return:
        """

        # with tf.Session() as sess:
        # self.init = tf.global_variables_initializer()
        # self.sess.run(self.init)
        # self.model.load(sess, self.save_path)

        indexs = self.du.transform_sentence(input_string)
        x = np.asarray(indexs).reshape((1,-1))
        xl = np.asarray(len(indexs)).reshape((1,))
        pred = self.model.predict(
            self.sess, np.array(x),
            np.array(xl)
        )

        return self.du.transform_indexs(pred[0])

loadModel = LoadModel()
# webapp
app = Flask(__name__, static_folder='templates', static_url_path='')


'''
request.json
一维数组，784个特征
[255, 161, 0, 0,....]
'''
@app.route('/api/chat', methods=['GET'])
def chat():
    message = request.args.get('message')

    print(message)
    #一维数组，输出10个预测概率
    output = loadModel.predict(message)
    return output
    # return "你也好"


@app.route('/')
def main():
    return render_template('index.html')
    # return 'Hello World!'


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000)
    # app.run(host='localhost',port=8889)
    # print(loadModel.predict("你好"))
    # print(loadModel.predict("我喜欢你"))
