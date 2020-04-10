import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# incpetion模块，包含了1x1,3x3,5x5卷积层和max pooling
class InceptionBlock(layers.Layer):

    def __init__(self, filters):
        super(InceptionBlock, self).__init__()
        assert len(filters) == 6, "fliters's length is not euqal to 6"
        self.conv1x1 = layers.Conv2D(filters[0], kernel_size=1, strides=1, padding="same", activation=tf.nn.relu)
        self.conv3x3_reduce = layers.Conv2D(filters[1], kernel_size=1, strides=1, padding="same", activation=tf.nn.relu)
        self.conv3x3 = layers.Conv2D(filters[2], kernel_size=3, strides=1, padding="same", activation=tf.nn.relu)
        self.conv5x5_reduce = layers.Conv2D(filters[3], kernel_size=1, strides=1, padding="same", activation=tf.nn.relu)
        self.conv5x5 = layers.Conv2D(filters[4], kernel_size=5, strides=1, padding="same", activation=tf.nn.relu)
        self.maxPooling = layers.MaxPool2D(pool_size=3, strides=1, padding="same")
        self.poolingConv1x1 = layers.Conv2D(filters[5], kernel_size=1, strides=1, padding="same", activation=tf.nn.relu)

    def call(self, input, training=None):
        # 分支1
        path1 = self.conv1x1(input)
        # 分支2
        path2 = self.conv3x3_reduce(input)
        path2 = self.conv3x3(path2)
        # 分支3
        path3 = self.conv5x5_reduce(input)
        path3 = self.conv5x5(path3)
        # 分支4
        path4 = self.maxPooling(input)
        path4 = self.poolingConv1x1(path4)
        # 沿通道方向拼接
        outs = [path1, path2, path3, path4]
        outs = tf.concat(outs, axis=-1)
        return outs

# 分类器（可有可无）
class Classifier(keras.Model):

    def __init__(self, nc):
        super(Classifier, self).__init__()
        self.averagePooling = layers.AveragePooling2D(pool_size=5, strides=3)
        self.conv1x1 = layers.Conv2D(128, kernel_size=1, strides=1, padding="same", activation=tf.nn.relu)
        self.fc1 = layers.Dense(1024, activation=tf.nn.relu)
        self.fc2 = layers.Dense(nc)

    def call(self, input, training=None):
        out = self.averagePooling(input)
        out = self.conv1x1(out)
        out = tf.reshape(out, [-1, 4 * 4 * 128])
        out = self.fc1(out)
        out = self.fc2(out)

        return out


class Inception(keras.Model):

    def __init__(self, nc):
        super(Inception, self).__init__()
        self.conv7x7 = layers.Conv2D(64, kernel_size=7, strides=2, padding="same", activation="relu")
        self.maxPooling3x3 = layers.MaxPool2D(pool_size=3, strides=2, padding="same")
        self.batchNormal1 = layers.BatchNormalization()
        self.conv1x1 = layers.Conv2D(64, kernel_size=1, strides=1, padding="same", activation="relu")
        self.conv3x3 = layers.Conv2D(192, kernel_size=3, strides=1, padding="same", activation="relu")
        self.batchNormal2 = layers.BatchNormalization()

        self.inception3a = InceptionBlock([64, 96, 128, 16, 32, 32])
        self.inception3b = InceptionBlock([128, 128, 192, 32, 96, 64])

        self.inception4a = InceptionBlock([192, 96, 208, 16, 48, 64])
        self.inception4b = InceptionBlock([160, 112, 224, 24, 64, 64])
        self.inception4c = InceptionBlock([128, 128, 256, 24, 64, 64])
        self.inception4d = InceptionBlock([112, 144, 288, 32, 64, 64])
        self.inception4e = InceptionBlock([256, 160, 320, 32, 128, 128])

        self.inception5a = InceptionBlock([256, 160, 320, 32, 128, 128])
        self.inception5b = InceptionBlock([384, 192, 384, 48, 128, 128])

        self.averagePooling = layers.AveragePooling2D(pool_size=7, strides=1)
        self.fc = layers.Dense(nc)

        self.classifier1 = Classifier(nc)
        self.classifier2 = Classifier(nc)

    def call(self, input, training=None):
        # input:[batchsz,224,224,3]

        out = self.maxPooling3x3(self.conv7x7(input))
        out = self.batchNormal1(out)
        out = self.conv3x3(self.conv1x1(out))
        out = self.batchNormal2(out)
        out = self.maxPooling3x3(out)

        out = self.inception3a(out)
        out = self.inception3b(out)

        out = self.maxPooling3x3(out)
        out = self.inception4a(out)
        sub_out_1 = self.classifier1(out)
        out = self.inception4b(out)
        out = self.inception4c(out)
        out = self.inception4d(out)
        out = self.inception4e(out)
        sub_out_2 = self.classifier2(out)

        out = self.maxPooling3x3(out)
        out = self.inception5a(out)
        out = self.inception5b(out)

        out = self.averagePooling(out)
        out = tf.squeeze(out, axis=[1, 2])
        out = self.fc(out)

        return sub_out_1, sub_out_2, out



