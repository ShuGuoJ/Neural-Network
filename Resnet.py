import tensorflow as tf
from tensorflow.keras import layers,Sequential
from tensorflow import keras

class BasicBlock(layers.Layer):

    def __init__(self,filter_num,strides=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(filter_num,kernel_size=3,strides=strides,padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()

        self.conv2 = layers.Conv2D(filter_num,kernel_size=3,strides=1,padding='same')
        self.bn2 = layers.BatchNormalization()

        if strides!=1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num,kernel_size=1,strides=strides))
        else:
            self.downsample = lambda x:x

    def call(self, input, training=None):
        out = self.conv1(input)
        out = self.bn1(out,training=training)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out,training=training)

        out += self.downsample(input)
        out = self.relu(out)
        return out

class Resnet(keras.Model):

    def __init__(self,layer_dims,num_classes): #[2,2,2,2]
        super(Resnet,self).__init__()
        self.stem = Sequential([
            layers.Conv2D(64,3,1),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(2,1,padding='same')
        ])

        self.layer1 = self.build_resblock(64,layer_dims[0])
        self.layer2 = self.build_resblock(128,layer_dims[1],strides=2)
        self.layer3 = self.build_resblock(256,layer_dims[2],strides=2)
        self.layer4 = self.build_resblock(512,layer_dims[3],strides=2)

        # [b,c]
        self.avgpool = layers.AveragePooling2D(4)
        # [b,num_classes]
        self.fc = layers.Dense(num_classes)



    def call(self,input,training=None):
        out = self.stem(input,training=training)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # print(out.shape)

        # [b,c]
        out = self.avgpool(out)
        out = tf.squeeze(out,axis=[1,2])
        # print(out.shape)

        # [b,num_classes]
        out = self.fc(out)
        return out

    def build_resblock(self,filter_num,block,strides=1):
        res_block = Sequential()
        res_block.add(BasicBlock(filter_num,strides=strides))

        for _ in range(1,block):
            res_block.add(BasicBlock(filter_num,strides=1))

        return  res_block

# model = Resnet([2,2,2,2],10)
# model.build(input_shape=(None,32,32,3))
# input = tf.random.normal([1,32,32,3])
# out = model(input)
# print(out.shape)