import numpy as np
import random
random.seed(0)

from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler

def load_data(file_path, targets, max_length=200, min_length=30):
    messages = []
    spam_messages = []

    with open(file_path) as f:
        for l in f:
            id, message = l.split("\t", 1)
            message = [ord(x) for x in message.strip()]
            message = message[:max_length]
            message_len = len(message)
            if message_len < min_length:
                continue
            if message_len < max_length:
                message += ([0] * (max_length - message_len))
            
            if id == targets:
                spam_messages.append((1, message))
            else:
                messages.append((0, message))

    # random.shuffle(messages)
    messages = messages[:len(spam_messages)]
    all_messages = messages + spam_messages
    random.shuffle(all_messages)
    return all_messages

def create_model(embed_size=128, max_length=200, filter_sizes=(2, 3, 4, 5), filter_num=64):
    # Input
    input_ = Input(shape=(max_length,))
    # Embedding
    emb = Embedding(0xffff, embed_size)(input_)
    emb_reshape = Reshape((max_length, embed_size, 1))(emb)
    convs = []
    # Conv2D
    for filter_size in filter_sizes:
        conv = Conv2D(filter_num, (filter_size, embed_size), activation='relu')(emb_reshape)
        pool = MaxPooling2D((max_length - filter_size + 1, 1))(conv)
        convs.append(pool)
    # Concatenate
    convs_merged = Concatenate()(convs)
    reshape = Reshape((filter_num * len(filter_sizes),))(convs_merged)
    # Dense
    fc1 = Dense(64, activation="relu")(reshape)
    bn1 = BatchNormalization()(fc1)
    do1 = Dropout(0.5)(bn1)
    # 2class
    fc2 = Dense(1, activation='sigmoid')(do1)

    # Model
    model = Model(input=[input_], outputs=[fc2])
    return model
def train(inputs, targets, batch_size=64, epoch_count=100, max_length=200, model_filepath="model.h5", learning_rate=0.001):
    # 学習率を少しずつ下げる
    start = learning_rate
    stop = learning_rate * 0.01
    learning_rates = np.linspace(start, stop, epoch_count)

    # モデル
    model = create_model(max_length=max_length)
    model.summary()
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    lr_scheduler = LearningRateScheduler(lambda epoch: learning_rates[epoch], verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')

    # 学習
    model.fit(inputs, targets, epochs=epoch_count, batch_size=batch_size, verbose=1, validation_split=0.2, shuffle=True, callbacks=[lr_scheduler, early_stopping])

    # 保存
    model.save(model_filepath)

if __name__ == '__main__':
    messages = load_data('./SMSSpamCollection.txt', targets='spam')

    input_values = []
    target_values = []
    for target_value, input_value in messages:
        input_values.append(input_value)
        target_values.append(target_value)
    input_values = np.array(input_values)
    target_values = np.array(target_values)
    train(input_values, target_values, epoch_count=50)
