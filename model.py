import tensorflow as tf
from keras import regularizers

def FCN_model(len_feature=10, len_target=3, reg=0.001):
    
    input = tf.keras.layers.Input(shape=(len_feature,))


    # Fully connected layer 1

    x = tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(reg))(input)

    # Fully connected layer 2
    x = tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(reg))(x)

    x = tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(reg))(x)
    # Fully connected layer 3
    predictions = tf.keras.layers.Dense(units=len_target, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=input, outputs=predictions)
    
    print(model.summary())
    print(f'Total number of layers: {len(model.layers)}')

    return model

if __name__ == "__main__":
    FCN_model(len_feature=10, len_target=3,reg=0.001)
    