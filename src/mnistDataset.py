import tensorflow as tf
import numpy as np
from PIL import Image

###############
#GENERATOR
###############
def create_generator(img_shape):
    noise_shape = (100,)

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(256, input_shape=noise_shape))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))

    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))

    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))

    model.add(tf.keras.layers.Dense(np.prod(img_shape), activation='tanh'))
    model.add(tf.keras.layers.Reshape(img_shape))

    model.summary()

    noise = tf.keras.layers.Input(shape=noise_shape)
    img = model(noise)

    return tf.keras.Model(noise, img)


###############
#DISCRIMINATOR
###############
def create_discriminator(img_shape):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(32, (4, 4), activation='relu', input_shape=img_shape))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(64, (4, 4), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.summary()

    input = tf.keras.layers.Input(shape=img_shape)
    img = model(input)

    return tf.keras.Model(input, img)


###############
#SAVE IMAGES
###############
def save_imgs(generator, epoch):
    r, c = 1, 1
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = generator.predict(noise)

    gen_imgs = 0.5 * gen_imgs + 0.5
    gen_imgs = 255 * gen_imgs

    gen_imgs = gen_imgs.reshape((28 ,28))

    new_im = Image.fromarray(gen_imgs)
    new_im = new_im.convert("RGB")
    new_im.save("output/"+str(epoch)+".jpg")


###############
#TRAIN
###############
def train(epochs, batch_size, generator, discriminator, combined, save_interval=50):
    (X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)

    half_batch_size = int(batch_size / 2)

    for epoch in range(epochs):
        #Train discriminator
        idx = np.random.randint(0, X_train.shape[0], half_batch_size)
        imgs = X_train[idx]

        noise = np.random.normal(0, 1, (half_batch_size, 100))
        gen_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


        #Train generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        valid_y = np.array([1] * batch_size)

        g_loss = combined.train_on_batch(noise, valid_y)

        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        if epoch % save_interval == 0:
            save_imgs(generator, epoch)


generator = create_generator((28, 28, 1))
generator.compile(loss='binary_crossentropy', optimizer='adam')

discriminator = create_discriminator((28, 28, 1))
discriminator.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
discriminator.trainable = False

z = tf.keras.layers.Input(shape=(100,))
img = generator(z)
valid = discriminator(img)

combined = tf.keras.Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer='adam')


train(2000, 20, generator, discriminator, combined)
