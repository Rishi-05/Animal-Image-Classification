import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
# setting the environment variable to disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

data_dir = "C:\\Users\\RAMESH\\Desktop\\codes\\Python Projects\\photo detector\\Image Dataset for Support Vector Machine"  
classes = ["cats", "dogs", "human", "horses"] 
image_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2,
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    class_mode="categorical",
    classes=classes,
    batch_size=batch_size,
    subset="training",
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    class_mode="categorical",
    classes=classes,
    batch_size=batch_size,
    subset="validation",
)

# creating the model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(len(classes), activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

# training the model
epochs = 20
model.fit(train_gen, epochs=epochs, validation_data=val_gen)

model.save("./model.h5")
print(model.summary())
