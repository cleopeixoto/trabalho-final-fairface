import os
from enum import Enum
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Enum de arquiteturas
class ModelType(Enum):
    MOBILENET = "mobilenet"
    RESNET = "resnet"
    EFFICIENTNET = "efficientnet"

# Mapa de modelos
model_map = {
    ModelType.MOBILENET: MobileNetV2,
    ModelType.RESNET: ResNet50,
    ModelType.EFFICIENTNET: EfficientNetB0
}

# Caminhos locais para os pesos pré-treinados (include_top=False), versionados e localizados em ../weights
# URLs oficiais para referência:
# MobileNetV2: https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5
# ResNet50: https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
# EfficientNetB0: https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5

WEIGHTS_PATHS = {
    ModelType.MOBILENET: "../weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5",
    ModelType.RESNET: "../weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
    ModelType.EFFICIENTNET: "../weights/efficientnetb0_notop.h5"
}

# Função para criar modelo
def build_model(model_type, input_shape=(224, 224), num_classes=9, dropout_rate=0.5, fine_tune_at=None):
    ModelClass = model_map[model_type]

    base_model = ModelClass(
        input_shape=input_shape + (3,),
        include_top=False,
        weights=WEIGHTS_PATHS[model_type]
    )
    base_model.trainable = False

    if model_type == ModelType.MOBILENET and fine_tune_at is not None:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Função para criar geradores
def build_generators(df_train, df_val, data_dir, target_size=(224, 224), batch_size=32, preprocessing_function=None, shuffle=True):
    datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)

    train_gen = datagen.flow_from_dataframe(
        dataframe=df_train,
        directory=data_dir,
        x_col='file',
        y_col='label',
        target_size=target_size,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=shuffle,
        seed=42
    )

    val_gen = datagen.flow_from_dataframe(
        dataframe=df_val,
        directory=data_dir,
        x_col='file',
        y_col='label',
        target_size=target_size,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False
    )

    return train_gen, val_gen

# Função para plotar histórico
def plot_history(history, title_suffix=""):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title(f'Acurácia {title_suffix}')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title(f'Loss {title_suffix}')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()