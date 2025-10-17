import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

print("TensorFlow 버전:", tf.__version__)

# 1. 데이터 로드 및 전처리
print("\n📦 데이터 로딩 중...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 데이터 정규화 (0-255 -> 0-1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 28x28 이미지를 784차원 벡터로 변환
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

print(f"훈련 데이터 형태: {x_train.shape}")
print(f"테스트 데이터 형태: {x_test.shape}")

# 2. ANN 모델 구축
print("\n🏗️  모델 구축 중...")
model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation='relu', name='hidden_layer_1'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu', name='hidden_layer_2'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax', name='output_layer')
])

# 모델 컴파일
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 구조 출력
model.summary()

# 3. 모델 학습
print("\n🚀 모델 학습 시작...")
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.1,
    verbose=1
)

# 4. 모델 평가
print("\n📊 모델 평가 중...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\n테스트 정확도: {test_accuracy*100:.2f}%")
print(f"테스트 손실: {test_loss:.4f}")

# 5. 학습 과정 시각화
plt.figure(figsize=(12, 4))

# 정확도 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='훈련 정확도')
plt.plot(history.history['val_accuracy'], label='검증 정확도')
plt.title('모델 정확도')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='훈련 손실')
plt.plot(history.history['val_loss'], label='검증 손실')
plt.title('모델 손실')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

