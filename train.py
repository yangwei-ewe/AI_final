import os
from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 設定圖片和標籤的資料夾路徑
train_img_folder = 'train_img'
train_label_folder = 'train_label'
test_img_folder = 'test_img'
test_label_folder = 'test_label'

# 創建空的訓練和測試圖片、標籤列表
train_images = []
train_labels = []
test_images = []
test_labels = []

# 載入訓練圖片和標籤
for filename in os.listdir(train_img_folder):
    img_path = os.path.join(train_img_folder, filename)
    label_path = os.path.join(train_label_folder, f"{os.path.splitext(filename)[0]}.txt")

    # 載入圖片
    img = Image.open(img_path)
    # 將圖片轉換為NumPy數組
    img_array = np.array(img)
    # 將圖片添加到train_images列表中
    train_images.append(img_array)

    # 讀取標籤
    with open(label_path, 'r') as f:
        label = f.read().strip()
    # 將標籤添加到train_labels列表中
    train_labels.append(label)

# 載入測試圖片和標籤
for filename in os.listdir(test_img_folder):
    img_path = os.path.join(test_img_folder, filename)
    label_path = os.path.join(test_label_folder, f"{os.path.splitext(filename)[0]}.txt")

    # 載入圖片
    img = Image.open(img_path)
    # 將圖片轉換為NumPy數組
    img_array = np.array(img)
    # 將圖片添加到test_images列表中
    test_images.append(img_array)

    # 讀取標籤
    with open(label_path, 'r') as f:
        label = f.read().strip()
    # 將標籤添加到test_labels列表中
    test_labels.append(label)

print(f"訓練圖片數量: {len(train_images)}")
print(f"訓練標籤數量: {len(train_labels)}")
print(f"測試圖片數量: {len(test_images)}")
print(f"測試標籤數量: {len(test_labels)}")

# 轉換圖片和標籤為NumPy數組
train_images = np.array(train_images, dtype='float32') / 255
test_images = np.array(test_images, dtype='float32') / 255

# 轉換標籤為NumPy數組
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# 將標籤進行獨熱編碼
num_classes = len(label_encoder.classes_)
train_labels_encoded = tf.one_hot(train_labels_encoded, depth=num_classes)
test_labels_encoded = tf.one_hot(test_labels_encoded, depth=num_classes)

print(f"訓練圖片數量: {len(train_images)}")
print(f"訓練標籤數量: {len(train_labels_encoded)}")
print(f"測試圖片數量: {len(test_images)}")
print(f"測試標籤數量: {len(test_labels_encoded)}")

# 建立模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 120, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# 編譯模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 訓練模型
history = model.fit(train_images, train_labels_encoded, epochs=10, batch_size=32)

# 在測試集上評估模型
test_loss, test_accuracy = model.evaluate(test_images, test_labels_encoded)

print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# 繪製準確度和損失圖像
plt.figure(figsize=(12, 4))

# 準確度圖像
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 損失圖像
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()