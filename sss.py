import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io

# 读取图像并预处理
image = io.imread("input.jpg")  # 替换为您的图片路径
original_shape = image.shape
pixels = image.reshape(-1, 3)  # 关键步骤：维度变换（对应图片中的reshape函数）

# K均值聚类核心代码（对应图片中的KMMeans类使用）
def kmeans_compress(n_clusters=16):
    """使用K均值对图像颜色进行聚类压缩"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)  # 初始化模型
    kmeans.fit(pixels)  # 训练过程（对应fit方法）
    
    # 颜色数据压缩实现（对应图片中的关键语句）
    compressed_pixels = kmeans.cluster_centers_[kmeans.labels_]  # cluster_centers_和labels_属性的应用
    compressed_image = compressed_pixels.reshape(original_shape).astype(np.uint8)
    
    return compressed_image

# 执行实验并可视化结果
compressed_img = kmeans_compress(16)
plt.figure(figsize=(10,5))
plt.subplot(121), plt.imshow(image), plt.title('Original')
plt.subplot(122), plt.imshow(compressed_img), plt.title('Compressed (K=16)')
plt.axis('off')
plt.savefig('kmeans_compression.png', bbox_inches='tight')
