智慧化农业器材检测数据集的训练及应用 通过训练农业机械数据集的权重 推理识别检测 无人机 收割机 拖拉机检测等进行识别，建立深度学习农业器材检测识别系统
@[toc]
以下为代码示例，仅供参考学习。

农业机械检测数据集7379张，yolo和voc标注

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/97f1391d14c54dca978099b7f2391162.png)

### 🚜 农业机械标注数据集



| 序号 | 中文名称     | 英文类别名         | 标注数量 |
|------|--------------|--------------------|----------|
| 1    | 无人机       | Drone              | 1,129    |
| 2    | 收割机       | Harvester          | 1,157    |
| 3    | 拖拉旋耕机   | Rotocultivator     | 1,170    |
| 4    | 移植机       | Transplanter       | 1,074    |
| 5    | 小型拖拉机   | Minitractor        | 725      |
| 6    | 割草机       | Mower              | 499      |
| 7    | 拖拉机       | Tractor            | 2,540    |

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1f431c383f4f4c4e911a2f86d2d94c36.png)


- **总标注数量**：8,294 条记录
- **类别数**：7 类

要使用YOLOv8模型训练农业机械检测数据集，你可以遵循以下步骤：从数据集准备与预处理开始，到划分数据集、创建YAML配置文件、环境搭建、模型训练、推理与部署、性能评估以及模型导出。以下是详细的步骤和代码示例。

### 1. 数据集准备与预处理

确保数据集已经按照YOLO格式（`class_id x_center y_center width height`）和VOC格式（XML文件）标注好，并且图片和标签文件都整理妥当。假设你的目录结构如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5999d0c0885842a5a89a379dee5ff059.png)

```
agricultural_machinery_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

如果你的数据集是VOC格式的，你需要将其转换为YOLO格式。这里提供一个简单的Python脚本用于转换：

```python
import xml.etree.ElementTree as ET
import os

def convert_voc_to_yolo(xml_file, classes):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    image_width = int(root.find('size/width').text)
    image_height = int(root.find('size/height').text)

    yolo_labels = []
    for obj in root.iter('object'):
        class_name = obj.find('name').text
        if class_name not in classes:
            continue
        class_id = classes.index(class_name)
        
        xml_box = obj.find('bndbox')
        box_xmin = float(xml_box.find('xmin').text)
        box_ymin = float(xml_box.find('ymin').text)
        box_xmax = float(xml_box.find('xmax').text)
        box_ymax = float(xml_box.find('ymax').text)
        
        # 转换为YOLO格式
        x_center = (box_xmin + box_xmax) / 2.0 / image_width
        y_center = (box_ymin + box_ymax) / 2.0 / image_height
        width = (box_xmax - box_xmin) / image_width
        height = (box_ymax - box_ymin) / image_height
        
        yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")
    
    return yolo_labels

# 使用该函数
classes = ['Drone', 'Harvester', 'Rotocultivator', 'Transplanter', 'Minitractor', 'Mower', 'Tractor']
xml_files = [os.path.join('path/to/voc_labels', f) for f in os.listdir('path/to/voc_labels') if f.endswith('.xml')]
for xml_file in xml_files:
    yolo_labels = convert_voc_to_yolo(xml_file, classes)
    txt_file = xml_file.replace('.xml', '.txt')
    with open(txt_file, 'w') as f:
        for label in yolo_labels:
            f.write(label + '\n')
```

### 2. 划分数据集

如果还没有划分好训练集、验证集和测试集，可以使用以下Python脚本进行随机划分：

```python
import os
import random
import shutil

def split_dataset(img_dir, ratio=(0.7, 0.2, 0.1)):
    files = [f for f in os.listdir(img_dir) if not f.startswith('.')]
    random.shuffle(files)
    
    total_files = len(files)
    train_size = int(total_files * ratio[0])
    val_size = int(total_files * ratio[1])
    
    train_set = files[:train_size]
    val_set = files[train_size:train_size + val_size]
    test_set = files[train_size + val_size:]
    
    return {'train': train_set, 'val': val_set, 'test': test_set}

# 使用该函数
split_result = split_dataset('agricultural_machinery_dataset/images')
for set_name, file_names in split_result.items():
    os.makedirs(f'agricultural_machinery_dataset/{set_name}/images', exist_ok=True)
    os.makedirs(f'agricultural_machinery_dataset/{set_name}/labels', exist_ok=True)
    for fn in file_names:
        shutil.move(os.path.join('agricultural_machinery_dataset/images', fn),
                    os.path.join(f'agricultural_machinery_dataset/{set_name}/images', fn))
        label_fn = fn.replace('.jpg', '.txt')  # 假设是.jpg图像和对应的.txt标签
        shutil.move(os.path.join('agricultural_machinery_dataset/labels', label_fn),
                    os.path.join(f'agricultural_machinery_dataset/{set_name}/labels', label_fn))
```

### 3. 创建YAML配置文件

创建一个名为`agricultural_machinery.yaml`的配置文件，内容如下：

```yaml
train: ./agricultural_machinery_dataset/train/images
val: ./agricultural_machinery_dataset/val/images
test: ./agricultural_machinery_dataset/test/images

nc: 7  # 类别数量
names: ['Drone', 'Harvester', 'Rotocultivator', 'Transplanter', 'Minitractor', 'Mower', 'Tractor']
```

### 4. 环境搭建

安装必要的依赖项，如PyTorch和YOLOv8：

```bash
pip install torch torchvision torchaudio
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip install -r requirements.txt
```

### 5. 模型训练

使用下面的命令开始训练：

```bash
yolo task=detect mode=train model=yolov8n.yaml data=agricultural_machinery.yaml epochs=100 imgsz=640
```

### 6. 推理与部署

完成训练后，你可以用训练好的权重对新数据进行推理：

```bash
yolo task=detect mode=predict model=runs/train/exp/weights/best.pt source=path/to/new_images
```

### 7. 性能评估

YOLOv8会自动生成一些性能指标（如mAP），你可以在训练完成后查看这些结果。此外，还可以通过绘制PR曲线等方法进一步分析。

### 8. 模型导出

将模型部署到不同的平台上，可能需要将其转换为其他格式，例如ONNX：

```bash
yolo task=detect mode=export model=runs/train/exp/weights/best.pt format=onnx
```


