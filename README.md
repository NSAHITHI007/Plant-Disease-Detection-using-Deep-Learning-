# Plant-Disease-Detection-using-Deep-Learning-

🌱 Plant Disease Detection using Transfer Learning

This project focuses on early detection of plant diseases using deep learning, helping farmers and researchers improve crop management and prevent food production losses. Built a web-based application that classifies plant leaf images as healthy or diseased using state-of-the-art CNN models.

📌 Project Overview

Developed a plant disease detection system leveraging transfer learning.

Dataset: PlantVillage (AIcrowd) – 8,679 images across Apple, Corn, Tomato (healthy + 2 unhealthy classes each).

Implemented data preprocessing, resizing, and augmentation to improve model generalization.

Integrated models into a user-friendly web application for real-time disease detection.

⚙️ Tech Stack

Programming: Python (Jupyter Notebooks)

Frameworks: TensorFlow, Keras

Tools: Google Colab, GitHub, ClickUp

Dataset: PlantVillage (Zenodo, AIcrowd)

🧠 Models Implemented

VGG-16

InceptionV3

DenseNet-121

MobileNet V2

Each model was fine-tuned and evaluated using Accuracy, Loss, Precision, Recall, F1-score, and Confusion Matrix.

📊 Best Performer: InceptionV3 – 99.5% accuracy on test data.

📂 Pipeline Files

Resize.ipynb → Resizing images to 224x224.

DataAugumentation.ipynb → Data augmentation (mirroring, normalization).

Vgg16.ipynb → VGG16 model training & evaluation.

InceptionV3.ipynb → InceptionV3 model training & evaluation.

DenseNet121.ipynb → DenseNet121 model training & evaluation.

Mobilenet.ipynb → MobileNetV2 model training & evaluation.

Test.ipynb → Model testing & validation pipeline.

🚀 Results

VGG16 → 98.2%

InceptionV3 → 99.5% (Best)

DenseNet121 → 98.6%

MobileNetV2 → 98.3%

🌐 Deliverables

Trained models with high accuracy that resulted in good test accuracy for classification tasks.

Functional web-based disease detection app.

⭐️ Conclusion

✨ This project demonstrates how AI & Agriculture can work together to enhance food security by enabling early detection of plant diseases.
