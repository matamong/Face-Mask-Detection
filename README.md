# Face-Mask-Detection
[Face Mask Detection using Python, Keras, OpenCV and MobileNet | Detect masks real-time video  by.
Balaji Srinivasan](https://youtu.be/Ax6P93r32KU) 을 참고한 Python, Keas, OpenCV를 이용한 실시간 마스크 착용 감지 어플리케이션입니다. <br>
**`CNN(Covolution Neural Network)`** 을 이용하여 마스크감지모델을 만들어 사용합니다. <br>
(얼굴감지모델은 미리 준비된 것을 사용하였습니다.)


<br><br>

## 과정

### 데이터 준비
-  마스크를 착용한 이미지 / 마스크를 벗은 이미지 두개의 데이터셋을 준비합니다.
- 데이터셋이 있는 경로를 리스트화 하고 `tensorflow.keras.preprocessing.image` 모듈을 이용해 이미지를 리스트화하고 정렬해서 준비합니다.
    - 이 때, `MobileNet` 을 이용하기 위해서 이미지를 `preprocess_input`를 사용하여 preprocess해줍니다.
- 원핫인코딩을 통하여 0과 1로 데이터를 구분해줍니다.
- `Overfitting` 을 방지해주기 위해서 `train/test` 를 분리해줍니다.
- 성능을 더 좋게 만들기 위하여 이미지를 `augmentation(증강)` 합니다.

### 모델제작
- 구글에서 사전 훈련된 컨볼루션 네트워크 모델 `MobileNet V2` 을 통하여 `base model(기본 모델)` 을 생성합니다.
- `base model` 의 위에 있게 될 `head model` 을 생성합니다.
    - `basemodel` 의 `output`에서부터 공간위치의 평균을 구하고(AveragePooling2D), 단일 예측(Dense)으로 변환 시키는 등 CNN을 돌려줍니다.
    - 여기서는 `Overfitting` 을 방지하기 위하여 `Dropuout` 을 한번해줍니다.
- 기본 모델을 동결하고 특징 추출기로 사용합니다.
    - `MobileNet V2` 에는 많은 층이 있으므로 동결하여 가중치가 업데이트되지않도록 하는 것이 중요합니다.
- 모델을 컴파일합니다.
    - 최적화 방식은 `Adam` 을 이용합니다.
### 모델학습
- 모델을 학습시킵니다.
- 테스트 세트를 예측해봅니다.
    - 테스트 세트의 각 이미지에 대해 해당하는 가장 큰 예측 확률을 가진 레이블의 인덱스를 찾아서 잘 보여줄 수 있게 합니다.
- 모델을 저장합니다.
- 끝으로 `pyplot` 을 이용하여 모델의 학습 및 검증 정확도 / 손실의 학습 곡선을 살펴봅니다.

* * *

### 참고 

<br>

[사전 학습된 ConvNet을 이용한 전이 학습
](https://www.tensorflow.org/tutorials/images/transfer_learning?hl=ko)

[fine-tuning-resnet-with-keras-tensorflow-and-deep-learning](https://www.pyimagesearch.com/2020/04/27/fine-tuning-resnet-with-keras-tensorflow-and-deep-learning/)

[train_test_split 모듈을 활용하여 학습과 테스트 세트 분리
 by.테디노트](https://teddylee777.github.io/scikit-learn/train-test-split)

[MobileNet이란? 쉬운 개념 설명 by.melonicedlatte](http://melonicedlatte.com/machinelearning/2019/11/01/212800.html)

[자습해도 모르겠던 딥러닝, 머리속에 인스톨 시켜드립니다. by.
Yongho Ha](https://www.slideshare.net/yongho/ss-79607172)

[Keras API - imagedatagenerator ](https://keras.io/api/preprocessing/image/#imagedatagenerator-class)

[생활코딩 Tensorflow 102 - 이미지 분류(CNN) by.이선비](https://opentutorials.org/module/5268)

[Face Mask Detection using Python, Keras, OpenCV and MobileNet | Detect masks real-time video  by.
Balaji Srinivasan](https://youtu.be/Ax6P93r32KU)