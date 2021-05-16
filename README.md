# 대회 개요

COVID-19의 확산으로 우리나라는 물론 전 세계 사람들은 경제적, 생산적인 활동에 많은 제약을 가지게 되었습니다. 우리나라는 COVID-19 확산 방지를 위해 사회적 거리 두기를 단계적으로 시행하는 등의 많은 노력을 하고 있습니다. 과거 높은 사망률을 가진 사스(SARS)나 에볼라(Ebola)와는 달리 COVID-19의 치사율은 오히려 비교적 낮은 편에 속합니다. 그럼에도 불구하고, 이렇게 오랜 기간 동안 우리를 괴롭히고 있는 근본적인 이유는 바로 COVID-19의 강력한 전염력 때문입니다.

감염자의 입, 호흡기로부터 나오는 비말, 침 등으로 인해 다른 사람에게 쉽게 전파가 될 수 있기 때문에 감염 확산 방지를 위해 무엇보다 중요한 것은 모든 사람이 마스크로 코와 입을 가려서 혹시 모를 감염자로부터의 전파 경로를 원천 차단하는 것입니다. 이를 위해 공공 장소에 있는 사람들은 반드시 마스크를 착용해야 할 필요가 있으며, 무엇 보다도 코와 입을 완전히 가릴 수 있도록 올바르게 착용하는 것이 중요합니다. 하지만 넓은 공공장소에서 모든 사람들의 올바른 마스크 착용 상태를 검사하기 위해서는 추가적인 인적자원이 필요할 것입니다.

따라서, 우리는 카메라로 비춰진 사람 얼굴 이미지 만으로 이 사람이 마스크를 쓰고 있는지, 쓰지 않았는지, 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 시스템이 필요합니다. 이 시스템이 공공장소 입구에 갖춰져 있다면 적은 인적자원으로도 충분히 검사가 가능할 것입니다.

# 데이터
마스크를 착용하는 건 COIVD-19의 확산을 방지하는데 중요한 역할을 합니다. 제공되는 이 데이터셋은 사람이 마스크를 착용하였는지 판별하는 모델을 학습할 수 있게 해줍니다. 모든 데이터셋은 아시아인 남녀로 구성되어 있고 나이는 20대부터 70대까지 다양하게 분포하고 있습니다. 간략한 통계는 다음과 같습니다.

전체 사람 명 수 : 4,500

한 사람당 사진의 개수: 7 [마스크 착용 5장, 이상하게 착용(코스크, 턱스크) 1장, 미착용 1장]

이미지 크기: (384, 512)

학습 데이터와 평가 데이터를 구분하기 위해 임의로 섞어서 분할하였습니다. 60%의 사람들은 학습 데이터셋으로 활용되고, 20%는 public 테스트셋, 그리고 20%는 private 테스트셋으로 사용됩니다.

위와 같은 데이터를 이용해서 아래에 있는 18개의 클래스를 분류(Classification)해야 합니다.

![Image](/readme_image/p1_class_detail.png)

# 평가 방법
Submission 파일에 대한 평가는 F1 Score 를 통해 진행합니다.

F1 Score 에 대해 간략히 설명 드리면 다음과 같습니다.

![Image](/readme_image/p1_f1score_detail.png)
 
# 최종 결과
Public LB - F1 score 0.7229, 112/224

Private LB - F1 score 0.7095, 112/224

# 대회 동안 시도한 것들

#### Toggle(▶)을 누르면 세부 설명을 볼 수 있습니다. 그리고 Bold 텍스트로 되어있는 것은 최종 결과에 적용한 항목들 입니다.


## 1. 데이터 처리

크게 분류의 카테고리를 마스크(mask), 성별(gender), 나이대(age)로 나눌 수 있었다.

여기서 age에 대한 클래스 분포가 (30세 미만):(30세 이상 60세 미만):(60세 이상) = 6:6:1 정도로 매우 불균형해서 age 불균형에 대한 처리가 이번 stage의 핵심이라고 생각했다.

불균형 데이터를 처리하는 방법으로는 불균형한 상태를 그대로 이용하는 방법과 class 별 균형을 맞춘 후 이용하는 방법이 있는데 성능지표가 accuracy가 아니라 F1 score이기 때문에 균형을 맞춰야겠다고 생각했다.

방법은 크게 oversampling과 undersampling이 있는데, undersampling을 하면 age데이터가 너무 적어져서 학습이 거의 진행되지 않을 것이라 생각해서 oversampling 방법을 이용하기로 하였다.

<details>
<summary><b>augmentation을 이용한 oversampling을 적용</b></summary>
<div markdown="1">       
<br>

<pre><code>가장 좋은 oversampling방법은 주어진 데이터셋과 유사한 외부 데이터를 이용하는 것이라 생각했지만, 이에 맞는 데이터셋을 찾기가 어려워서 외부 데이터는 이용하지 않았다.

같은 이미지를 복사하므로 중복된 이미지가 존재하는데, 확률적으로 다양한 augmentation을 적용한다면 완전히 다른 이미지를 사용하는 것만큼은 아니지만 이와 비슷한 역할을 할 수 있을 것이라 생각했다.

확률적으로 다양한 augmentation을 적용한다면 완전히 다른 이미지를 사용하는 것만큼은 아니지만 이와 비슷한 역할을 할 수 있을 것이라 생각했다.

따라서 클래스가 바뀌지 않는 선에서 albumentation에서의 HorizontalFlip, ColorJitter, Rotate(최대 10도)을 적용하였다.
</code></pre>

</div>
</details>

<details>
<summary><b>(400, 400)으로 Resize를 한 후에 (224, 224)로 CenterCrop을 하였고, 0~1 범위를 갖도록 Normalize 적용</b></summary>
<div markdown="1">       
<br>

<pre><code>거의 모든 이미지에서 얼굴이 중앙에 위치하여서 얼굴 외의 부분들을 제거해주기 위해서 CenterCrop을 이용하였다.

그리고 학습이 빠르게 이루어지도록 해서 다양한 실험을 해보기 위해 이미지의 크기를 (224, 224)로 축소하였다.
</code></pre>

</div>
</details>

<details>
<summary><b>train, validation 데이터셋에 같은 사람이 겹치지 않도록 사람에 대해 분리를 한 후에 mask, gender, age 각각에 대해 비율에 맞게 나눔</b></summary>
<div markdown="1">       
<br>

<pre><code>한 사람당 7개의 이미지가 존재하는데 train, validation 데이터셋에 겹치는 사람이 존재하면 data leakage 문제가 발생할 수 있으므로 이를 방지하였다.
</code></pre>

</div>
</details>

<br>
다만 지금 생각해보면 같은 이미지를 복사한 후에 augmentation을 통해 oversampling을 한다는 것은 위험한 접근 방식이라는 생각이 든다.

실제로 피어세션 등에서 다른 캠퍼분들의 의견을 들어보면 augmentation을 하지 않는 캠퍼분들이 더 많았다.


## 2. 모델

모델은 이미지 분류에서 많이 사용되고, 다양한 시도를 하기위해 빠른 학습을 할 수 있는 모델인 ResNet50과 EfficientNet B0를 사용하였다.

그리고 3가지 카테고리(mask, gender, age)를 예측해야 하므로 다음과 같이 3가지 모델형식을 생각했다.

- 하나의 모델에서 하나의 output을 이용해서 총 18개의 클래스 전체에 대해서 예측</li>
- 하나의 모델에서 3개의 output을 이용해서 mask, gender, age 각각에 대해서 예측</li>
- <b>mask, gender, age에 해당하는 모델을 각각 만들어서 예측</b></li>

3번째 형식을 사용한 첫 번째 이유는 3가지 카테고리를 합친 총 18개 전체 클래스에 대한 분포와 따로따로 보았을 때의 분포를 확인해보면

18개 전체 클래스에 대해서 (가장 많은 count): (가장 적은 count)=50:1 정도이고, mask에서는 5:1, gender에서는 약 3:2, age는 약 6:1 정도였다.

이를 보고 mask, gender, age 이미지를 따로따로 다루는 것이 불균형이 더 적을 것이라 생각했다.

그리고 이번 stage에서는 age를 처리하는 것이 주요한 방법일 것이라 생각했으므로, 모델을 각각 만들면 age에 대한 추가적인 작업들을 쉽게 할 수 있을 것이라 생각했다.


## 3. loss
loss function의 경우 다음과 같이 3가지를 시도해보았다.

<details>
<summary><b>CrossEntropy Loss</b></summary>
<div markdown="1">       
<br>

<pre><code>Classification task에 적용하기 위해 사용하였다.
</code></pre>

</div>
</details>

<details>
<summary>Focal Loss</summary>
<div markdown="1">       
<br>

<pre><code>데이터 불균형이 있으므로 잘 분류가 되지 않는 데이터들이 있을 것이고, 이러한 데이터에 더 높은 가중치를 주면 학습이 더 잘 이루어질 것이라 생각하여 사용하였다.

mask, gender에 대해서는 분류가 잘 이루어진다고 생각하여 age에만 Focal Loss를 적용하였다.

하지만 오히려 Public LB F1 score가 하락하였다.

점수가 하락한 이유에 대해서는 age에 대한 validation F1 score가 0.80~0.82정도 나왔는데, 

Focal Loss가 효과를 볼 정도로 분류가 어렵다고 볼 수는 없었기 때문인 것 같다.
</code></pre>

</div>
</details>

<details>
<summary>LabelSmoothing Loss</summary>
<div markdown="1">       
<br>

<pre><code>LabelSmoothing Loss의 경우 일반화 성능이 좋으므로 test 성능이 더 높아질 것이라 기대하여 사용하였다.

그리고 age의 경우 57~59세 등 50대 후반과 60세를 이미지 상 거의 구별하기가 어려운데 어떻게 예측하느냐에 따라 클래스의 값이 바뀌는 경우가 어려움이 있었다.

이 경우 구별하기 어려우므로 CrossEntropy Loss를 통해 하나의 클래스에 대한 confidence가 100%가 되는 것 보다 90%, 80%가 되도록 하면 일반화 성능이 더 좋아질 것이라 생각했다.

Public LB F1 score가 0.016정도 상승하였으나, 하락했다고 잘못 판단해서 최종 결과에 사용하지 않았다.
</code></pre>

</div>
</details>


## 4. Inference

<details>
<summary><b>TTA</b></summary>
<div markdown="1">       
<br>

<pre><code>Inference단계에서 성능을 높이기 위해 TTA(Test Time Augmentation)을 적용하였다. 총 6가지 augmentation을 적용하였다.

1. (224, 224)로 CenterCrop

2. (300, 300)으로 scale up하여 CenterCrop

3. Rotate(최대 10도)

4. HorizontalFlip

5. ColorJitter

6. 2~5를 모두 추가.

TTA는 예측이 잘 이루어지는 데이터에 적용하면 오히려 성능이 떨어질 수 있다는 내용이 있었다.

mask, gender에 대한 validation F1 score는 0.95~0.98 정도로 측정되어 TTA를 하지 않아도 예측이 잘 이루어진다고 판단하였다. 반면에 age에 대한 validation F1 score는 약 0.80~0.82 정도로 예상대로 mask, gender에 비하면 예측 결과가 좋지 않았다.

실제로 age에 대해서만 TTA를 적용하면 Public LB F1 score가 약 0.07 상승했지만 mask, gender, age에 모두 TTA를 적용하면 오히려 떨어졌다.
</code></pre>

</div>
</details>

<details>
<summary><b>K-Fold Cross Validation</b></summary>
<div markdown="1">       
<br>

<pre><code>Inference에서 성능을 높이기 위해 K-Fold Cross validation(KFold=5)를 적용하였다.

TTA와 마찬가지로 mask, gender에 대해서는 충분히 성능이 높다고 판단하여 age에 대해서만 K-Fold Cross Validation을 적용하였고, Public LB F1 score가 약 0.09 상승하였다.

그리고 mask, gender, age에 대해 모두 K-Fold Cross Validation을 적용할 때는 오히려 Public LB F1 score가 하락하였다.
</code></pre>

</div>
</details>


## 5. 시도했지만 잘 되지 않은 것들

<details>
<summary>Model Ensemble</summary>
<div markdown="1">       
<br>

<pre><code>EfficientNet B0과 ResNet50에 대하여 Soft voting방식으로 앙상블을 적용하였으나 성능이 하락하였다. 

그 이유를 생각해보니 먼저 앙상블 하려는 모델의 성능이 유사한 경우 앙상블을 했을 때 성능이 상승할 가능성이 높은데, 

ResNet50이 EfficientNet B0보다 Public LB score가 약 0.02정도 낮아서 오히려 앙상블을 했을 때 성능이 하락했을 것이라 생각했다.

그리고 앙상블은 구조가 다른 모델끼리 앙상블을 진행할 때 효과가 더 좋은데, 

EfficientNet과 ResNet 모두 CNN기반 모델이라 구조 상 차이가 크지 않기 때문에 앙상블이 효과가 없었다고 생각했다.
</code></pre>

</div>
</details>

<details>
<summary>외부 데이터 이용</summary>
<div markdown="1">       
<br>

<pre><code>주어진 데이터와 유사하다고 판단할 수 있는 얼굴 이미지들을 이용하였다. 약 500장 정도를 추가해봤는데, Public LB F1 score가 오히려 하락했다.

그 이유는 우선 외부 데이터사 주어진 데이터와 비슷하게 이미지 중앙에 얼굴이 있지 않은 경우도 많았고,

주어진 이미지의 배경은 대부분 실내이지만 외부 데이터는 실내, 외부, 색칠(예를들어 빨간색 등)이 되어있어 주어진 데이터와 다른 부분이 상당히 많기 때문이라 생각했다.
</code></pre>

</div>
</details>


## 6. 하이퍼 파라미터 & 그 외
- Seed : 43
- Batch size : 64
- Epochs : 10(Validation F1 score를 이용한 Early stopping 적용, Validation F1 score가 가장 높은 모델만 저장)
- Learning rate : 4e-4(mask) / 4e-4(gender) / 1e-4(age)
- Optimizer : Adam
- Scheduler : CosineAnnealingWarmRestarts 


## 7. 아쉬웠던 점들
1. Stage초반 seed값을 고정하지 않아서 정확한 실험 결과를 얻을 수 없었다.

2. LabelSmoothing Loss를 적용할 경우 Public LB F1 score가 약 0.016정도 상승했는데, 점수를 하락했다고 잘못 봐서 최종 결과에 적용하지 못한 점이 아쉬웠다.

3. Jupyter Notebook 환경이 아닌 IDE를 사용해보고 싶었는데, 못 해봐서 아쉬웠다.

4. 충분한 학습 전략들이 다 세워지면 이미지 사이즈를 (224, 224)에서 더 키우고, 규모가 더 큰 모델을 사용하여 성능을 더 높이고 싶었는데 못 해봐서 아쉬웠다.
