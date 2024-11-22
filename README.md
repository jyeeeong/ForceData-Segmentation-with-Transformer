# Transformer를 활용한 시계열 데이터 분류 및 학습

### 데이터 수집
조각도의 날과 손잡이 사이에 힘/토크센서를 부착하여, 판각 동작 중 x, y, z 축의 힘 데이터 수집.

### 데이터 전처리
아래 표와 같은 기준으로 동작 segment 라벨링
| | Segments Labels | 동작 분류 기준 |
| --- | --- | --- |
| F1(0) | No force | 고무판과 조각도의 접촉이 없을 때 |
| F2(1) | Applying Force | 고무판과 조각도의 접촉 시점부터 고무판이 변형되기 직전까지 |
| F3(2) | Deforming | 고무판이 변형될 때 |
| F4(3) | Releasing Force | 변형 이후, 고무판과 조각도의 접촉이 끝나기 직전까지 |


### Transformer 모델 학습
6개의 Encoder Layer로 구성된 Transformer 모델 학습 
