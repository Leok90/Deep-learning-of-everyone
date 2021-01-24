## 모두의 딥러닝

\# 최적화 방법(optimizer) 교재 p.124

확률적경사하강법(SGD)

모멘텀

네스테로프 모멘텀

아다그라드

알엠에스프롭

아담

\# **은닉층**에 **relu**를 쓰는 이유 :

오차 역전파의 경우, 시그모이드 함수를 쓰면 미분 시에 기울기 최대값이
0.3이고, 은닉층을 여러 겹 지나다 보면 기울기 소실 문제가 발생한다. 그
대안으로 찾은 것이 relu. relu는 미분하면 x\>0일 때, 기울기가 1으로
일정하므로 relu를 은닉층에 쓰면 기울기가 끝까지 남아 있게 되어, 기울기
소실 문제로부터 해방

![1](./img/1.JPG)

\# 활성화함수로 sigmoid를 쓰는 이유 :

(수학적) 선형함수의 경우 층을 아무리 깊게 하더라도 또 다른 하나의 선형
함수로 표현할 수 있고, 은닉층이 의미가 없어지므로 비선형함수를 사용해야
한다.

ex) f(x) = c \* x

f(f(f(x))) = c \* c \* c \* x = C \* x

(비수학적)선형 함수를 활성화 함수로 이용하면 섬세하게 True와 False를
분류 못해서 비선형인 sigmoid를 채택

![1](./img/2.JPG)

(수학적) 선형함수를 사용할 경우, 새로운 입력값과 회귀선과의 오차가 클
경우 입력값에 의해 예측 결과가 큰 영향을 받는 문제가 발생하고,
sigmoid함수를 사용하면 해결할 수 있다.

(비수학적)계단함수는 x=0에서 불연속이고 미분불가능하므로 sigmoid를 씀

![1](./img/3.JPG)

\# **다중분류**(iris)의 **출력층**에는 **소프트맥스**를 쓰는 이유 :

합이 1이 되는 (ex. y\[0\]=0.1, y\[1\]=0.2, y\[2\]=0.7) 여러 개의
출력으로 표현할 수 있음

![1](./img/4.JPG)

**\# 손실함수(Loss Function)** 종류

-   **평균 제곱 오차(MSE)** : 회귀(regression)으로 결과를 **예측**하고자
    > 할 때, 사용

-   **교차엔트로피(Cross Entrophy Error)** : 평균제곱오차와는 달리 오직
    > 실제 정답과의 오차만을 파악하는 손실함수이다(수식참고). 실제값이
    > 원핫 인코딩일 때, 0일 경우 y의 값이 무시되기 때문이다. **분류**
    > 문제에서 accuracy가 평균제곱오차(MSE)보다 정확하게 나온다.

\# 모델의 설정 p.131

**model = Sequential()**

**model.add(Dense(노드의 수, input_dim=독립변수의 개수,
activation=활성화 함수)**

(input_dim은 입력층에만 씀)

(첫번째 층은 은닉층이면서 입력층의 역할도 한다)

![1](./img/5.JPG)

**\# model.compile(loss=손실함수,**

**optimizer=최적화방법,**

**metrics=측정항목함수)**

\*\*손실함수와 측정항목함수의 차이

측정항목을 평가한 결과는 모델을 학습(오차역전파)시키는 데 사용되지 않음

\*\*최적화방법 p.124

**\# model.fit(X_train, Y_train, epochs=반복횟수, batch_size=한번에
입력되는 데이터 수)**

\*\*배치 처리는 컴퓨터 연산 시에 데이터 전송 I/O의 병목현상을 줄여주고,
연산 시간을 단축해준다. batch_size가 증가할수록 컴퓨터 연산 속도 증가.
I/O 전송시간만 단축해주고, 순수 CPU와 GPU 연산시간만 남게 되므로, 연산
속도 증가에 한계는 있음.
