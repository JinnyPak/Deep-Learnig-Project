# 머신러닝과 딥러닝을 활용한 크라우드 펀딩 성공 예측

# 프로젝트 개요
크라우드 펀딩 성공 예측 모델을 구축한 프로젝트로, 펀딩 런칭 전 데이터(메타데이터,이미지 데이터, 텍스트 데이터)로 머신러닝 및 딥러닝 모델을 구축하였습니다. 3가지 데이터를 전처리를 진행하고
1. 3가지 형태의 데이터 각각의 분류 모델
2. 3가지 형태의 테이터를 결합한 분류 모델

총 4개의 모델을 구축하였습니다. 
<b>이 문서에서는 텍스트 데이터로 성공/실패를 분류하는 분석 파트를 설명합니다.</b>






# Dataset 설명
국제적으로 유명한 크라우드 펀딩 플랫폼 Kickstarter 의 데이터로 모델을 구축. 데이터 출처는 webrobots ( https://webrobots.io/ ) 가 제공하는 Free
Kickstarter Dataset. webrobots 측 에서 한달에 한번 한달치의 메타 데이터를 크롤링한 csv 파일로, 본 프로젝트에서는 2015년~2019년 5년치 데이터셋으로 분석

1. 메타 데이터 (일부분)

  
  |    | category_name    | location_name   | country_displayable_name   |   date_launched_month | is_starrable   | state|   
  |---:|:-----------------|:----------------|:---------------------------|----------------------:|:---------------|:-----------|
  |  0 | Children's Books | Ste.-Maxime     | France                     |                     6 | False          | successful |
  |  1 | Graphic Novels   | Minneapolis     | the United States          |                     8 | False          | successful |
  |  2 | Apps             | London          | the United Kingdom         |                     6 | False          | failed     |
  |  3 | Graphic Novels   | New York        | the United States          |                    10 | False          | successful |
  |  4 | Drama            | Sydney          | Australia                  |                     8 | False          | failed     |

<br>


  - 주요 컬럼 설명
    - 총 컬럼 개수 : 38개
      - state : 펀딩 프로젝트 상태 (성공함, 실패함, 취소됨, 현재 진행중, 강제 중단됨) Target(종속변수-성광, 실패)로 사용할 변수
      - source_url : Kickstarter 웹사이트에서의 펀딩 유형/분류 별 링크. 해당링크에 접속하여 크롤링함.
      - category : 펀딩(제품/서비스) 유형
      - country : 국가 코드
      - created_at : 펀딩 프로젝트가 만들어진 시간 (단위 : milliseconds)






이미지와 텍스트 데이터는 메타 데이터의 source_url (각 펀딩의 URL) 로 접속하여 크롤링하여 수집


2. 이미지 데이터
  - 펀딩 프로젝트에 사용된 메인 사진
  - 웹사이트 메뉴나 펀딩 페이지 상에서 가장 먼저 노출되어 펀딩을 표현하는 데이터

3. 텍스트 데이터
  - 총 컬럼 개수 : 4개
    - name : 펀딩 제목
    - blurb : 펀딩의 부제목
    - content : 펀딩에 대해 상세히 설명한 본문 부분
    - risk_challenge : 펀딩의 위험요소에 대한 설명 부분
  - 본 프로젝트에서는 본문과 Risk Challenge 2가지를 사용하여 텍스트 모델 구축




||name|	blurb|	content|	risk_challenge|
|---:|--------:|:-----------------|:----------------|:---------------------------|
|0|	Strange Wit, an original graphic novel about J...	|The true biography of the historical figure, w...	|['Meet Jane Bowles: incredible author, inspiri...	|The main obstacles this book faces are the siz...
1	|Living Life Tiny	|Educating my community about self-sufficiency ...|	["I've always been passionate about efficiency...	|The biggest challenge in completing this proje...








# 프레임워크
<b>Python 3.7
1. 크롤링
```python
from bs4 import BeautifulSoup
import requests
import time
from selenium import webdriver
from selenium.webdriver import Chrome
from selenium.webdriver import ChromeOptions
```

2. 전처리
```python
import NLTK
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

```
3. 모델
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense,LSTM,Dropout
```

# 분석과정

1. 데이터 수집
2. 데이터 전처리
3. 텍스트 임베딩
4. 모델 구축
5. 평가
6. 예측



- 추가적인 설명은 문서 하단에 주석을 달았습니다.

--------------------------------

## 1. 데이터 수집

크롤링에 대한 자세한 과정은 아래 깃허브 링크로 접속하여 확인가능합니다.

```python
## 크롤링 코드 일부분
driver = Chrome(options=options)
content_list = []
for idx, url in enumerate( ks_2016['url_project'][7000:8000], start=7000):
    driver.get(url)
    time.sleep(random.uniform(8,20))
    req = driver.page_source
    soup = BeautifulSoup(req, 'lxml')
    try:
        content_tag = soup.select_one('div.rte__content')
        contents = content_tag.select('p')
        contents_collected = []
        for c in contents:
            content = c.get_text().strip()
            contents_collected.append(content)
        try:
            risk_challenge_tag = soup.select_one('div#risksAndChallenges')
            risk_challenge_list = risk_challenge_tag.select('p.js-risks-text.text-preline')
            for rc in risk_challenge_list:
                risk_challenge = rc.get_text().strip()
        except:
            risk_challenge=" " 
```

(https://github.com/JinnyPak/Deep-Learning-Project/blob/master/deeplearning_project/kickstarter_text_crawling.ipynb) 


---------------------------
## 2. 데이터 전처리 
전처리 과정은 데이터 결측치 처리와 독립변수/종속변수 전처리 2가지 과정을 진행<br>

(1) 데이터 결측치 처리와 데이터 병합
- 본문내용과 Risk Challenge 컬럼에서 
Null값이거나 공백, "[]"만 있는 행을 제거 (주석 1 참고)
- 본문과 Risk Challenge 병합
- 메타데이터에 있는 목표변수 성공/실패 여부 컬럼과 병합




```python
## 데이터 불러오기
text_dataset=pd.read_csv('text_dataset_2015_2019.csv')#텍스트 데이터
meta_2015_2019=pd.read_csv('meta_dataset_2015_2019.csv') #메타데이터
```

```python
## 텍스트 컬럼과 목표변수인 성공여부 컬럼 병합
ks_text_state=pd.concat([text_dataset[['content','risk_challenge']],meta_2015_2019['state']],axis=1)
```

```python
## Dataframe 형태 확인
ks_text_state
```


  <div>
  <style scoped>
      .dataframe tbody tr th:only-of-type {
          vertical-align: middle;
      }

      .dataframe tbody tr th {
          vertical-align: top;
      }

      .dataframe thead th {
          text-align: right;
      }
  </style>
  <table border="1" class="dataframe">
    <thead>
      <tr style="text-align: right;">
        <th></th>
        <th>content</th>
        <th>risk_challenge</th>
        <th>state</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th>0</th>
        <td>['Meet Jane Bowles: incredible author, inspiri...</td>
        <td>The main obstacles this book faces are the siz...</td>
        <td>successful</td>
      </tr>
      <tr>
        <th>1</th>
        <td>["I've always been passionate about efficiency...</td>
        <td>The biggest challenge in completing this proje...</td>
        <td>successful</td>
      </tr>
      <tr>
        <th>2</th>
        <td>['Billet Dice', 'Billet Dice are made from bil...</td>
        <td>The international shipping is a worry but I wa...</td>
        <td>successful</td>
      </tr>
      <tr>
        <th>3</th>
        <td>['With this film we want to entertain you in a...</td>
        <td>We will be using the money to prevent any prob...</td>
        <td>failed</td>
      </tr>
      <tr>
        <th>4</th>
        <td>['The Splash Drone is a fully waterproof quad ...</td>
        <td>We have spent countless hours and many months ...</td>
        <td>successful</td>
      </tr>
      <tr>
        <th>...</th>
        <td>...</td>
        <td>...</td>
        <td>...</td>
      </tr>

    </tbody>
  </table>
  <p>151061 rows × 3 columns</p>
  </div>



```python
## 결측치 개수 확인
ks_text_state.isna().sum()
```

    content            27
    risk_challenge    189
    state               0
    dtype: int64




```python
## 본문 컬럼을 기준으로 결측치가 있는 행 걸러내기
null_index=ks_text_state[ks_text_state.content.isna()].index.tolist()#결측치 행의 index
ks_text_state=ks_text_state.drop(null_index) #결측치 행 제거
```



```python
## "[]"만 있는 행 걸러내기
cont_null_index=[]
for i in ks_text_state.index:
    if ks_text_state.content.loc[i]=="[]":
        cont_null_index.append(i)

ks_text_state.loc[cont_null_index]
ks_text_state=ks_text_state.drop(cont_null_index)
```
본문 컬럼 외에 Risk Challenge 컬럼에 있는 결측치 행 처리는 " "로 대체.
```python
risk_isna=ks_text_state[ks_text_state.risk_challenge.isna()].index.tolist()
ks_text_state.risk_challenge=ks_text_state.risk_challenge.fillna(" ")
```


본문과 Risk Challenge 컬럼을 병합한 새로운 컬럼 생성
```python
ks_text_state['all_text']=ks_text_state['content']+ks_text_state['risk_challenge']
```
독립변수와 목표변수 설정
```python
y=ks_text_state['state']
X=ks_text_state['all_text']
X.shape,y.shape
```

    ((151061,), (151061,))

(2) 텍스트 토큰화, 목표변수 Label Encoding

```python
#y label encoding

#Series를 받아서 라벨인코딩 처리하는 함수
from sklearn.preprocessing import LabelEncoder

def encoding(x):
    col_dict={}
    le=LabelEncoder()
    le.fit(x)
#변환
    label_x=le.transform(x)
    col_dict[x.name]=le.classes_
    return label_x
y_enc=encoding(y)
# 성공 - 1
# 실패 - 0

```


```python
# X(텍스트) 전처리
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# 전처리 함수
def text_preprocessing(document):
    # 소문자 변환
    document = document.lower()
    # 특수문자 제거
    document = document.replace('\\xa0',' ')
    document = document.replace('•\\t',' ') 
    pattern = '[{}]'.format(string.punctuation)
    document = re.sub(pattern, ' ', document)
    # stopword 제거, stemming
    #기존의 stopword에는 조동사 'may'가 포함되지 않아 추가함.
    sw = stopwords.words('english')+['may']
    word_token=nltk.word_tokenize(document)
    stemmer = PorterStemmer()

    result_token=[ stemmer.stem(word) for word in word_token if word not in sw]
    #문자열로 변환 후 반환
    return ' '.join(result_token)
    
```

```python
## 전처리 함수 적용하여 각 펀딩 내용을 담은 문자열 리스트를 생성
X_list=list(X.values)
text = [text_preprocessing(x) for x in X_list]
```


## 4. 텍스트 데이터 벡터화/ 임베딩

텍스트의 특성을 더 잘 반영하도록 사전훈련없이 토큰화와 임베딩을 진행하여 
모델을 학습시키는 방향으로 수정 (주석 2)

파라미터들은 '케라스 창시자에게 배우는 딥러닝' 도서를 참고하여 정의
```python
max_features=10000
maxlen=500
batch_size=32
embedding_dim=300

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(preprocessed_all_text)
vocab_size = len(tokenizer.word_index) + 1
print('단어 집합의 크기 : %d' % vocab_size)
```
    단어 집합의 크기 : 600997
    
```python
sequences=tokenizer.texts_to_sequences(preprocessed_all_text) data=pad_sequences(sequences,maxlen=maxlen)
labels=np.array(y_enc)

data.shape,labels.shape
```
    (151061, 500),(151061,)


```python
# train/test data 분리
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, labels,stratify=labels)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
```
    ((113295, 500), (37766, 500), (113295,), (37766,))

## 5. 모델 구축
본 프로젝트에서는 타겟변수인 성공여부의 데이터 분포가 불균형합니다. 
(성공사례가 15만건중 113295건으로 실패사례보다 더 많은 건수임)
성공예측의  Accuracy 점수는 당연히 높게 나올 수 밖에 없습니다. 
그러므로 Recall 점수 개선에 초점을 맞추어 모델을 수정하였습니다.

LSTM은 sequence 데이터( 시계열 데이터,텍스트 데이터) 같은  경우에 적합한 모델이므로
Embedding layer와 함께 모델에 포함시켰습니다.
처음에 구축한 모델은 Embedding, LSTM Layer 1개만을 사용하였으나 평가지표가 좋지않아 (Recall : 약 0.7) 
Dense Layer를 3개를 추가하고, 과적합 방지를 위하여 Dropout도 추가하여 
최종 모델은 아래와 같이 수정하였습니다.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense,LSTM,Dropout

model = Sequential()
model.add(Embedding(max_features, embedding_dim, input_length=maxlen))
model.add(LSTM(32))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 500, 300)          3000000   
    _________________________________________________________________
    lstm (LSTM)                  (None, 32)                42624     
    _________________________________________________________________
    dense (Dense)                (None, 32)                1056      
    _________________________________________________________________
    dropout (Dropout)            (None, 32)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 32)                1056      
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 32)                0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 32)                1056      
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 32)                0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 33        
    =================================================================
    Total params: 3,045,825
    Trainable params: 3,045,825
    Non-trainable params: 0
    _________________________________________________________________
    

이진분류 모델이므로 loss는 'binary_crossentropy'로, 평가지표는 'accuracy','AUC','Recall','Precision'로 설정하였습니다.

```python
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy','AUC','Recall','Precision'])

```

    Train on 90636 samples, validate on 22659 samples
    Epoch 1/10
    90636/90636 [==============================] - 789s 9ms/sample - loss: 0.5440 - accuracy: 0.7493 - AUC: 0.7936 - Recall: 0.8531 - Precision: 0.7631 - val_loss: 0.5092 - val_accuracy: 0.7733 - val_AUC: 0.8428 - val_Recall: 0.8650 - val_Precision: 0.7844
    Epoch 2/10
    90636/90636 [==============================] - 793s 9ms/sample - loss: 0.4950 - accuracy: 0.7844 - AUC: 0.8355 - Recall: 0.8625 - Precision: 0.7993 - val_loss: 0.4988 - val_accuracy: 0.7568 - val_AUC: 0.8449 - val_Recall: 0.9520 - val_Precision: 0.7301
    Epoch 3/10
    90636/90636 [==============================] - 797s 9ms/sample - loss: 0.4702 - accuracy: 0.7952 - AUC: 0.8548 - Recall: 0.8798 - Precision: 0.8028 - val_loss: 0.4996 - val_accuracy: 0.7908 - val_AUC: 0.8585 - val_Recall: 0.8238 - val_Precision: 0.8308
    Epoch 4/10
    90636/90636 [==============================] - 788s 9ms/sample - loss: 0.4548 - accuracy: 0.8036 - AUC: 0.8662 - Recall: 0.8874 - Precision: 0.8087 - val_loss: 0.4596 - val_accuracy: 0.7841 - val_AUC: 0.8619 - val_Recall: 0.9223 - val_Precision: 0.7688
    Epoch 5/10
    90636/90636 [==============================] - 783s 9ms/sample - loss: 0.4409 - accuracy: 0.8099 - AUC: 0.8754 - Recall: 0.8961 - Precision: 0.8113 - val_loss: 0.4636 - val_accuracy: 0.7943 - val_AUC: 0.8659 - val_Recall: 0.8337 - val_Precision: 0.8290
    Epoch 6/10
    90636/90636 [==============================] - 792s 9ms/sample - loss: 0.4288 - accuracy: 0.8172 - AUC: 0.8835 - Recall: 0.9043 - Precision: 0.8155 - val_loss: 0.5237 - val_accuracy: 0.7783 - val_AUC: 0.8610 - val_Recall: 0.9426 - val_Precision: 0.7542
    Epoch 7/10
    90636/90636 [==============================] - 795s 9ms/sample - loss: 0.4170 - accuracy: 0.8237 - AUC: 0.8909 - Recall: 0.9172 - Precision: 0.8160 - val_loss: 0.5347 - val_accuracy: 0.7792 - val_AUC: 0.8439 - val_Recall: 0.8483 - val_Precision: 0.8005
    Epoch 8/10
    90636/90636 [==============================] - 797s 9ms/sample - loss: 0.4077 - accuracy: 0.8285 - AUC: 0.8973 - Recall: 0.9234 - Precision: 0.8182 - val_loss: 0.5175 - val_accuracy: 0.7879 - val_AUC: 0.8556 - val_Recall: 0.8359 - val_Precision: 0.8189
    Epoch 9/10
    90636/90636 [==============================] - 807s 9ms/sample - loss: 0.3933 - accuracy: 0.8360 - AUC: 0.9057 - Recall: 0.9300 - Precision: 0.8235 - val_loss: 0.5019 - val_accuracy: 0.7921 - val_AUC: 0.8588 - val_Recall: 0.8952 - val_Precision: 0.7906
    Epoch 10/10
    90636/90636 [==============================] - 803s 9ms/sample - loss: 0.3836 - accuracy: 0.8415 - AUC: 0.9119 - Recall: 0.9340 - Precision: 0.8278 - val_loss: 0.5833 - val_accuracy: 0.7910 - val_AUC: 0.8561 - val_Recall: 0.8794 - val_Precision: 0.7976
    

```python
model.evaluate(X_test, y_test)
```

    37766/37766 [==============================] - 77s 2ms/sample - loss: 0.5847 - accuracy: 0.7868 - AUC: 0.8530 - Recall: 0.8688 - Precision: 0.7986
    


```python
## 모델 저장
model.save('lstm_all_text.h5')
```

## 6. 저장한 모델 로드하여 예측하기
2019년 이후에 오픈된 펀딩 중 성공여부가 정해진 펀딩의 본문으로 테스트를 했습니다. 해당 펀딩은 모금에 성공한 펀딩입니다.

    테스트할 펀딩 내용 일부 :
    "Ptolus: Monte Cook's City by the Spire, the classic and much beloved campaign setting, returns for 5e and Cypher System fans. We're reissuing the whole, deluxe product in two versions, one for each of these popular, critically acclaimed game systems.(Can't read the text in the graphics above? Click any of the images for graphics-free text.)
    The Ptolus Player's Guide  
    In addition to the two new versions of Ptolus: Monte Cook’s City by the Spire, this campaign will also fund the Ptolus Player’s Guide in print and PDF. (생략) "

    링크 :
    (https://www.kickstarter.com/projects/montecookgames/ptolus-monte-cooks-city-by-the-spire)

```python
from tensorflow.keras.models import load_model
new_model = load_model('lstm_all_text.h5')
```

```python
test1_pre = [text_preprocessing(x) for x in test1]
test1_pre
```
```python
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(test1_pre)
vocab_size = len(tokenizer.word_index) + 1

sequences=tokenizer.texts_to_sequences(test1_pre)
data=pad_sequences(sequences,maxlen=maxlen)
data.shape
```
    (1, 500)



```python
pred_cls = new_model.predict_classes(data)
pred_cls
```
    array([[1]])


```python
pred_proba=new_model.predict_proba(data)
pred_proba
```
    array([[0.6899058]], dtype=float32)
<!-- 테스트 데이터로 사용한 펀딩은 실제로 투자유치에 성공한 펀딩이었습니다.<br><br> -->


---------------------------
## 7. 예측모델을 활용한 웹 어플리케이션
- Django 로 구현한 예측 웹 어플리케이션은 아래 링크로 접속하여 확인가능합니다.
https://github.com/JinnyPak/django_prediction_service
<br><br>

# 8. 한계점과 보완점
- 추후에 유지/보수나 적용이 용이하도록 모델을 생성하는 함수 코드 정리 필요


# 참고 서적 및 논문
- 'Success Prediction on Crowdfunding with Multimodal DeepLearning'  (ChaoranCheng, FeiTan, XiuruiHou and ZhiWei )   <br>
- '케라스 창시자에게 배우는 딥러닝'(프랑소와 숄레) <br>
- '탠서플로와 머신러닝의 시작하는 자연어처리' (전창욱,최태균,조중현)<br><br>

# 주석
1) 본문 내용이 공백이거나 Null값인 경우가 전체 건수 중 100건미만으로
데이터 손실에 큰 영향을 주지 않을 것으로 판단하여 해당 행을 제거하였습니다.
또한 간혹 Risk Challenge가 없는 경우가 있습니다.
처음 분석 방향으로는 Risk Challenge의 포함 여부가 
프로젝트 성공여부에 영향을 미치는 변수일 수도 있다고 생각했지만
실제로 Risk Challenge가 공백이거나 Null값인 펀딩의 건수는 2400여건 정도로 전체 데이터 셋에 비해
매우 적어 더미변수로 사용하지 않았습니다.
공백, Null값인 Risk Challenge 행도 본문과 병합시키는 방향으로 수정하여 분석을 진행하였습니다.
<br><br>
2) 이전에 시도했던 학습방법 중에는 사전훈련된 임베딩(Glove)을 사용한 모델학습도 있습니다.
학습속도를 더 빠르게 개선하고자 사용하였으나 사전훈련되지 않은 모델과 속도 차이가 거의 없었고 평가지표가 좋지 않았습니다.
