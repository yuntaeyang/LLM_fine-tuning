import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from transformers import TextStreamer, GenerationConfig

#base_model_name='yanolja/KoSOLAR-10.7B-v0.1-deprecated'
#peft_model_name = '/home/yuntaeyang_0629/taeyang_2024/CDSI/keyword/sllm_ft/checkpoints/KoSOLAR-10.7B-v0.1-deprecated_lora_ft/checkpoint-13000'
base_model_name="yuntaeyang/KoSOLAR-10.7B-keword-v1.0" #
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto",torch_dtype=torch.float16)


#peft_model = PeftModel.from_pretrained(base_model, peft_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

#peft_model=peft_model.merge_and_unload()

prompt= '''다음은 문서 목록입니다. 텍스트의 주제를 설명하는 상위 키워드를 쉼표로 구분하여 추출하십시오.

문서:
- 대부분의 문화에서 전통적인 식단은 주로 약간의 고기를 위에 얹은 식물성 식단이었지만, 산업 스타일의 고기 생산과 공장 농업의 증가로 고기는 주식이 되었습니다.

키워드 : 전통식단, 식물성, 육류, 산업식단, 공장식단, 주식, 문화식단

문서:
- 웹사이트에는 배송에 며칠밖에 걸리지 않는다고 나와 있지만 여전히 제 것을 받지 못했습니다.

키워드 : 웹사이트, 배송, 언급, 기간, 미수신, 대기, 주문이행

문서:
- 지도 학습은 예제 입력-출력 쌍을 기반으로 입력을 출력에 매핑하는 기능을 학습하는 기계 학습 작업입니다. 
훈련 예제 세트로 구성된 레이블이 지정된 훈련 데이터에서 함수를 추론합니다. 
지도 학습에서 각 예제는 입력 개체(일반적으로 벡터)와 원하는 출력 값(감독 신호라고도 함)으로 구성된 쌍입니다. 
지도 학습 알고리즘은 훈련 데이터를 분석하고 추론된 함수를 생성하며, 이는 새로운 예제를 매핑하는 데 사용될 수 있습니다. 
최적의 시나리오는 알고리즘이 보이지 않는 인스턴스에 대한 클래스 레이블을 올바르게 결정할 수 있도록 합니다. 이는 학습 알고리즘이 '합리적인' 방식으로 훈련 데이터에서 보이지 않는 상황으로 일반화해야 합니다(유도 바이어스 참조).

키워드:'''

generation_config = GenerationConfig(
        temperature=0.1,
        # top_p=0.8,
        # top_k=100,
        max_new_tokens=100,
        repetiton_penalty=1.2,
        early_stopping=True,
        do_sample=True,
    )

gened = base_model.generate(
        **tokenizer(
            prompt,
            return_tensors='pt',
            return_token_type_ids=False
        ).to('cuda'),
        generation_config=generation_config,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        #streamer=streamer,
    )
result_str = tokenizer.decode(gened[0])
print(result_str.split("키워드 :")[-1])