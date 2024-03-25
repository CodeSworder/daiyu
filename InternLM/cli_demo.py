import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
#SDK模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('mjh985/daiyu')
hf_merge = os.path.join(model_dir, 'hf_merge')
model_name_or_path = hf_merge

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
model = model.eval()

system_prompt = """现在你要扮演《红楼梦中的女主角--林黛玉."""

messages = [(system_prompt, '')]

print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")

while True:
    input_text = input("User  >>> ")
    input_text = input_text.replace(' ', '')
    if input_text == "exit":
        break
    response, history = model.chat(tokenizer, input_text, history=messages)
    messages.append((input_text, response))
    print(f"robot >>> {response}")