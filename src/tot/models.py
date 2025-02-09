import os
from together import Together
import backoff

completion_tokens = prompt_tokens = 0

# Together AI setup
api_key = os.getenv("TOGETHER_API_KEY", "")
if api_key != "":
    client = Together()
else:
    print("Warning: TOGETHER_API_KEY is not set")

@backoff.on_exception(backoff.expo, Exception)
def completions_with_backoff(**kwargs):
    try:
        if kwargs.get('model', '').startswith('deepseek'):
            # Set model name for Together API
            if kwargs['model'] == 'deepseek-v3':
                kwargs['model'] = 'deepseek-ai/DeepSeek-V3'
            else:
                kwargs['model'] = 'deepseek-ai/DeepSeek-R1'
            
            # Convert OpenAI format to Together format
            messages = kwargs.pop('messages', [])
            max_tokens = kwargs.pop('max_tokens', 10000)
            temperature = kwargs.pop('temperature', 0.7)
            n = kwargs.pop('n', 1)
            stop = kwargs.pop('stop', [""])
            
            responses = []
            for _ in range(n):
                response = client.chat.completions.create(
                    model=kwargs['model'],
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1,
                    stop=stop,
                    stream=False
                )
                responses.append(response)
            
            # Convert Together response format to match OpenAI's
            class Choice:
                def __init__(self, message):
                    self.message = message
            
            class Message:
                def __init__(self, content):
                    self.content = content
            
            class Usage:
                def __init__(self):
                    self.completion_tokens = 0
                    self.prompt_tokens = 0
            
            class Response:
                def __init__(self, choices, usage):
                    self.choices = choices
                    self.usage = usage
            
            choices = []
            for resp in responses:
                message = Message(resp.choices[0].message.content)
                choices.append(Choice(message))
                usage = Usage()
                usage.completion_tokens = sum([resp.usage.completion_tokens for resp in responses])
                usage.prompt_tokens = sum([resp.usage.prompt_tokens for resp in responses])
                
            return Response(choices, usage)
            
        return openai.ChatCompletion.create(**kwargs)
    except Exception as e:
        print(f"Error calling API: {str(e)}")
        raise

def gpt(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    
def chatgpt(messages, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)  # Process in batches of 20
        n -= cnt
        res = completions_with_backoff(
            model=model, 
            messages=messages, 
            temperature=temperature, 
            max_tokens=max_tokens, 
            n=cnt, 
            stop=stop
        )
        outputs.extend([choice.message.content for choice in res.choices])
        # Track token usage for both OpenAI and Deepseek
        if hasattr(res, 'usage'):
            completion_tokens += res.usage.completion_tokens
            prompt_tokens += res.usage.prompt_tokens
    return outputs
    
def gpt_usage(model="gpt-4"):
    global completion_tokens, prompt_tokens
    if model.startswith('deepseek'):
        # Together AI pricing for DeepSeek models
        if model == 'deepseek-v3':
            cost = (completion_tokens + prompt_tokens) * 0.00000125
        else:
            cost = (completion_tokens + prompt_tokens) * 0.000007
    else:
        # OpenAI pricing
        if model == "gpt-4":
            cost = completion_tokens * 0.03 + prompt_tokens * 0.01
        elif model == "gpt-3.5-turbo":
            cost = completion_tokens * 0.002 + prompt_tokens * 0.0015
        else:
            cost = 0
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
