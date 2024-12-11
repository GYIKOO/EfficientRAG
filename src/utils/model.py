import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Literal

from tenacity import retry, stop_after_attempt
from tqdm.rich import tqdm_rich

from language_models import LanguageModel


@retry(stop=stop_after_attempt(3), reraise=False, retry_error_callback=lambda x: None)
def ask_model(
    model: LanguageModel,
    prompt: str,
    system_msg: str = None,
    type: Literal["json", "text"] = "json",
    check_if_valid: Callable = None,
    sleep: bool = True,
    mode: Literal["chat", "completion"] = "chat",
) -> dict:
    if sleep:
        sleep_time = random.uniform(1.0, 3.0)
        time.sleep(sleep_time)
    if mode == "chat":
        result = model.chat(prompt, system_msg, json_mode=(type == "json"))
        # print('chat result', result)
    elif mode == "completion":
        result = model.complete(prompt)
    # print('result', result)
    parser = get_type_parser(type)
    info = parser(result)
    print('after parse', info)
    if check_if_valid is not None and not check_if_valid(info):
        print(f"Invalid response {info}")
        raise ValueError("Invalid response")
    return info


def ask_model_in_parallel(
    model: LanguageModel,
    prompts: list[str],
    system_msg: str = None,
    type: Literal["json", "text"] = "json",
    check_if_valid_list: list[Callable] = None,
    max_workers: int = 4,
    desc: str = "Processing...",
    verbose=True,
    mode: Literal["chat", "completion"] = "chat",
):
    if max_workers == -1:
        max_workers = len(prompts)
    assert max_workers >= 1, "max_workers should be greater than or equal to 1"
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if check_if_valid_list is None:
            check_if_valid_list = [None] * len(prompts)
        assert len(prompts) == len(
            check_if_valid_list
        ), "Length of prompts and check_if_valid_list should be the same"
        tasks = {
            executor.submit(
                ask_model, model, prompt, system_msg, type, check_if_valid, mode
            ): idx
            for idx, (prompt, check_if_valid) in enumerate(
                zip(prompts, check_if_valid_list)
            )
        }
        results = []
        for future in tqdm_rich(
            as_completed(tasks), total=len(tasks), desc=desc, disable=not verbose
        ):
            task_id = tasks[future]
            try:
                result = future.result()
                results.append((task_id, result))
            finally:
                ...
        results = [result[1] for result in sorted(results, key=lambda r: r[0])]
        return results


def get_type_parser(type: str) -> Callable:
    def json_parser(result: str):
        # pattern = r"```json(.*?)```"
        pattern = r"{.*?}"
        matches = re.findall(pattern, result, re.DOTALL)
        # print('original result', result)
        if matches:
            result = matches[0].strip()
        if '```json' in result:
            result = result.replace('```json', '').replace('```', '')

        # 在尝试 json.loads() 之前，确保结果字符串格式正确
        result_fixed = result.replace(',"decomposed_questions",', ',"decomposed_questions": {')

        try:
            # 修复常见的 JSON 格式错误
            # Replace property names with double quotes, but preserve single quotes in text content
            # 首先替换属性名的单引号为双引号
            result = re.sub(r"([{,]\s*)\'([^\']+?)\'(\s*:)", r'\1"\2"\3', result_fixed)
            # 然后替换字符串值的外层单引号为双引号，但保留内部的单引号
            result = re.sub(r':\s*\'([^\']*(?:\'[^\']*)*?)\'', r': "\1"', result)
            # 3. 处理数组，支持数字和带引号的字符串
            def process_array(match):
                if not match.group(1).strip():
                    return ': []'
                # 分割时同时处理有无空格的情况
                elements = [x.strip() for x in re.split(r'\s*,\s*', match.group(1)) if x.strip()]
                processed_elements = []
                for elem in elements:
                    # 移除可能存在的引号（单引号或双引号）
                    elem = elem.strip("'\"")
                    # 添加双引号
                    processed_elements.append(f'"{elem}"')
                return ': [' + ','.join(processed_elements) + ']'
            
            # 使用更精确的正则表达式匹配数组
            result = re.sub(r':\s*\[([\d\s,\'\"]*?)\]', process_array, result)
            json.loads(result)
        except Exception as e:
            print('final result', result)
            print('here')
            print(e)
        return json.loads(result)
    def text_parser(result: str):
        return result

    if type == "json":
        return json_parser
    elif type == "text":
        return text_parser
    else:
        raise ValueError(f"Unsupported type: {type}")
