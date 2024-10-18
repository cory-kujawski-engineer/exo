import numpy as np
from llama_cpp import Llama
from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard
from typing import Optional, Tuple
from exo.download.shard_download import ShardDownloader
from concurrent.futures import ThreadPoolExecutor
import asyncio

class LlamaCppInferenceEngine(InferenceEngine):
  def __init__(self, shard_downloader: ShardDownloader):
    self.shard = None
    self.shard_downloader = shard_downloader
    self.executor = ThreadPoolExecutor(max_workers=1)

  async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, image_str: Optional[str] = None, inference_state: Optional[str] = None) -> (np.ndarray, str, bool):
    await self.ensure_shard(shard)
    input_ids = await asyncio.get_event_loop().run_in_executor(self.executor, self.tokenizer.encode, prompt)
    output_data: np.ndarray = np.array(await asyncio.get_event_loop().run_in_executor(self.executor, self.model.generate, input_ids))
    return output_data, "", output_data.size == 1 and output_data.item() == self.tokenizer.eos_token_id

  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool):
    await self.ensure_shard(shard)
    output_data: np.ndarray = np.array(await asyncio.get_event_loop().run_in_executor(self.executor, self.model.generate, input_data))
    return output_data, "", output_data.size == 1 and output_data.item() == self.tokenizer.eos_token_id

  async def ensure_shard(self, shard: Shard):
    if self.shard == shard:
      return

    model_path = await self.shard_downloader.ensure_shard(shard)

    if self.shard != shard:
      self.model = await asyncio.get_event_loop().run_in_executor(self.executor, Llama, model_path)
      self.tokenizer = self.model.tokenizer
      self.shard = shard
