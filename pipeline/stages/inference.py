import torch
import torch.nn as nn
import torch.nn.functional as F
from meshtron.mesh_tokenizer import MeshTokenizer
from pipeline.config import ConfigurationManager
from pipeline.utils.model import get_model

def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs
    
class Inference(nn.Module):
    def __init__(self, weights_path: str):
        super().__init__()
        self.tokenizer = MeshTokenizer(1024)
        model_params = ConfigurationManager.model_params()
        model_params.pad_token = self.tokenizer.PAD.item()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = get_model(model_params).to(self.device)

        try:
            state = torch.load(weights_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Weight file not found: {weights_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {e}")
        
        self.model.load_state_dict(state["model_state_dict"])


    def _apply_ordering_mask(self, logits: torch.Tensor, generated: list) -> torch.Tensor:
        """Mask logits to enforce lexicographic YZX ordering within and across faces.

        Each triangle face has 9 tokens: y1,z1,x1, y2,z2,x2, y3,z3,x3.
        Constraints:
          - Within face: v1 <= v2 <= v3 lexicographically (YZX priority)
          - Across faces: v1 of face[i] <= v1 of face[i+1] lexicographically
          - EOS only allowed at face boundaries (pos 0)
          - SOS and PAD never generated
        """
        n = len(generated)
        pos = n % 9

        min_tok = 0

        if pos == 0 and n >= 9:
            min_tok = generated[n - 9]
        elif pos == 1 and n >= 10:
            if generated[n - 1] == generated[n - 10]:
                min_tok = generated[n - 9]
        elif pos == 2 and n >= 11:
            if generated[n - 2] == generated[n - 11] and generated[n - 1] == generated[n - 10]:
                min_tok = generated[n - 9]
        elif pos == 3:
            min_tok = generated[n - 3]
        elif pos == 4:
            if generated[n - 1] == generated[n - 4]:
                min_tok = generated[n - 3]
        elif pos == 5:
            if generated[n - 2] == generated[n - 5] and generated[n - 1] == generated[n - 4]:
                min_tok = generated[n - 3]
        elif pos == 6:
            min_tok = generated[n - 3]
        elif pos == 7:
            if generated[n - 1] == generated[n - 4]:
                min_tok = generated[n - 3]
        elif pos == 8:
            if generated[n - 2] == generated[n - 5] and generated[n - 1] == generated[n - 4]:
                min_tok = generated[n - 3]

        if min_tok > 0:
            logits[:, :min_tok] = float('-inf')

        # EOS only at face boundary (pos == 0)
        if pos != 0:
            logits[:, self.tokenizer.EOS.item()] = float('-inf')

        # SOS and PAD are never valid outputs
        logits[:, self.tokenizer.SOS.item()] = float('-inf')
        logits[:, self.tokenizer.PAD.item()] = float('-inf')

        return logits

    def _safe_sample(self, probs):
        probs_cpu = probs.detach().cpu().float()
        probs_cpu = probs_cpu.clamp(min=0.0)
        probs_cpu[probs_cpu != probs_cpu] = 0.0

        prob_sum = probs_cpu.sum(dim=-1, keepdim=True)
        if prob_sum.item() == 0.0 or not torch.isfinite(prob_sum).all():
            probs_cpu = torch.ones_like(probs_cpu)
            probs_cpu = probs_cpu / probs_cpu.sum(dim=-1, keepdim=True)
        
        return torch.multinomial(probs_cpu, num_samples=1).to(probs.device)

    def _sample_next_token(self, logits, generated, temperature=1.0):
        logits = logits.detach().float()
        logits = self._apply_ordering_mask(logits, generated)
        
        if temperature != 1.0:
            logits = logits / temperature
            
        filtered_logits = top_k(logits, thres=0.9)

        if not torch.any(torch.isfinite(filtered_logits)):
            filtered_logits = logits

        probs = F.softmax(filtered_logits, dim=-1)

        return self._safe_sample(probs)

    def _autoregressive_loop(self, point_cloud, face_count, quad_ratio, window_size, temperature=1.0, use_kv_cache=True):
        self.model.eval()
        point_cloud = point_cloud.to(device=self.device)
        face_count = face_count.to(device=self.device)
        quad_ratio = quad_ratio.to(device=self.device)

        decoder_input = torch.empty(1, 9).fill_(self.tokenizer.SOS.item()).to(dtype=torch.int64, device=self.device)
        generated = []
        past_kvs = None

        while True:
            if decoder_input.size(1) >= 10377:
                break

            if use_kv_cache and past_kvs is not None:
                model_input = decoder_input[:, -1:]
            elif decoder_input.size(1) > window_size:
                model_input = decoder_input[:, -window_size:]
            else:
                model_input = decoder_input

            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    if use_kv_cache:
                        out, past_kvs = self.model(model_input, point_cloud, face_count, quad_ratio, None, past_kvs=past_kvs, use_cache=True)
                        
                        for layer_idx in range(len(past_kvs)):
                            trimmed_kvs = []
                            for block_kv in past_kvs[layer_idx]:
                                k, v = block_kv
                                if k.size(2) > window_size:
                                    k = k[:, :, -window_size:, :]
                                    v = v[:, -window_size:, :]
                                trimmed_kvs.append((k, v))
                            past_kvs[layer_idx] = trimmed_kvs
                    else:
                        out = self.model(model_input, point_cloud, face_count, quad_ratio, None)
                    
                    logits = self.model.project(out[:, -1])

            next_token = self._sample_next_token(logits, generated, temperature)
            next_token_val = next_token.item()

            yield next_token_val, decoder_input

            if next_token_val == self.tokenizer.EOS.item():
                break

            generated.append(next_token_val)
            next_token_tensor = torch.tensor([[next_token_val]], device=self.device, dtype=torch.int64)
            decoder_input = torch.cat([decoder_input, next_token_tensor], dim=1)

    def run(self, point_cloud: torch.Tensor, face_count: torch.Tensor, quad_ratio: torch.Tensor,
            window_size: int = 1017, temperature: float = 1.0, use_kv_cache: bool = True):
        for token_val, _ in self._autoregressive_loop(point_cloud, face_count, quad_ratio, window_size, temperature, use_kv_cache):
            yield self.tokenizer.dequantize(torch.tensor(token_val)).item()
            if token_val == self.tokenizer.EOS.item():
                break

    def generate(self, point_cloud: torch.Tensor, face_count: torch.Tensor, quad_ratio: torch.Tensor,
                 window_size: int = 1017, temperature: float = 1.0, use_kv_cache: bool = True):
        generated_tokens = []
        for token_val, _ in self._autoregressive_loop(point_cloud, face_count, quad_ratio, window_size, temperature, use_kv_cache):
            if token_val == self.tokenizer.EOS.item():
                break
            generated_tokens.append(token_val)
        
        if len(generated_tokens) == 0:
            return torch.zeros(0, 3)
        
        remainder = len(generated_tokens) % 3
        if remainder != 0:
            generated_tokens = generated_tokens[:-(remainder)]
        
        if len(generated_tokens) == 0:
            return torch.zeros(0, 3)
        
        tokens_tensor = torch.tensor(generated_tokens, dtype=torch.int64).unsqueeze(0)
        return self.tokenizer.decode(tokens_tensor)

def get_generator(weights_path: str):
    return Inference(weights_path=weights_path).to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
