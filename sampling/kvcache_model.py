import torch
from typing import Optional

from sampling.utils import norm_logits, sample
from transformers.models.bloom.modeling_bloom import BloomForCausalLM

def _debug_show_kvcache(past_key_values):
    if  past_key_values is None:
        return
    for elem in past_key_values:
        k, v = elem
        print(f"kv cache: k shape {k.shape}, v shape {v.shape}")
        break

class KVCacheModel():
    def __init__(self, model : torch.nn.Module, temperature : float = 1, top_k : int = 0, top_p : float = 0) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p


    def selective_prune_kv(self, attention_scores, threshold: float):
            """
            Prune KV cache based on attention scores.
            
            Args:
                attention_scores (torch.Tensor): Attention scores of the elements in the KV cache.
                threshold (float): Threshold below which KV elements will be pruned.
            """
            pruned_past_key_values = []
            for i, (k, v) in enumerate(self._past_key_values):
                # Assuming attention_scores has the same dimensions as k for selection purposes
                mask = attention_scores[i] > threshold
                k_pruned = k[:, :, mask, :]
                v_pruned = v[:, :, mask, :]
                pruned_past_key_values.append((k_pruned, v_pruned))
            self._past_key_values = pruned_past_key_values
            
            
    def align_and_prune_kv(self, approx_attention_scores, target_attention_scores, threshold=0.1):
        # Implement pruning logic based on attention scores
        # Example: Prune KV entries where attention scores are below the threshold
        pruned_kv = []
        for idx, (approx_score, target_score) in enumerate(zip(approx_attention_scores, target_attention_scores)):
            if target_score.mean() < threshold:
                continue  # Skip or prune low-attention elements
            pruned_kv.append(self._past_key_values[idx])  # Keep only high-attention elements
        self._past_key_values = pruned_kv

        
        
    def _forward_with_kvcache(self, input_ids : torch.Tensor, use_debug = True) -> torch.Tensor:
        if self._past_key_values is None:
            assert self._prob_history is None, f"{self._prob_history.shape}"
            # the first forward (prefill) returns the prompt's logits
            outputs = self._model(input_ids)
            self._prob_history = outputs.logits
            for i in range(self._prob_history.shape[-2]):   
                self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], self._temperature, self._top_k, self._top_p)
            self._past_key_values = outputs.past_key_values
            last_q = self._prob_history[:, -1, :]
        else:
            # return the last token's logits
            cached_len = 0
            for kv in self._past_key_values:
                k, v = kv
                cached_len = k.shape[2]
                
            last_input_id = input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)
            
            if use_debug:
                print(f"last_input_id shape {last_input_id.shape}")
                _debug_show_kvcache(self._past_key_values)
            
            outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)
            
            not_cached_q = outputs.logits
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
                
            for i in range(not_cached_q.shape[-2]):   
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)    
                
            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            
            last_q = not_cached_q[:, -1, :]
            self._past_key_values = outputs.past_key_values
            # Apply selective pruning based on attention scores
            if hasattr(outputs, 'attentions'):
                attention_scores = outputs.attentions[-1]  # Hypothetical attention scores from model
                self.selective_prune_kv(attention_scores, threshold=0.1)  # Adjust threshold as needed
        
        return last_q


    def _generate_with_kvcache(self, prefix : torch.Tensor, 
                                    gamma : int, 
                                    use_debug = False) -> torch.Tensor:
        """ forward the model gamma times

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses

        Returns:
            Torch.Tensor: prefix+generated tokens
        """
        x = prefix

        for _ in range(gamma):
            q = self._forward_with_kvcache(x, use_debug)
            next_tok = sample(q)
            x = torch.cat((x, next_tok), dim=1)
        return x

    @torch.no_grad()
    def generate(self, input : torch.Tensor, gamma : int) -> torch.Tensor:
        output = self._generate_with_kvcache(input, gamma)
        return output
    
    @torch.no_grad()
    def rollback(self, end_pos : int):
        past_key_values_trimmed = []
        assert self._past_key_values
        for kv in self._past_key_values:
            k, v = kv
            # NOTE() the indexing is specific for bloom. This won't work for other models
            # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)
            
            # Bloom is special one
            if isinstance(self._model, BloomForCausalLM):
                # k (batch * head, hidden_dim, seq); v (batch * head, seq, hidden_dim)
                k = k[:, :, :end_pos]
                v = v[:, :end_pos, :]
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)
            else:
                # k, v (batch, head, seq, hidden_dim)
                k = k[:, :, :end_pos, :]
                v = v[:, :, :end_pos, :]
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)
        
        self._past_key_values = past_key_values_trimmed
        self._prob_history = self._prob_history[:, :end_pos, :]

