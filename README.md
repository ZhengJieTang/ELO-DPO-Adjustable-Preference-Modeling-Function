
# ELO-DPO-Adjustable-Preference-Modeling-Function:innocent:

This codebase is adapted from the HALOs(https://github.com/ContextualAI/HALOs) repository with the addition of an ELO-DPO module.
This repo allows you to align LLMs with various methods, such as DPO, KTO, and an offline version of PPO.

Configs are handled by [Hydra](https://hydra.cc/), jobs are launched with [Accelerate](https://huggingface.co/docs/accelerate/en/index), and all training is done with FSDP by default.   To first SFT a model from the Hugginface repo `meta-llama/Meta-Llama-3-8B`, run a command like

```accelerate launch --config_file accelerate_config/fsdp_8gpu.yaml --main_process_port 29500 launch.py loss=sft model=llama datasets=[ultrabin] exp_name=llama3-8b_sft ++cache_dir=/data/models ++model.name_or_path=meta-llama/Meta-Llama-3-8B```

which will save a model to `/data/models/llama3-8b_sft/FINAL/`. To then align the SFT model with ELO-DPO, run a command like

```accelerate launch --config_file accelerate_config/fsdp_8gpu.yaml --main_process_port 29500 launch.py loss=elodpo model=llama datasets=[ultrabin] exp_name=llama3-8b_sft_elodpo ++cache_dir=/data/models ++model.name_or_path=meta-llama/Meta-Llama-3-8B ++model.load_from=/data/models/llama3-8b_sft/FINAL/```

which will save a model to `/data/models/llama3-8b_sft_elodpo/FINAL`.


##ELO-DPO Quickstart

1. First, clone the repo and install the dependencies. This might take a while. The package versions are important---if you change them, there is no guarantee the code will run.

   ```console
   . install.sh
   ```

2. We perform SFT training on the zephyr-7b-beta model using the Ultrafeedback dataset:
   ```console
   accelerate launch \
  --config_file accelerate_config/fsdp_8gpu.yaml \
  --main_process_port 29500 \
  launch.py \
  loss=sft \
  model=mistral \
  datasets=[ultrabin] \
  exp_name=zephyr_7b_beta_sft \
  ++cache_dir=cache_dir\
  ++model.name_or_path=model.name_or_path
   
   ```

Your model will be saved to `/cache_dir/zephyr_7b_beta_sft`

3.Modify the parameter of I in elodpo. In the /train/trainers file, the class ELODPOTrainer is:
   ```
class ELODPOTrainer(PairedPreferenceTrainer):
    def custom_logsigmoid(x, a):
        return torch.log(1 / (1 + torch.pow(a, -x)))
    def loss(self,
        batch: Dict,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        *args,
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model token-level log probabilities."""
        chosen_rewards = self.config.loss.beta * (policy_chosen_logps.sum(-1) - reference_chosen_logps.sum(-1))
        rejected_rewards = self.config.loss.beta * (policy_rejected_logps.sum(-1) - reference_rejected_logps.sum(-1))
        current_step = self.batch_counter 
        if current_step < 100:
            a = torch.e
        else:
            a = 2.5
        #losses = -F.logsigmoid(chosen_rewards - rejected_rewards) the loss of dpo

        losses = -custom_logsigmoid(chosen_rewards - rejected_rewards,a)
        return losses, chosen_rewards.detach(), rejected_rewards.detach()
   ```
By modifying the if function within it, the change of the parameter I can be modified.


4.. Now we can start ELO-DPO training a model! Let's align a Llama3-8B model on the Ultrafeedback and SHP datasets. First, setup up logging with `wandb login` and run `wandb offline` if your GPUs are not connected to the Internet. Then to launch a job:

   ```console
accelerate launch \
  --config_file accelerate_config/fsdp_8gpu.yaml \      # Accelerate configuration for 8-GPU setup
  --main_process_port 29500 \                           # Communication port for multi-GPU coordination
  launch.py \                                           # Main training script
  loss=elodpo \                                         # ELO-DPO loss function (must match config/loss filename)
  model=mistral \                                       # Model architecture (must match config/model filename)
  datasets=[ultrabin] \                                 # Ultrafeedback dataset from Hugging Face
  exp_name=zephyr_7b_beta_sft_elodpo \                  # Experiment name and save directory
  ++cache_dir=cache_dir \                               # Root directory for cached files
  ++model.name_or_path=model.name_or_path \             # Base model (Mistral-based Zephyr-7B)
  ++model.load_from=/cache_dir/zephyr_7b_beta_sft \     # SFT checkpoint
  ++wandb.enabled=false                                 # Disable Weights & Biases logging
   ```

   That's it! Your model will be saved to `/cache.dir/zephyr_7b_beta_sft_elodpo`.


5.Model Evaluation Guide
First, download the lm-evaluation-harness evaluation tool


   ```console
        git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
        cd lm-evaluation-harness
        pip install -e .
   ```
Start the assessment

   ```console
lm-eval \
    --model hf \
    --model_args pretrained=$MODEL_PATH \
    --tasks gsm8k_cot \  
    --device cuda:0 \    
    --batch_size 4       
lm_eval \
    --model hf \
    --model_args pretrained=$MODEL_PATH \
    --tasks mmlu \      
    --device cuda:0 \  
    --batch_size 1 \    
    --no_cache 
lm-eval \
    --model hf \
    --model_args pretrained=$MODEL_PATH \
    --tasks bbh \        
    --batch_size 8 \    
    --num_fewshot 3 \    
    --device cuda \      
    --output_path ./bbh_results.json 
   ```
Result：

|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |↑  |0.5899|±  |0.0039|
| - humanities     |      2|none  |      |acc   |↑  |0.5371|±  |0.0068|
| - other          |      2|none  |      |acc   |↑  |0.6640|±  |0.0082|
| - social sciences|      2|none  |      |acc   |↑  |0.6870|±  |0.0081|
| - stem           |      2|none  |      |acc   |↑  |0.5011|±  |0.0086|


|  Tasks  |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|---------|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k_cot|      3|flexible-extract|     8|exact_match|↑  |0.4882|±  |0.0138|
|         |       |strict-match    |     8|exact_match|↑  |0.4685|±  |0.0137|


|Groups|Version|  Filter  |n-shot|  Metric   |   |Value |   |Stderr|
|------|------:|----------|------|-----------|---|-----:|---|-----:|
|bbh   |      3|get-answer|      |exact_match|↑  |0.5368|±  |0.0055|

