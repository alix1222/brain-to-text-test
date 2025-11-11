from omegaconf import OmegaConf
from transformer_trainer import BrainToTextDecoder_Trainer

args = OmegaConf.load('transformer_args.yaml')
trainer = BrainToTextDecoder_Trainer(args)
metrics = trainer.train()