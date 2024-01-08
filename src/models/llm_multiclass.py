import torch
from torch import nn
from transformers import AutoModel, AutoConfig, BertModel


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = #
        sum_embeddings = #
        sum_mask = #
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class CustomModel(nn.Module):
    def __init__(self,
                 llm_model_path,
                 cfg,
                 num_classes):
        super(CustomModel, self).__init__()
        # Path to model flat files
        #

        # Custom config information for the model specified in YAML file
        #

        # Number of classes / labels
        #

        # HF AutoConfig
        #

        # HF AutoModel
        #

        # Freeze Layers if Specified
        # https://discuss.huggingface.co/t/how-to-freeze-some-layers-of-bertmodel/917/4
        if cfg.freeze.apply:
            # Use modules to specify order
            modules = [self.llm_model.embeddings,
                       self.llm_model.encoder.layer[:cfg.freeze.num_layers]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = #

        # Gradient checkpointing [TODO check this]
        if self.cfg.gradient_checkpointing:
            self.llm_model.gradient_checkpointing_enable()

        # Mean Pooling
        self.pool = MeanPooling()

        # Dense layer for classification and weight initialization
        self.fc = nn.Linear(self.llm_model_config.hidden_size, num_classes)
        self._init_weights(self.fc)


    def _init_weights(self, module):
        "Initialize weights for classification weights for dense layer"
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0,
                                       std=self.llm_model_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            #
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            #
            module.weight.data.fill_(1.0)


    def forward(self, inputs):
        # Outputs from model
        llm_outputs = #
        # Apply custom pooling
        if self.cfg.mean_pooling.apply:
            feature = self.pool(last_hidden_state=#,
                                attention_mask=#)
        else:
            #CLS default pooling
            feature = #
        # Pooling
        feature = self.pool(last_hidden_state=#,
                            attention_mask=#)
        # Dense layer
        logits = self.fc(feature)
        return logits
