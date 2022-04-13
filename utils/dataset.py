import os
import torch
from torch.utils.data import Dataset
import numpy as np

TRAIN_CBR_FILE_PATH = "./cbr_sim_matrix_train.npy"
TEST_CBR_FILE_PATH = "./cbr_sim_matrix_test.npy"

class TokenizedDataset(Dataset):
    # TODO: A unified structure-representation.
    def __init__(self, args, training_args, tokenizer, seq2seq_dataset, train=False, train_dataset=None):
        self.args = args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.seq2seq_dataset = seq2seq_dataset
        self.my_flag_var = True
        self.conv_sep = " || "
        if train:
            self.CBR_MAT = np.load(open(TRAIN_CBR_FILE_PATH, 'rb'))
        else:
            self.CBR_MAT = np.load(open(TEST_CBR_FILE_PATH, 'rb'))
        if train_dataset is None:
            self.train_dataset = self.seq2seq_dataset
        else:
            self.train_dataset = train_dataset
        self.db_dict = {}
        for i in range(len(self.seq2seq_dataset)):
            d = self.seq2seq_dataset[i]
            db_id = d['db_id']
            if self.db_dict.get(db_id) is None:
                self.db_dict[db_id] = []
            self.db_dict[db_id].append(i)

    def __getitem__(self, index):
        raw_item = self.seq2seq_dataset[index]
        if raw_item["text_in"]:
            ###################
            # With text input #
            ###################
            if self.conv_sep in raw_item["text_in"]:
                ##################
                # Conversational #
                ##################
                # TODO (commented by Chen): the context part roughly follows the implementation of CoSQL by Tianbao.
                # text_in = "[utt n] || [utt n-1] | [utt n-2] | ..."
                index = raw_item["text_in"].index(self.conv_sep)
                if self.args.model.knowledge_usage == 'concatenate' or self.args.model.knowledge_usage is None:
                    # seq_in  = "[utt n] ; structured knowledge: struct_in ; context: [utt n-1] | [utt n-2] | ..."
                    seq_in = "{} ; structured knowledge: {} ; context: {}".format(raw_item["text_in"][:index],
                                                                                  raw_item["struct_in"],
                                                                                  raw_item["text_in"][index + len(self.conv_sep):])
                elif self.args.model.knowledge_usage == 'separate':
                    # seq_in  = "[utt n] ; context: [utt n-1] | [utt n-2] | ..."
                    seq_in = "{} ; context: {}".format(raw_item["text_in"][:index],
                                                       raw_item["text_in"][index + len(self.conv_sep):])
                else:
                    raise ValueError()
            else:
                ######################
                # Non-conversational #
                ######################
                if self.args.model.knowledge_usage == 'concatenate' or self.args.model.knowledge_usage is None:
                    # seq_in  = "text_in ; structured knowledge: struct_in"
                    seq_in = "{} ; structured knowledge: {}".format(raw_item["text_in"], raw_item["struct_in"])
                elif self.args.model.knowledge_usage == 'separate':
                    # seq_in  = "text_in"
                    seq_in = raw_item["text_in"]
                else:
                    raise ValueError()
        else:
            ######################
            # Without text input #
            ######################
            if self.args.model.knowledge_usage == 'concatenate':
                # seq_in  = "structured knowledge: struct_in"
                seq_in = "structured knowledge: {}".format(raw_item["struct_in"])
            elif self.args.model.knowledge_usage == 'separate':
                # seq_in  = ""
                seq_in = ""
            else:
                raise ValueError()

        # Concatenate description.
        if self.args.model.use_description and self.args.model.concatenate_description:
            seq_in = "{} ; {}".format(raw_item["description"], seq_in)

        

        tokenized_question_and_schemas = self.tokenizer(
            seq_in,
            padding="max_length",
            truncation=True,
            max_length=self.training_args.input_max_length,
            # We found that set it as large as possible can boost the performance significantly
            # , meanwhile, due to the t5 uses a relative position coding, we need to manually
            # assign the max input length into some large numbers, instead of using the "max_model_length"
            # ,which the default is 512, which will hurt the performance a lot.
        )

        db_id = raw_item['db_id']
        if self.db_dict.get(db_id) is not None:  # During training, no issues
            nn_list = self.db_dict[db_id]
            nn_list_index = self.CBR_MAT[index][nn_list].argmin()
            nn_index = self.db_dict[db_id][nn_list_index]
        else:                                    # During eval, db_id may not be available
            nn_index = self.CBR_MAT[index].argmin()
        nn_ted = self.CBR_MAT[index][nn_index].argmin()
        # nn_item = self.train_dataset[nn_index]
        # nn_question = nn_item['question']
        # nn_query    = nn_item['query']
        # seq_in = "{} ; {} ; {} ; {}".format(raw_item["description"], seq_in, nn_question, nn_query)
        
        # tokenized_question_and_schemas_cbr_concat = self.tokenizer(
        #     seq_in,
        #     padding="max_length",
        #     truncation=True,
        #     max_length=self.training_args.input_max_length,
        #     # We found that set it as large as possible can boost the performance significantly
        #     # , meanwhile, due to the t5 uses a relative position coding, we need to manually
        #     # assign the max input length into some large numbers, instead of using the "max_model_length"
        #     # ,which the default is 512, which will hurt the performance a lot.
        # )

        tokenized_inferred = self.tokenizer(
            raw_item["seq_out"],
            padding="max_length",
            truncation=True,
            max_length=self.training_args.generation_max_length,
            # We set the max_length of "seq_out" during training is the same with the one in inference.
        )

        tokenized_inferred_input_ids = torch.LongTensor(tokenized_inferred.data["input_ids"])
        # Here -100 will let the model not to compute the loss of the padding tokens.
        tokenized_inferred_input_ids[tokenized_inferred_input_ids == self.tokenizer.pad_token_id] = -100

        item = {
            'idx': index,
            'nn_index': nn_index,
            'nn_ted': nn_ted,
            'input_ids': torch.LongTensor(tokenized_question_and_schemas.data["input_ids"]),
            'attention_mask': torch.LongTensor(tokenized_question_and_schemas.data["attention_mask"]),
            'labels': tokenized_inferred_input_ids,
        }
        # Add task name.
        if 'task_id' in raw_item:
            item['task_ids'] = raw_item['task_id']

        # Separate description tokenization.
        if self.args.model.use_description and self.args.model.map_description:
            tokenized_description = self.tokenizer(raw_item["description"],
                                                   padding="max_length",
                                                   truncation=True,
                                                   max_length=self.args.dataset.description_max_length,
                                                   )
            item['description_input_ids'] = torch.LongTensor(tokenized_description.data["input_ids"])
            item['description_attention_mask'] = torch.LongTensor(tokenized_description.data["attention_mask"])

        # Separate knowledge tokenization.
        if self.args.model.knowledge_usage == 'separate':
            tokenized_knowledge = self.tokenizer(raw_item["struct_in"],
                                                 padding="max_length",
                                                 truncation=True,
                                                 max_length=self.tokenizer.model_max_length,
                                                 )
            item['knowledge_input_ids'] = torch.LongTensor(tokenized_knowledge.data["input_ids"])
            item['knowledge_attention_mask'] = torch.LongTensor(tokenized_knowledge.data["attention_mask"])

        return item

    def __len__(self):
        return len(self.seq2seq_dataset)
