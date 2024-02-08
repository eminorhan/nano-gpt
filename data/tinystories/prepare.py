# saves the tinystories dataset to a binary file for training

import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

# number of workers in .map() call, good number to use is ~number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
num_proc_load_dataset = 8

tokenizer = Tokenizer.from_file("tinystories-v2-gpt4.json")
train_file = 'TinyStoriesV2-GPT4-train.txt'
val_file = 'TinyStoriesV2-GPT4-valid.txt'

data_files = {"train": train_file, "val": val_file}
dataset_args = {"keep_linebreaks": False}

# load the data
raw_datasets = load_dataset("text", data_files=data_files, num_proc=num_proc_load_dataset, **dataset_args)

# tokenize the data
def process(example):
    output = tokenizer.encode(example['text'])
    out = {'ids': output.ids, 'len': len(output.ids)}
    return out

tokenized = raw_datasets.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=num_proc,
)

# concatenate all the ids in each dataset into one large file we can use for training
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    print(f"total size of {split} split is {arr_len} tokens.")
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    dtype = np.uint16  # (can do since vocab_size < 2**16)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = 1024

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        # Batch together samples for faster write
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        # Write into mmap
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()

# train.bin is ~1.1GB, val.bin ~10.7MB
# train has ~527M tokens (527,327,687)
# val has ~5M tokens (5,325,248)

# to read the bin files later, e.g. with numpy:
# m = np.memmap('train.bin', dtype=np.uint16, mode='r')