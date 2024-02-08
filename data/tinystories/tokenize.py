from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# initialize BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# initialize BPE trainer
trainer = BpeTrainer(special_tokens=["[UNK]","<|endoftext|>"], vocab_size=32768, min_frequency=10)

# files to train on
files = ["TinyStoriesV2-GPT4-train.txt", "TinyStoriesV2-GPT4-valid.txt"]

# train tokenizer
tokenizer.train(files, trainer)

# save trained tokenizer
tokenizer.save("tinystories-v2-gpt4.json")