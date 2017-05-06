import os
import pickle

#---------------------------------------------------------------------
# Data Parameters
#---------------------------------------------------------------------
max_vocab_size = {"en" : 200000, "fr" : 200000}

# Special vocabulary symbols - we always put them at the start.
PAD = b"_PAD"
GO = b"_GO"
EOS = b"_EOS"
UNK = b"_UNK"
START_VOCAB = [PAD, GO, EOS, UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

NO_ATTN = 0
SOFT_ATTN = 1

# for appending post fix to output
attn_post = ["NO_ATTN", "SOFT_ATTN"]

DATASET = ["IWSLT15", "CALLHOME_WORD", "INUKTITUT"][1]

EXP_NAME_PREFIX="baseline"

if DATASET == "IWSLT15":
#-----------------------------------------------------------------
# IWSLT15 configuration
#-----------------------------------------------------------------
    NUM_SENTENCES = 133000

    # en to vi
    VI_TO_EN = False
    # vi to en
    # VI_TO_EN = True

    print("IWSLT Vietnamese-English dataset configuration")
    # subtitles data
    input_dir = "../../corpora/iwslt15/"
    # use 90% of the data for training
    NUM_TRAINING_SENTENCES = (NUM_SENTENCES * 90) // 100
    # remaining (max 10%) left to be used for dev. For training, we limit the dev size to 500 to speed up perplexity and Bleu computation
    NUM_DEV_SENTENCES = 200
    NUM_TEST_SENTENCES = 1268
    BATCH_SIZE = 16
    # A total of 7 buckets, with a length range of 3 each, giving total
    # BUCKET_WIDTH * NUM_BUCKETS = 21 for e.g.
    BUCKET_WIDTH = 5
    NUM_BUCKETS = 50
    MAX_PREDICT_LEN = BUCKET_WIDTH*NUM_BUCKETS

    tokens_fname = os.path.join(input_dir, "tokens.list")
    vocab_path = os.path.join(input_dir, "vocab.dict")
    w2i_path = os.path.join(input_dir, "w2i.dict")
    i2w_path = os.path.join(input_dir, "i2w.dict")

    if VI_TO_EN:
        print("translating VI to EN")
        model_dir = "vi_en_model"

        text_fname = {"en": os.path.join(input_dir, "train.en"), "fr": os.path.join(input_dir, "train.vi")}

        test_fname = {"en": os.path.join(input_dir, "tst2013.en"), "fr": os.path.join(input_dir, "tst2013.vi")}

        EXP_NAME= "{0:s}_iwslt15_vi_en".format(EXP_NAME_PREFIX)

        bucket_data_fname = os.path.join(model_dir, "buckets_{0:d}.list")

        w2i = pickle.load(open(w2i_path, "rb"))
        i2w = pickle.load(open(i2w_path, "rb"))
        vocab = pickle.load(open(vocab_path, "rb"))
        vocab_size_en = min(len(i2w["en"]), max_vocab_size["en"])
        vocab_size_fr = min(len(i2w["fr"]), max_vocab_size["fr"])
        print("vocab size, en={0:d}, fr={1:d}".format(vocab_size_en, vocab_size_fr))

    else:
        print("translating EN to VI")
        model_dir = "en_vi_model"

        text_fname = {"en": os.path.join(input_dir, "train.vi"), "fr": os.path.join(input_dir, "train.en")}

        test_fname = {"en": os.path.join(input_dir, "tst2013.vi"), "fr": os.path.join(input_dir, "tst2013.en")}

        EXP_NAME= "{0:s}_iwslt15_en_vi".format(EXP_NAME_PREFIX)

        bucket_data_fname = os.path.join(model_dir, "buckets_{0:d}.list")

        w2i = {"en": {}, "fr": {}}
        i2w = {"en": {}, "fr": {}}
        vocab = {"en": {}, "fr": {}}

        w2i_temp = pickle.load(open(w2i_path, "rb"))
        w2i["en"] = w2i_temp["fr"]
        w2i["fr"] = w2i_temp["en"]

        i2w_temp = pickle.load(open(i2w_path, "rb"))
        i2w["en"] = i2w_temp["fr"]
        i2w["fr"] = i2w_temp["en"]

        vocab_temp = pickle.load(open(vocab_path, "rb"))
        vocab["en"] = vocab_temp["fr"]
        vocab["fr"] = vocab_temp["en"]

        vocab_size_en = min(len(i2w["en"]), max_vocab_size["en"])
        vocab_size_fr = min(len(i2w["fr"]), max_vocab_size["fr"])
        print("vocab size, en={0:d}, fr={1:d}".format(vocab_size_en, vocab_size_fr))
#-----------------------------------------------------------------
elif DATASET == "CALLHOME_WORD":
#-----------------------------------------------------------------
# CALLHOME word level nmt configuration
#-----------------------------------------------------------------

    print("callhome es-en word level configuration")
    input_dir = "../../corpora/callhome/uttr_fa_vad_wavs"

    NUM_SENTENCES = 17394
    # use 90% of the data for training
    NUM_TRAINING_SENTENCES = 13137
    # remaining (max 10%) left to be used for dev. For training, we limit the dev size to 500 to speed up perplexity and Bleu computation
    NUM_DEV_SENTENCES = 2476
    NUM_TEST_SENTENCES = 1781
    BATCH_SIZE = 40
    # A total of 11 buckets, with a length range of 7 each, giving total
    # BUCKET_WIDTH * NUM_BUCKETS = 77 for e.g.
    BUCKET_WIDTH = 3
    NUM_BUCKETS = 14
    MAX_PREDICT_LEN = BUCKET_WIDTH*NUM_BUCKETS

    tokens_fname = os.path.join(input_dir, "tokens.list")
    vocab_path = os.path.join(input_dir, "vocab.dict")
    w2i_path = os.path.join(input_dir, "w2i.dict")
    i2w_path = os.path.join(input_dir, "i2w.dict")

    print("translating es to en")
    model_dir = "es_en_model_adam_eps6_h300"

    text_fname = {"en": os.path.join(input_dir, "train.en"), "fr": os.path.join(input_dir, "train.es")}

    dev_fname = {"en": os.path.join(input_dir, "dev.en"), "fr": os.path.join(input_dir, "dev.es")}

    test_fname = {"en": os.path.join(input_dir, "test.en"), "fr": os.path.join(input_dir, "test.es")}

    EXP_NAME= "{0:s}_callhome_es_en".format(EXP_NAME_PREFIX)

    bucket_data_fname = os.path.join(model_dir, "buckets_{0:d}.list")

    if os.path.exists(w2i_path):
        w2i = pickle.load(open(w2i_path, "rb"))
        i2w = pickle.load(open(i2w_path, "rb"))
        vocab = pickle.load(open(vocab_path, "rb"))
        vocab_size_en = min(len(i2w["en"]), max_vocab_size["en"])
        vocab_size_fr = min(len(i2w["fr"]), max_vocab_size["fr"])
        print("vocab size, en={0:d}, fr={1:d}".format(vocab_size_en, vocab_size_fr))
#-----------------------------------------------------------------

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(input_dir):
    print("Input folder not found".format(input_dir))

#---------------------------------------------------------------------
# Model Parameters
#---------------------------------------------------------------------
num_layers_enc = 4
num_layers_dec = 4
use_attn = SOFT_ATTN
#---------------------------------------------------------------------
# !! NOTE !!
#---------------------------------------------------------------------
hidden_units = 512

load_existing_model = True
create_buckets_flag = False
#---------------------------------------------------------------------
# Training Parameters
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# Training EPOCHS
#---------------------------------------------------------------------
# if 0 - will only load a previously saved model if it exists
#---------------------------------------------------------------------
NUM_EPOCHS = 0

# Change the dev set to include all the sentences not used for training, instead of 500
# Using all during training impacts timing
# if NUM_EPOCHS == 0:
#     NUM_DEV_SENTENCES = NUM_SENTENCES-NUM_TRAINING_SENTENCES

#---------------------------------------------------------------------
# GPU/CPU
#---------------------------------------------------------------------
# if >= 0, use GPU, if negative use CPU
gpuid = 1
#---------------------------------------------------------------------
# Log file details
#---------------------------------------------------------------------
name_to_log = "{0:d}sen_{1:d}-{2:d}layers_{3:d}units_{4:s}_{5:s}".format(
                                                            NUM_SENTENCES,
                                                            num_layers_enc,
                                                            num_layers_dec,
                                                            hidden_units,
                                                            EXP_NAME,
                                                            attn_post[use_attn])

log_train_fil_name = os.path.join(model_dir, "train_{0:s}.log".format(name_to_log))
log_dev_fil_name = os.path.join(model_dir, "dev_{0:s}.log".format(name_to_log))
model_fil = os.path.join(model_dir, "seq2seq_{0:s}.model".format(name_to_log))
#---------------------------------------------------------------------
