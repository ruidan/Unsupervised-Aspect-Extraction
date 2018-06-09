import argparse
import logging
import numpy as np
from time import time
import utils as U
import codecs

logging.basicConfig(
                    #filename='out.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


###############################################################################################################################
## Parse arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', required=True, help="The path to the output directory")
parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=200, help="Embeddings dimension (default=200)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=50, help="Batch size (default=50)")
parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=9000, help="Vocab size. '0' means no limit (default=9000)")
parser.add_argument("-as", "--aspect-size", dest="aspect_size", type=int, metavar='<int>', default=14, help="The number of aspects specified by users (default=14)")
parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>', help="The path to the word embeddings file")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=15, help="Number of epochs (default=15)")
parser.add_argument("-n", "--neg-size", dest="neg_size", type=int, metavar='<int>', default=20, help="Number of negative instances (default=20)")
parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=0, help="Maximum allowed number of words during training. '0' means no limit (default=0)")
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1234, help="Random seed (default=1234)")
parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='adam', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=adam)")
parser.add_argument("--domain", dest="domain", type=str, metavar='<str>', default='restaurant', help="domain of the corpus {restaurant, beer}")
parser.add_argument("--ortho-reg", dest="ortho_reg", type=float, metavar='<float>', default=0.1, help="The weight of orthogonol regularizaiton (default=0.1)")

args = parser.parse_args()
out_dir = args.out_dir_path + '/' + args.domain
U.mkdir_p(out_dir)
U.print_args(args)

assert args.algorithm in {'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adam', 'adamax'}
assert args.domain in {'restaurant', 'beer'}

if args.seed > 0:
    np.random.seed(args.seed)


# ###############################################################################################################################
# ## Prepare data
# #

from keras.preprocessing import sequence
import reader as dataset

vocab, train_x, test_x, overall_maxlen = dataset.get_data(args.domain, vocab_size=args.vocab_size, maxlen=args.maxlen)
train_x = sequence.pad_sequences(train_x, maxlen=overall_maxlen)
test_x = sequence.pad_sequences(test_x, maxlen=overall_maxlen)

print 'Number of training examples: ', len(train_x)
print 'Length of vocab: ', len(vocab)

def sentence_batch_generator(data, batch_size):
    n_batch = len(data) / batch_size
    batch_count = 0
    np.random.shuffle(data)

    while True:
        if batch_count == n_batch:
            np.random.shuffle(data)
            batch_count = 0

        batch = data[batch_count*batch_size: (batch_count+1)*batch_size]
        batch_count += 1
        yield batch

def negative_batch_generator(data, batch_size, neg_size):
    data_len = data.shape[0]
    dim = data.shape[1]

    while True:
        indices = np.random.choice(data_len, batch_size * neg_size)
        samples = data[indices].reshape(batch_size, neg_size, dim)
        yield samples



###############################################################################################################################
## Optimizaer algorithm
#

from optimizers import get_optimizer

optimizer = get_optimizer(args)



###############################################################################################################################
## Building model

from model import create_model
import keras.backend as K

logger.info('  Building model')


def max_margin_loss(y_true, y_pred):
    return K.mean(y_pred)

model = create_model(args, overall_maxlen, vocab)
# freeze the word embedding layer
model.get_layer('word_emb').trainable=False
model.compile(optimizer=optimizer, loss=max_margin_loss, metrics=[max_margin_loss])


###############################################################################################################################
## Training
#
from keras.models import load_model
from tqdm import tqdm

logger.info('--------------------------------------------------------------------------------------------------------------------------')

vocab_inv = {}
for w, ind in vocab.items():
    vocab_inv[ind] = w


sen_gen = sentence_batch_generator(train_x, args.batch_size)
neg_gen = negative_batch_generator(train_x, args.batch_size, args.neg_size)
# batches_per_epoch = len(train_x) / args.batch_size
batches_per_epoch = 1000

min_loss = float('inf')
for ii in xrange(args.epochs):
    t0 = time()
    loss, max_margin_loss = 0., 0.

    for b in tqdm(xrange(batches_per_epoch)):
        sen_input = sen_gen.next()
        neg_input = neg_gen.next()

        batch_loss, batch_max_margin_loss = model.train_on_batch([sen_input, neg_input], np.ones((args.batch_size, 1)))
        loss += batch_loss / batches_per_epoch
        max_margin_loss += batch_max_margin_loss / batches_per_epoch

    tr_time = time() - t0

    if loss < min_loss:
        min_loss = loss
        word_emb = model.get_layer('word_emb').W.get_value()
        aspect_emb = model.get_layer('aspect_emb').W.get_value()
        word_emb = word_emb / np.linalg.norm(word_emb, axis=-1, keepdims=True)
        aspect_emb = aspect_emb / np.linalg.norm(aspect_emb, axis=-1, keepdims=True)
        aspect_file = codecs.open(out_dir+'/aspect.log', 'w', 'utf-8')
        model.save_weights(out_dir+'/model_param')

        for ind in range(len(aspect_emb)):
            desc = aspect_emb[ind]
            sims = word_emb.dot(desc.T)
            ordered_words = np.argsort(sims)[::-1]
            desc_list = [vocab_inv[w] for w in ordered_words[:100]]
            print 'Aspect %d:' % ind
            print desc_list
            aspect_file.write('Aspect %d:\n' % ind)
            aspect_file.write(' '.join(desc_list) + '\n\n')

    logger.info('Epoch %d, train: %is' % (ii, tr_time))
    logger.info('Total loss: %.4f, max_margin_loss: %.4f, ortho_reg: %.4f' % (loss, max_margin_loss, loss-max_margin_loss))
            
    







