

THEANO_FLAGS="device=gpu0,floatX=float32" python train.py \
--emb ../preprocessed_data/restaurant/w2v_embedding \
--domain restaurant \
-as 14 \
-o output_dir \
-a adam \
-b 50 \
--ortho-reg 0.1 \
--seed 1234 \
