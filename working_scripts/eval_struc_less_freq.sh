!/bin/bash
export CUDA_VISIBLE_DEVICES=1
BASE=/home/nikhil/Projects/E2E-NLG-Challenge/E2E/data
MODEL=struc_train
STEP=_step_5000
MODEL_DIR=/home/nikhil/Projects/E2E-NLG-Challenge/E2E/model
EVAL_DATA_DIR=$BASE/evaluation_dataset
OUTPUT=$BASE/output_$MODEL-less-freq
testset-struc-less-freq-source.tok
OPENNMT=/home/nikhil/OpenNMT-py
EVAL_SCRIPT_DIR=/home/nikhil/Projects/e2e-metrics
GPU=0
TOTAL_GPUS=1


source activate Python2
# measure scores run on python 2 so we switch back to the theano environment
python $EVAL_SCRIPT_DIR/measure_scores.py $EVAL_DATA_DIR/test-conc.txt $OUTPUT
source deactivate
