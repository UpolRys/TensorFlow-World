# Convolutional Neural Network

This is the code repository for the blog post [Train a Convolutional Neural Network as a Classifier](http://machinelearninguru.com/deep_learning/tensorflow/neural_networks/cnn_classifier/cnn_classifier.html). Please refer to [this wiki page](https://github.com/astorfi/TensorFlow-Turorials/wiki/Convolutional-Neural-Networks) for more details.


## Training


*Train:*

The traing can be run using the **train.sh** `bash script` file using the following command:

```bash
./train.sh
```

The bash script is as below:
```bash
# Where the logs will be saved to.
train_dir=/home/sina/GITHUB/Tensorflow-Turorials/NeuralNetworks/convolutional-neural-network/code/train_logs

# Where the checkpoints is saved to.
checkpoint_dir=/home/sina/GITHUB/Tensorflow-Turorials/NeuralNetworks/convolutional-neural-network/code/checkpoints


# Run training.
python train_classifier.py \
  --train_dir=${train_dir} \
  --checkpoint_dir=${checkpoint_dir} \
  --batch_size=512 \
  --num_epochs=20 \
  --max_num_checkpoint=10 \
  --is_training \
  --allow_soft_placement

```

*helper:*

In order to realize that what are the parameters as input running the following command is recommended:

```bash
python train_classifier.py --help
```
In which `train_classifier.py` is the main file for running the training. The result of the above command will be as below:
```bash
  --train_dir TRAIN_DIR
                        Directory where event logs are written to.
  --checkpoint_dir CHECKPOINT_DIR
                        Directory where checkpoints are written to.
  --max_num_checkpoint MAX_NUM_CHECKPOINT
                        Maximum number of checkpoints that TensorFlow will
                        keep.
  --num_classes NUM_CLASSES
                        Number of model clones to deploy.
  --batch_size BATCH_SIZE
                        Number of model clones to deploy.
  --num_epochs NUM_EPOCHS
                        Number of epochs for training.
  --initial_learning_rate INITIAL_LEARNING_RATE
                        Initial learning rate.
  --learning_rate_decay_factor LEARNING_RATE_DECAY_FACTOR
                        Learning rate decay factor.
  --num_epochs_per_decay NUM_EPOCHS_PER_DECAY
                        Number of epoch pass to decay learning rate.
  --is_training [IS_TRAINING]
                        Training/Testing.
  --fine_tuning [FINE_TUNING]
                        Fine tuning is desired or not?.
  --online_test [ONLINE_TEST]
                        Fine tuning is desired or not?.
  --allow_soft_placement [ALLOW_SOFT_PLACEMENT]
                        Automatically put the variables on CPU if there is no
                        GPU support.
  --log_device_placement [LOG_DEVICE_PLACEMENT]
                        Demonstrate which variables are on what device.

```


## Evaluating

The evaluation will be run using the **evaluation.sh** `bash script` file using the following command:
```bash
./evaluation.sh
```

