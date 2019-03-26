"""

The following script is used to evaluate a model.

Modified version of https://github.com/tensorflow/models/blob/master/research/object_detection/model_main.py
with a bugfix and specialized on evaluating only. The original only offers the possibility to evaluate
the last checkpoint while this one can evaluate all the checkpoints in checkpoint_dir

The bugfix is from this pull request: https://github.com/tensorflow/models/pull/5450
The addition is some flags and a modified RunConfig to save more models checkpoints.

NOTE: This comes from commit fe54563 - if you're using this script with a different commit check
if it doesn't cause any incompatibily

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import tensorflow as tf

from object_detection import model_hparams
from object_detection import model_lib

flags.DEFINE_string(
    'output_dir', None, 'Path to output directory where event files will be written.')
flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config '
                                                  'file.')
flags.DEFINE_boolean('eval_training_data', False,
                     'If training data should be evaluated for this job. Note '
                     'that one call only use this in eval-only mode, and '
                     '`checkpoint_dir` must be supplied.')
flags.DEFINE_integer('sample_1_of_n_eval_examples', 1, 'Will sample one of '
                                                       'every n eval input examples, where n is provided.')
flags.DEFINE_integer('sample_1_of_n_eval_on_train_examples', 5, 'Will sample '
                                                                'one of every n train input examples for evaluation, '
                                                                'where n is provided. This is only used if '
                                                                '`eval_training_data` is True.')
flags.DEFINE_string(
    'hparams_overrides', None, 'Hyperparameter overrides, '
                               'represented as a string containing comma-separated '
                               'hparam_name=value pairs.')
flags.DEFINE_string(
    'checkpoint_dir', None, 'Path to directory holding a checkpoint.  If '
                            '`checkpoint_dir` is provided, this binary operates in eval-only mode, '
                            'writing resulting metrics to `model_dir`.')
flags.DEFINE_boolean(
    'run_once', False, 'If running in eval-only mode, whether to run just '
                       'one round of eval vs running continuously (default).'
)


FLAGS = flags.FLAGS


def main(unused_argv):
    flags.mark_flag_as_required('output_dir')
    flags.mark_flag_as_required('checkpoint_dir')
    flags.mark_flag_as_required('pipeline_config_path')
    # Modified Runconfig
    config = tf.estimator.RunConfig(model_dir=FLAGS.output_dir)

    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=config,
        hparams=model_hparams.create_hparams(FLAGS.hparams_overrides),
        pipeline_config_path=FLAGS.pipeline_config_path,
        sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
        sample_1_of_n_eval_on_train_examples=FLAGS.sample_1_of_n_eval_on_train_examples)
    estimator = train_and_eval_dict['estimator']
    eval_input_fns = train_and_eval_dict['eval_input_fns']
    eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
    # train_steps = train_and_eval_dict['train_steps']

    if FLAGS.eval_training_data:
        name = 'training_data'
        input_fn = eval_on_train_input_fn
    else:
        name = 'validation_data'
        # The first eval input will be evaluated.
        input_fn = eval_input_fns[0]
    # tf.logging.set_verbosity(tf.logging.INFO)
    if FLAGS.run_once:
        estimator.evaluate(input_fn,
                           checkpoint_path=tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
    else:
        checkpoint_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        for checkpoint_path in reversed(checkpoint_state.all_model_checkpoint_paths):
            tf.logging.warning('Evaluating checkpoint path: {}'.format(checkpoint_path))
            estimator.evaluate(input_fn,
                               checkpoint_path=checkpoint_path)


if __name__ == '__main__':
    tf.app.run()
