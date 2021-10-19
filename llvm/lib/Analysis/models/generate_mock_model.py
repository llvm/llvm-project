"""Generate a mock model for LLVM tests.

The generated model is not a neural net - it is just a tf.function with the
correct input and output parameters. By construction, the mock model will always
output 1.
"""

import os
import importlib.util
import sys

import tensorflow as tf


def get_output_spec_path(path):
  return os.path.join(path, 'output_spec.json')


def build_mock_model(path, signature):
  """Build and save the mock model with the given signature"""
  module = tf.Module()

  # We have to set this useless variable in order for the TF C API to correctly
  # intake it
  module.var = tf.Variable(0.)

  def action(*inputs):
    s = tf.reduce_sum([tf.cast(x, tf.float32) for x in tf.nest.flatten(inputs)])
    return {signature['output']: float('inf') + s + module.var}

  module.action = tf.function()(action)
  action = {'action': module.action.get_concrete_function(signature['inputs'])}
  tf.saved_model.save(module, path, signatures=action)

  output_spec_path = get_output_spec_path(path)
  with open(output_spec_path, 'w') as f:
    print(f'Writing output spec to {output_spec_path}.')
    f.write(signature['output_spec'])


def get_external_signature(config_path):
  """Get the signature for the desired model.

  We manually import the python file at config_path to avoid adding a gin
  dependency to the LLVM build.
  """
  spec = importlib.util.spec_from_file_location('config', config_path)
  config = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(config)

  return {
      'inputs': config.get_input_signature(),
      'output': config.get_output_signature(),
      'output_spec': config.get_output_spec()
  }


def main(argv):
  assert len(argv) == 3
  config_path = argv[1]
  model_path = argv[2]

  print(f'Using config file at [{argv[1]}]')
  signature = get_external_signature(config_path)
  build_mock_model(model_path, signature)


if __name__ == '__main__':
  main(sys.argv)
