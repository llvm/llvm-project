"""A tool for squashing the HF history. This is very simple, but for some
reason can't be performed from the HF CLI or the web interface.
"""

import logging

from absl import app

from huggingface_hub import HfApi


def main(_):
  logging.info('Squashing history')

  api = HfApi()

  api.super_squash_history(repo_id='llvm-ml/ComPile', repo_type='dataset')

  logging.info('History should be squashed')


if __name__ == '__main__':
  app.run(main)
