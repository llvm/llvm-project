"""Tool for building a list of cargo packages."""

import logging
import json

from absl import app
from absl import flags

import ray

from llvm_ir_dataset_utils.builders import builder

FLAGS = flags.FLAGS

flags.DEFINE_string('package_list', None, 'The path to the package list.')
flags.DEFINE_string('source_dir', None,
                    'The path to the directory to download source code into.')
flags.DEFINE_string('build_dir', None,
                    'The base directory to perform builds in.')
flags.DEFINE_string('corpus_dir', None, 'The directory to place the corpus in')
flags.DEFINE_integer('thread_count', 2,
                     'The number of threads to use per package build.')
flags.DEFINE_bool(
    'archive_corpus', False,
    'Whether or not to put the output corpus for each package into an archive.')


def main(_):
  ray.init()

  with open(FLAGS.package_list) as package_list_file:
    package_repositories = json.load(package_list_file)

  build_futures = []

  for index, package_repository in enumerate(package_repositories):
    corpus_description = {
        'sources': [{
            'type': 'git',
            'repo_url': package_repository['repo'],
            'commit_sha': None
        }],
        'folder_name': f'build-{index}',
        'build_system': 'swift',
        'package_name': f'build-{index}',
        'license': package_repository['license'],
        'license_source': package_repository['license_source']
    }

    build_futures.append(
        builder.get_build_future(
            corpus_description,
            FLAGS.source_dir,
            FLAGS.build_dir,
            FLAGS.corpus_dir,
            FLAGS.thread_count, {},
            cleanup=True,
            archive_corpus=FLAGS.archive_corpus))

  while len(build_futures) > 0:
    finished, build_futures = ray.wait(build_futures, timeout=5.0)
    finished_data = ray.get(finished)
    logging.info(
        f'Just finished {len(finished_data)}, {len(build_futures)} remaining.')


if __name__ == '__main__':
  app.run(main)
