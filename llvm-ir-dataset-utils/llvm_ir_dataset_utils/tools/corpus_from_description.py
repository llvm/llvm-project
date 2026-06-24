"""Tool that builds a bitcode corpus from a description"""

import json
import multiprocessing
import logging

from absl import app
from absl import flags
import ray

from llvm_ir_dataset_utils.builders import builder

FLAGS = flags.FLAGS

flags.DEFINE_string("corpus_description", None,
                    "The path to the JSON description file")
flags.DEFINE_string("source_dir", None,
                    "The base directory to download source code into.")
flags.DEFINE_string("build_dir", None,
                    "The base directory to perform the build in")
flags.DEFINE_string("corpus_dir", None, "The base directory to put the corpus")
flags.DEFINE_string(
    "buildcache_dir", "/tmp/buildcache",
    "The directory of the spack build cache to store packages in. Only used "
    "the spack builder.")
flags.DEFINE_bool(
    'cleanup', False, 'Whether or not to cleanup the source and '
    'build directories after finishing a build.')
flags.DEFINE_integer('thread_count', multiprocessing.cpu_count(), 'The number '
                     'of threads to use per job.')
flags.DEFINE_bool(
    'archive_corpus', False,
    'Whether or not to put the output corpus into an archive to reduce inode usage.'
)

flags.mark_flag_as_required("corpus_description")
flags.mark_flag_as_required("source_dir")
flags.mark_flag_as_required("build_dir")
flags.mark_flag_as_required("corpus_dir")


def main(_):
  ray.init()
  with open(FLAGS.corpus_description) as corpus_description_file:
    corpus_description = json.load(corpus_description_file)
    extra_builder_arguments = {'buildcache_dir': FLAGS.buildcache_dir}
    build_future = builder.get_build_future(
        corpus_description,
        FLAGS.source_dir,
        FLAGS.build_dir,
        FLAGS.corpus_dir,
        FLAGS.thread_count, {},
        cleanup=FLAGS.cleanup,
        extra_builder_arguments=extra_builder_arguments,
        archive_corpus=FLAGS.archive_corpus)
    logging.info('Starting build.')
    ray.get(build_future)
    logging.info('Build finished.')


if __name__ == "__main__":
  app.run(main)
