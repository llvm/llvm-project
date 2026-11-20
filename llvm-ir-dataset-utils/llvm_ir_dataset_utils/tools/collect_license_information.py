"""Tool for collecting license information on all projects and putting it into a
JSON file.
"""

import os
import logging
import json
import shutil

from absl import flags
from absl import app

import ray

from llvm_ir_dataset_utils.util import dataset_corpus

FLAGS = flags.FLAGS

flags.DEFINE_string('corpus_dir', None, 'The base directory of the corpus')
flags.DEFINE_string('output_file', None, 'The path to the output JSON file.')
flags.DEFINE_string('license_dir', None,
                    'The path to place license files in. Optional')

flags.mark_flag_as_required('corpus_dir')
flags.mark_flag_as_required('output_file')


@ray.remote(num_cpus=1)
def get_license_information(corpus_path, license_dir):
  build_manifest = dataset_corpus.load_json_from_corpus(
      corpus_path, './build_manifest.json')

  if build_manifest is None:
    return None

  archive_url = ""
  if len(build_manifest["sources"]) == 0:
    # If we don't have any sources listed, this is a spack package
    archive_url = f'spack:{build_manifest["targets"][0]["name"]}'
  else:
    if build_manifest["sources"][-1]["type"] == "git":
      archive_url = build_manifest["sources"][-1]["repo_url"]
    elif build_manifest["sources"][-1]["type"] == "tar":
      archive_url = build_manifest["sources"][-1]["archive_url"]

  if license_dir:
    for license_file in build_manifest["license_files"]:
      license_data = dataset_corpus.load_file_from_corpus(
          corpus_path, license_file["file"])

      if license_data is None:
        logging.warning(
            f'Failed to load license {license_file} in corpus {corpus_path}')
        continue

      with open(os.path.join(license_dir, license_file["file"]),
                "wb") as license_file_handle:
        license_file_handle.write(license_data)

  return (corpus_path, build_manifest['license'],
          build_manifest['license_source'], build_manifest["license_files"],
          archive_url)


def main(_):
  build_corpora = os.listdir(FLAGS.corpus_dir)

  license_info_futures = []
  for build_corpus in build_corpora:
    corpus_path = os.path.join(FLAGS.corpus_dir, build_corpus)
    license_info_futures.append(
        get_license_information.remote(corpus_path, FLAGS.license_dir))

  raw_license_information = ray.get(license_info_futures)

  license_information = [
      license_info for license_info in raw_license_information
      if license_info is not None
  ]

  with open(FLAGS.output_file, 'w') as output_json_file:
    json.dump(license_information, output_json_file, indent=4)


if __name__ == '__main__':
  app.run(main)
