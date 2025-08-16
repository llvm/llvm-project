"""Library functions for extracting ource files produced by the compiler
wrapper.
"""

import pathlib
import os
import shutil

SOURCE_EXTENSIONS = ['.source', '.preprocessed_source']


def copy_source(source_base_dir, output_dir):
  for source_extension in SOURCE_EXTENSIONS:
    for source_base_path in pathlib.Path(source_base_dir).glob(
        '**/*' + source_extension):
      source_rel_path = os.path.relpath(source_base_path, start=source_base_dir)
      destination_path = os.path.join(output_dir, source_rel_path)
      os.makedirs(os.path.dirname(destination_path), exist_ok=True)
      shutil.copy(source_base_path, destination_path)
