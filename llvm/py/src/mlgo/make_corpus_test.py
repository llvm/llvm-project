# coding=utf-8
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test for compiler_opt.tools.make_corpus_lib"""

import json
import os

from absl.testing import absltest

from compiler_opt.tools import make_corpus_lib


class MakeCorpusTest(absltest.TestCase):

  def test_load_bitcode_from_directory(self):
    outer = self.create_tempdir()
    tempdir = outer.mkdir(dir_path='nested')
    tempdir.create_file('test1.bc')
    tempdir.create_file('test2.bc')
    relative_paths = make_corpus_lib.load_bitcode_from_directory(outer)
    relative_paths = sorted(relative_paths)
    self.assertEqual(relative_paths[0], 'nested/test1')
    self.assertEqual(relative_paths[1], 'nested/test2')

  def test_copy_bitcode(self):
    build_dir = self.create_tempdir()
    nested_dir = build_dir.mkdir(dir_path='nested')
    nested_dir.create_file('test1.bc')
    nested_dir.create_file('test2.bc')
    relative_paths = ['nested/test1', 'nested/test2']
    corpus_dir = self.create_tempdir()
    make_corpus_lib.copy_bitcode(relative_paths, build_dir, corpus_dir)
    output_files = sorted(os.listdir(os.path.join(corpus_dir, './nested')))
    self.assertEqual(output_files[0], 'test1.bc')
    self.assertEqual(output_files[1], 'test2.bc')

  def test_write_corpus_manifest(self):
    relative_output_paths = ['test/test1', 'test/test2']
    output_dir = self.create_tempdir()
    default_args = ['-O3', '-c']
    make_corpus_lib.write_corpus_manifest(relative_output_paths, output_dir,
                                          default_args)
    with open(
        os.path.join(output_dir, 'corpus_description.json'),
        encoding='utf-8') as corpus_description_file:
      corpus_description = json.load(corpus_description_file)
    self.assertEqual(corpus_description['global_command_override'],
                     default_args)
    self.assertEqual(corpus_description['has_thinlto'], False)
    self.assertEqual(corpus_description['modules'], relative_output_paths)


if __name__ == '__main__':
  absltest.main()
