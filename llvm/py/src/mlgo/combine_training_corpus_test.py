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
"""Tests for combining training corpora."""

import json
import os

from absl.testing import absltest

from compiler_opt.tools import combine_training_corpus_lib


class CombineTrainingCorpusTest(absltest.TestCase):

  def test_combine_corpus(self):
    corpus_dir = self.create_tempdir()
    subcorpus1_dir = corpus_dir.mkdir(dir_path='subcorpus1')
    subcorpus2_dir = corpus_dir.mkdir(dir_path='subcorpus2')
    subcorpus1_description = {
        'has_thinlto': False,
        'modules': ['test1.o', 'test2.o']
    }
    subcorpus2_description = {
        'has_thinlto': False,
        'modules': ['test3.o', 'test4.o']
    }
    subcorpus1_description_file = subcorpus1_dir.create_file(
        file_path='corpus_description.json')
    subcorpus2_description_file = subcorpus2_dir.create_file(
        file_path='corpus_description.json')
    subcorpus1_description_file.write_text(json.dumps(subcorpus1_description))
    subcorpus2_description_file.write_text(json.dumps(subcorpus2_description))
    combine_training_corpus_lib.combine_corpus(corpus_dir.full_path)
    with open(
        os.path.join(corpus_dir, 'corpus_description.json'),
        encoding='utf-8') as combined_corpus_description_file:
      combined_corpus_description = json.load(combined_corpus_description_file)
    self.assertEqual(combined_corpus_description['has_thinlto'], False)
    self.assertLen(combined_corpus_description['modules'], 4)
    self.assertIn('subcorpus1/test1.o', combined_corpus_description['modules'])
    self.assertIn('subcorpus1/test2.o', combined_corpus_description['modules'])
    self.assertIn('subcorpus2/test3.o', combined_corpus_description['modules'])
    self.assertIn('subcorpus2/test4.o', combined_corpus_description['modules'])

  def test_empty_folder(self):
    corpus_dir = self.create_tempdir()
    subcorpus1_dir = corpus_dir.mkdir(dir_path='subcorpus1')
    _ = corpus_dir.mkdir(dir_path='empty_dir')
    subcorpus1_description = {'modules': ['test1.o', 'test2.o']}
    subcorpus1_description_file = subcorpus1_dir.create_file(
        file_path='corpus_description.json')
    subcorpus1_description_file.write_text(json.dumps(subcorpus1_description))
    combine_training_corpus_lib.combine_corpus(corpus_dir.full_path)
    with open(
        os.path.join(corpus_dir, 'corpus_description.json'),
        encoding='utf-8') as combined_corpus_description_file:
      combined_corpus_description = json.load(combined_corpus_description_file)
    self.assertLen(combined_corpus_description['modules'], 2)

  def test_ignore_extra_file(self):
    corpus_dir = self.create_tempdir()
    subcorpus1_dir = corpus_dir.mkdir(dir_path='subcorpus1')
    _ = corpus_dir.create_file(file_path='empty.log')
    subcorpus1_description = {'modules': ['test1.o', 'test2.o']}
    subcorpus1_description_file = subcorpus1_dir.create_file(
        file_path='corpus_description.json')
    subcorpus1_description_file.write_text(json.dumps(subcorpus1_description))
    combine_training_corpus_lib.combine_corpus(corpus_dir.full_path)
    with open(
        os.path.join(corpus_dir, 'corpus_description.json'),
        encoding='utf-8') as combined_corpus_description_file:
      combined_corpus_description = json.load(combined_corpus_description_file)
    self.assertLen(combined_corpus_description['modules'], 2)

  def test_different_corpora(self):
    corpus_dir = self.create_tempdir()
    subcorpus1_dir = corpus_dir.mkdir(dir_path='subcorpus1')
    subcorpus2_dir = corpus_dir.mkdir(dir_path='subcorpus2')
    subcorpus1_description = {'has_thinlto': False, 'modules': ['test1.o']}
    subcorpus2_description = {'has_thinlto': True, 'modules': ['test2.o']}
    subcorpus1_description_file = subcorpus1_dir.create_file(
        file_path='corpus_description.json')
    subcorpus2_description_file = subcorpus2_dir.create_file(
        file_path='corpus_description.json')
    subcorpus1_description_file.write_text(json.dumps(subcorpus1_description))
    subcorpus2_description_file.write_text(json.dumps(subcorpus2_description))
    self.assertRaises(ValueError, combine_training_corpus_lib.combine_corpus,
                      corpus_dir.full_path)


if __name__ == '__main__':
  absltest.main()
