# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tests for combining training corpora."""

import json
import os

from absl.testing import absltest

from mlgo import combine_training_corpus_lib


class CombineTrainingCorpusTest(absltest.TestCase):
    def test_combine_corpus(self):
        corpus_dir = self.create_tempdir()
        subcorpus1_dir = corpus_dir.mkdir(dir_path="subcorpus1")
        subcorpus2_dir = corpus_dir.mkdir(dir_path="subcorpus2")
        subcorpus1_description = {
            "has_thinlto": False,
            "modules": ["test1.o", "test2.o"],
        }
        subcorpus2_description = {
            "has_thinlto": False,
            "modules": ["test3.o", "test4.o"],
        }
        subcorpus1_description_file = subcorpus1_dir.create_file(
            file_path="corpus_description.json"
        )
        subcorpus2_description_file = subcorpus2_dir.create_file(
            file_path="corpus_description.json"
        )
        subcorpus1_description_file.write_text(json.dumps(subcorpus1_description))
        subcorpus2_description_file.write_text(json.dumps(subcorpus2_description))
        combine_training_corpus_lib.combine_corpus(corpus_dir.full_path)
        with open(
            os.path.join(corpus_dir, "corpus_description.json"), encoding="utf-8"
        ) as combined_corpus_description_file:
            combined_corpus_description = json.load(combined_corpus_description_file)
        self.assertEqual(combined_corpus_description["has_thinlto"], False)
        self.assertLen(combined_corpus_description["modules"], 4)
        self.assertIn("subcorpus1/test1.o", combined_corpus_description["modules"])
        self.assertIn("subcorpus1/test2.o", combined_corpus_description["modules"])
        self.assertIn("subcorpus2/test3.o", combined_corpus_description["modules"])
        self.assertIn("subcorpus2/test4.o", combined_corpus_description["modules"])

    def test_empty_folder(self):
        corpus_dir = self.create_tempdir()
        subcorpus1_dir = corpus_dir.mkdir(dir_path="subcorpus1")
        _ = corpus_dir.mkdir(dir_path="empty_dir")
        subcorpus1_description = {"modules": ["test1.o", "test2.o"]}
        subcorpus1_description_file = subcorpus1_dir.create_file(
            file_path="corpus_description.json"
        )
        subcorpus1_description_file.write_text(json.dumps(subcorpus1_description))
        combine_training_corpus_lib.combine_corpus(corpus_dir.full_path)
        with open(
            os.path.join(corpus_dir, "corpus_description.json"), encoding="utf-8"
        ) as combined_corpus_description_file:
            combined_corpus_description = json.load(combined_corpus_description_file)
        self.assertLen(combined_corpus_description["modules"], 2)

    def test_ignore_extra_file(self):
        corpus_dir = self.create_tempdir()
        subcorpus1_dir = corpus_dir.mkdir(dir_path="subcorpus1")
        _ = corpus_dir.create_file(file_path="empty.log")
        subcorpus1_description = {"modules": ["test1.o", "test2.o"]}
        subcorpus1_description_file = subcorpus1_dir.create_file(
            file_path="corpus_description.json"
        )
        subcorpus1_description_file.write_text(json.dumps(subcorpus1_description))
        combine_training_corpus_lib.combine_corpus(corpus_dir.full_path)
        with open(
            os.path.join(corpus_dir, "corpus_description.json"), encoding="utf-8"
        ) as combined_corpus_description_file:
            combined_corpus_description = json.load(combined_corpus_description_file)
        self.assertLen(combined_corpus_description["modules"], 2)

    def test_different_corpora(self):
        corpus_dir = self.create_tempdir()
        subcorpus1_dir = corpus_dir.mkdir(dir_path="subcorpus1")
        subcorpus2_dir = corpus_dir.mkdir(dir_path="subcorpus2")
        subcorpus1_description = {"has_thinlto": False, "modules": ["test1.o"]}
        subcorpus2_description = {"has_thinlto": True, "modules": ["test2.o"]}
        subcorpus1_description_file = subcorpus1_dir.create_file(
            file_path="corpus_description.json"
        )
        subcorpus2_description_file = subcorpus2_dir.create_file(
            file_path="corpus_description.json"
        )
        subcorpus1_description_file.write_text(json.dumps(subcorpus1_description))
        subcorpus2_description_file.write_text(json.dumps(subcorpus2_description))
        self.assertRaises(
            ValueError, combine_training_corpus_lib.combine_corpus, corpus_dir.full_path
        )


if __name__ == "__main__":
    absltest.main()
