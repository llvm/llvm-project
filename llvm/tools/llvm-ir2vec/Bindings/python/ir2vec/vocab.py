# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bundled vocabulary files for IR2Vec.

Attributes are filesystem paths to JSON vocabulary files bundled in the
wheel, suitable for passing directly to ``ir2vec.initEmbedding(vocabPath=...)``.
"""

import importlib.resources as _resources


def _resolve(filename: str) -> str:
    return str(_resources.files("ir2vec.vocab_data").joinpath(filename))


seedEmbedding75D: str = _resolve("seedEmbeddingVocab75D.json")
seedEmbedding100D: str = _resolve("seedEmbeddingVocab100D.json")
seedEmbedding300D: str = _resolve("seedEmbeddingVocab300D.json")
