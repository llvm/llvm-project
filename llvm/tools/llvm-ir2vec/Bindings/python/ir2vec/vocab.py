# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bundled vocabulary files for IR2Vec.

This module exposes the packaged seed embedding vocabulary JSON as a filesystem
path.

Usage:

```python
vocabObj = ir2vec.loadVocab(
    ir2vec.vocab.seedEmbedding75D
)
```

"""

import importlib.resources as _resources


def _resolve(filename: str) -> str:
    return str(_resources.files("ir2vec.vocab_data").joinpath(filename))


# The vocab_data/ directory is intentionally empty in the repository.
#
# The vocabulary JSON file (seedEmbeddingVocab75D.json) lives at:
#   llvm/lib/Analysis/models/seedEmbeddingVocab75D.json
#
# It is to be injected into the directory at wheel build time
# by the build script so the assembled wheel is self-contained.
# The Analysis models directory is the single source of truth.
seedEmbedding75D: str = _resolve("seedEmbeddingVocab75D.json")
