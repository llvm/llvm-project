# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib.resources as _resources
from .ir2vec import *  # noqa: F401, F403

__versioninfo__ = (23, 0, 0)
__version__ = ".".join(str(v) for v in __versioninfo__)

SEED_EMBEDDING_75D: str = str(
    _resources.files(__name__).joinpath("seedEmbeddingVocab75D.json")
)
