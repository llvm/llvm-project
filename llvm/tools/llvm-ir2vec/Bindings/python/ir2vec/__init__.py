# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib.resources as _resources

__versioninfo__ = (23, 0, 0)
__version__ = ".".join(str(v) for v in __versioninfo__)


def _resolve_bundled_vocab(filename: str) -> str:
    """Resolve a vocabulary file bundled inside this package."""
    return str(_resources.files("ir2vec").joinpath(filename))


# Pre-built vocabulary paths.
# The actual JSON files are injected into the ir2vec/ package directory
# at wheel build time by the downstream build scripts.
class vocab:
    """Namespace for bundled IR2Vec vocabulary paths."""

    seedEmbedding75D: str = _resolve_bundled_vocab("seedEmbeddingVocab75D.json")


from .ir2vec import *  # noqa: F401, F403
