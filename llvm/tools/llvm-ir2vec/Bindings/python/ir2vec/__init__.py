# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""IR2Vec: LLVM IR Embedding Framework.

Python bindings for IR2Vec, which generates vector representations of
LLVM IR for use in machine learning-based compiler optimization.

Example usage::

    import ir2vec

    vocabObj = ir2vec.loadVocab(
        ir2vec.vocab.seedEmbedding75D
    )

    emb = ir2vec.initEmbedding (
        filename="file_path.ll",
        mode=ir2vec.IR2VecKind.FlowAware,
        vocab=vocabObj
    )


    func_names = emb.getFuncNames()
    func_emb_map = emb.getFuncEmbMap()
"""


from importlib.metadata import version, PackageNotFoundError

try:
    # After `pip install`, reads the version baked into the wheel's METADATA.
    # Always consistent with what `pip show ir2vec` reports.
    __version__ = version("ir2vec")
except PackageNotFoundError:
    # Running directly from the source tree without installing.
    __version__ = "0.0.0+DEADBEEF"

try:
    from ir2vec.ir2vec import *  # noqa: F401, F403
except ImportError as e:
    import sys

    raise ImportError(
        f"Failed to import the IR2Vec native module.\n"
        f"\n"
        f"This usually means one of:\n"
        f"  1. The package was not installed correctly (missing .so/.pyd)\n"
        f"  2. Python version mismatch (built for a different Python)\n"
        f"  3. Platform mismatch (built for a different OS/architecture)\n"
        f"\n"
        f"Your Python: {sys.version}\n"
        f"Your platform: {sys.platform}\n"
        f"\n"
        f"Original error: {e}"
    ) from e

# Make the vocab module available as ir2vec.vocab
from ir2vec import vocab  # noqa: E402, F401
