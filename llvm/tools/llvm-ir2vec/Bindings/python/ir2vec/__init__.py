# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ir2vec")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

from .ir2vec import *  # noqa: F401, F403
from . import vocab  # noqa: E402, F401
