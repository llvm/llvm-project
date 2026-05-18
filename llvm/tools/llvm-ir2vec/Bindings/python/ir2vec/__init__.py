# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

__versioninfo__ = (23, 0, 0)
__version__ = ".".join(str(v) for v in __versioninfo__)

from .ir2vec import *  # noqa: F401, F403
from . import vocab    # noqa: E402, F401
