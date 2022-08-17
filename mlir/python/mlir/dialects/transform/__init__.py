#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum


class FailurePropagationMode(Enum):
  """Propagation mode for silenceable errors."""
  PROPAGATE = 1
  SUPPRESS = 2

  def _as_int(self):
    if self is FailurePropagationMode.PROPAGATE:
      return 1

    assert self is FailurePropagationMode.SUPPRESS
    return 2

from .._transform_ops_gen import *
