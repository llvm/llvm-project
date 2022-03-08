#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
  from ..ir import *
except ImportError as e:
  raise RuntimeError("Error loading imports from extension module") from e

class ModuleOp:
  """Specialization for the module op class."""

  def __init__(self, *, loc=None, ip=None):
    super().__init__(self.build_generic(results=[], operands=[], loc=loc,
                                        ip=ip))
    body = self.regions[0].blocks.append()

  @property
  def body(self):
    return self.regions[0].blocks[0]
