#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# TODO: This file should be auto-generated.

from . import _cext

@_cext.register_dialect
class _Dialect(_cext.ir.Dialect):
  # Special case: 'std' namespace aliases to the empty namespace.
  DIALECT_NAMESPACE = "std"
  pass

@_cext.register_operation(_Dialect)
class AddFOp(_cext.ir.OpView):
  OPERATION_NAME = "std.addf"

  def __init__(self, loc, lhs, rhs):
    super().__init__(loc.context.create_operation(
      "std.addf", loc, operands=[lhs, rhs], results=[lhs.type]))

  @property
  def lhs(self):
    return self.operation.operands[0]

  @property
  def rhs(self):
    return self.operation.operands[1]

  @property
  def result(self):
    return self.operation.results[0]
