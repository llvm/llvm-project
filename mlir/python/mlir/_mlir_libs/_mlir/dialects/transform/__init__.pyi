#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from mlir.ir import Type, Context


class AnyOpType(Type):
  @staticmethod
  def isinstance(type: Type) -> bool: ...

  @staticmethod
  def get(context: Context | None = None) -> AnyOpType: ...


class OperationType(Type):
  @staticmethod
  def isinstance(type: Type) -> bool: ...

  @staticmethod
  def get(operation_name: str, context: Context | None = None) -> OperationType: ...

  @property
  def operation_name(self) -> str: ...
