#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

from mlir.ir import Type, Context

__all__ = [
    'PDLType',
    'AttributeType',
    'OperationType',
    'RangeType',
    'TypeType',
    'ValueType',
]


class PDLType(Type):
  @staticmethod
  def isinstance(type: Type) -> bool: ...


class AttributeType(Type):
  @staticmethod
  def isinstance(type: Type) -> bool: ...

  @staticmethod
  def get(context: Optional[Context] = None) -> AttributeType: ...


class OperationType(Type):
  @staticmethod
  def isinstance(type: Type) -> bool: ...

  @staticmethod
  def get(context: Optional[Context] = None) -> OperationType: ...


class RangeType(Type):
  @staticmethod
  def isinstance(type: Type) -> bool: ...

  @staticmethod
  def get(element_type: Type) -> RangeType: ...

  @property
  def element_type(self) -> Type: ...


class TypeType(Type):
  @staticmethod
  def isinstance(type: Type) -> bool: ...

  @staticmethod
  def get(context: Optional[Context] = None) -> TypeType: ...


class ValueType(Type):
  @staticmethod
  def isinstance(type: Type) -> bool: ...

  @staticmethod
  def get(context: Optional[Context] = None) -> ValueType: ...
