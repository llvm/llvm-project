#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from mlir.ir import Type

__all__ = [
  "QuantizedType",
  "AnyQuantizedType",
  "UniformQuantizedType",
  "UniformQuantizedPerAxisType",
  "CalibratedQuantizedType",
]

class QuantizedType(Type):
  @staticmethod
  def isinstance(type: Type) -> bool: ...

  @staticmethod
  def default_minimum_for_integer(is_signed: bool, integral_width: int) -> int:
    ...

  @staticmethod
  def default_maximum_for_integer(is_signed: bool, integral_width: int) -> int:
    ...

  @property
  def expressed_type(self) -> Type: ...

  @property
  def flags(self) -> int: ...

  @property
  def is_signed(self) -> bool: ...

  @property
  def storage_type(self) -> Type: ...

  @property
  def storage_type_min(self) -> int: ...

  @property
  def storage_type_max(self) -> int: ...

  @property
  def storage_type_integral_width(self) -> int: ...

  def is_compatible_expressed_type(self, candidate: Type) -> bool: ...

  @property
  def quantized_element_type(self) -> Type: ...

  def cast_from_storage_type(self, candidate: Type) -> Type: ...

  @staticmethod
  def cast_to_storage_type(type: Type) -> Type: ...

  def cast_from_expressed_type(self, candidate: Type) -> Type: ...

  @staticmethod
  def cast_to_expressed_type(type: Type) -> Type: ...

  def cast_expressed_to_storage_type(self, candidate: Type) -> Type: ...


class AnyQuantizedType(QuantizedType):

  @classmethod
  def get(cls, flags: int, storage_type: Type, expressed_type: Type,
          storage_type_min: int, storage_type_max: int) -> Type:
    ...


class UniformQuantizedType(QuantizedType):

  @classmethod
  def get(cls, flags: int, storage_type: Type, expressed_type: Type,
          scale: float, zero_point: int, storage_type_min: int,
          storage_type_max: int) -> Type: ...

  @property
  def scale(self) -> float: ...

  @property
  def zero_point(self) -> int: ...

  @property
  def is_fixed_point(self) -> bool: ...


class UniformQuantizedPerAxisType(QuantizedType):

  @classmethod
  def get(cls, flags: int, storage_type: Type, expressed_type: Type,
          scales: list[float], zero_points: list[int], quantized_dimension: int,
          storage_type_min: int, storage_type_max: int):
    ...

  @property
  def scales(self) -> list[float]: ...

  @property
  def zero_points(self) -> list[int]: ...

  @property
  def quantized_dimension(self) -> int: ...

  @property
  def is_fixed_point(self) -> bool: ...


def CalibratedQuantizedType(QuantizedType):

  @classmethod
  def get(cls, expressed_type: Type, min: float, max: float): ...

  @property
  def min(self) -> float: ...

  @property
  def max(self) -> float: ...