#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Facility for symbolically referencing type variables.

Type variables are instances of the TypeVar class, which is uniqued by name.
An "expando" accessor `TV` is provided that generates a named TypeVar for
any attribute access:

  >>> TV.T
  TypeVar(T)
  >>> TV.T is TV.U
  False
  >>> TV.T is TV.T
  True
"""

from enum import Enum
from typing import Dict

__all__ = [
    "TypeVar",
    "TV",

    # Predefined types.
    "I32",
    "I64",
    "F32",
    "F64",

    # TypeVar aliases.
    "T",
    "U",
    "V",
]


class TypeVar:
  """A replaceable type variable.

  Type variables are uniqued by name.
  """
  ALL_TYPEVARS = dict()  # type: Dict[str, "TypeVar"]

  def __new__(cls, name: str):
    existing = cls.ALL_TYPEVARS.get(name)
    if existing is not None:
      return existing
    new = super().__new__(cls)
    new.name = name
    cls.ALL_TYPEVARS[name] = new
    return new

  def __repr__(self):
    return f"TypeVar({self.name})"

  @classmethod
  def create_expando(cls):
    """Create an expando class that creates unique type vars on attr access."""

    class ExpandoTypeVars:

      def __getattr__(self, n):
        return cls(n)

    return ExpandoTypeVars()


# Expando access via TV.foo
TV = TypeVar.create_expando()

# Predefined types.
I32 = TV.I32
I64 = TV.I64
F32 = TV.F32
F64 = TV.F64

# Some common type name aliases.
T = TV.T
U = TV.U
V = TV.V
