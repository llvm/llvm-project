#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Models DAGs of scalar math expressions.

Used for generating region bodies at the "math" level where they are still type
polymorphic. This is modeled to be polymorphic by attribute name for interop
with serialization schemes that are just plain-old-dicts.

These classes are typically not user accessed and are created as a by-product
of interpreting a comprehension DSL and model the operations to perform in the
op body. The class hierarchy is laid out to map well to a form of YAML that
can be easily consumed from the C++ side, not necessarily for ergonomics.
"""

from typing import Optional, Sequence

from .yaml_helper import *
from .types import *

__all__ = [
    "ScalarAssign",
    "ScalarApplyFn",
    "ScalarArg",
    "ScalarCapture",
    "ScalarConst",
    "ScalarIndex",
    "ScalarExpression",
    "ScalarSymbolicCast",
]


class ScalarApplyFn:
  """A type of ScalarExpression that applies a named function to operands."""

  def __init__(self, fn_name: str, *operands: "ScalarExpression"):
    self.fn_name = fn_name
    self.operands = operands

  def expr(self) -> "ScalarExpression":
    return ScalarExpression(scalar_apply=self)

  def __repr__(self):
    return f"ScalarApplyFn<{self.fn_name}>({', '.join(self.operands)})"


class ScalarArg:
  """A type of ScalarExpression that references a named argument."""

  def __init__(self, arg: str):
    self.arg = arg

  def expr(self) -> "ScalarExpression":
    return ScalarExpression(scalar_arg=self)

  def __repr__(self):
    return f"(ScalarArg({self.arg})"


class ScalarCapture:
  """A type of ScalarExpression that references a named capture."""

  def __init__(self, capture: str):
    self.capture = capture

  def expr(self) -> "ScalarExpression":
    return ScalarExpression(scalar_capture=self)

  def __repr__(self):
    return f"(ScalarCapture({self.capture})"


class ScalarConst:
  """A type of ScalarExpression representing a constant."""

  def __init__(self, value: str):
    self.value = value

  def expr(self) -> "ScalarExpression":
    return ScalarExpression(scalar_const=self)

  def __repr__(self):
    return f"(ScalarConst({self.value})"


class ScalarIndex:
  """A type of ScalarExpression accessing an iteration index."""

  def __init__(self, dim: int):
    self.dim = dim

  def expr(self) -> "ScalarExpression":
    return ScalarExpression(scalar_index=self)

  def __repr__(self):
    return f"(ScalarIndex({self.dim})"


class ScalarSymbolicCast:
  """A type of ScalarExpression that symbolically casts an operand to a TypeVar."""

  def __init__(self, to_type: TypeVar, operand: "ScalarExpression"):
    self.to_type = to_type
    self.operand = operand

  def expr(self) -> "ScalarExpression":
    return ScalarExpression(symbolic_cast=self)

  def __repr__(self):
    return f"ScalarSymbolicCast({self.to_type}, {self.operand})"


class ScalarExpression(YAMLObject):
  """An expression on scalar values.

  Can be one of:
    - ScalarApplyFn
    - ScalarArg
    - ScalarCapture
    - ScalarConst
    - ScalarIndex
    - ScalarSymbolicCast
  """
  yaml_tag = "!ScalarExpression"

  def __init__(self,
               scalar_apply: Optional[ScalarApplyFn] = None,
               scalar_arg: Optional[ScalarArg] = None,
               scalar_capture: Optional[ScalarCapture] = None,
               scalar_const: Optional[ScalarConst] = None,
               scalar_index: Optional[ScalarIndex] = None,
               symbolic_cast: Optional[ScalarSymbolicCast] = None):
    if (bool(scalar_apply) + bool(scalar_arg) + bool(scalar_capture) +
        bool(scalar_const) + bool(scalar_index) + bool(symbolic_cast)) != 1:
      raise ValueError(
          "One of 'scalar_apply', 'scalar_arg', 'scalar_capture', 'scalar_const', "
          "'scalar_index', 'symbolic_cast' must be specified")
    self.scalar_apply = scalar_apply
    self.scalar_arg = scalar_arg
    self.scalar_capture = scalar_capture
    self.scalar_const = scalar_const
    self.scalar_index = scalar_index
    self.symbolic_cast = symbolic_cast

  def to_yaml_custom_dict(self):
    if self.scalar_apply:
      return dict(
          scalar_apply=dict(
              fn_name=self.scalar_apply.fn_name,
              operands=list(self.scalar_apply.operands),
          ))
    elif self.scalar_arg:
      return dict(scalar_arg=self.scalar_arg.arg)
    elif self.scalar_capture:
      return dict(scalar_capture=self.scalar_capture.capture)
    elif self.scalar_const:
      return dict(scalar_const=self.scalar_const.value)
    elif self.scalar_index:
      return dict(scalar_index=self.scalar_index.dim)
    elif self.symbolic_cast:
      # Note that even though operands must be arity 1, we write it the
      # same way as for apply because it allows handling code to be more
      # generic vs having a special form.
      return dict(
          symbolic_cast=dict(
              type_var=self.symbolic_cast.to_type.name,
              operands=[self.symbolic_cast.operand]))
    else:
      raise ValueError(f"Unexpected ScalarExpression type: {self}")


class ScalarAssign(YAMLObject):
  """An assignment to a named argument (LHS of a comprehension)."""
  yaml_tag = "!ScalarAssign"

  def __init__(self, arg: str, value: ScalarExpression):
    self.arg = arg
    self.value = value

  def to_yaml_custom_dict(self):
    return dict(arg=self.arg, value=self.value)

  def __repr__(self):
    return f"ScalarAssign({self.arg}, {self.value})"
