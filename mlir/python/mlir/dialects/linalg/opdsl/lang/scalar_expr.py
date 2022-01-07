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
    "ScalarArithFn",
    "ScalarTypeFn",
    "ScalarArg",
    "ScalarConst",
    "ScalarIndex",
    "ScalarExpression",
]


class ScalarArithFn:
  """A type of ScalarExpression that applies an arithmetic function."""

  def __init__(self, fn_name: str, *operands: "ScalarExpression"):
    self.fn_name = fn_name
    self.operands = operands

  def expr(self) -> "ScalarExpression":
    return ScalarExpression(arith_fn=self)

  def __repr__(self):
    return f"ScalarArithFn<{self.fn_name}>({', '.join(self.operands)})"


class ScalarTypeFn:
  """A type of ScalarExpression that applies a type conversion function."""

  def __init__(self, fn_name: str, type_var: TypeVar,
               operand: "ScalarExpression"):
    self.fn_name = fn_name
    self.type_var = type_var
    self.operand = operand

  def expr(self) -> "ScalarExpression":
    return ScalarExpression(type_fn=self)

  def __repr__(self):
    return f"ScalarTypeFn<{self.fn_name}>({self.type_var}, {self.operand})"


class ScalarArg:
  """A type of ScalarExpression that references a named argument."""

  def __init__(self, arg: str):
    self.arg = arg

  def expr(self) -> "ScalarExpression":
    return ScalarExpression(scalar_arg=self)

  def __repr__(self):
    return f"(ScalarArg({self.arg})"


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


class ScalarExpression(YAMLObject):
  """An expression on scalar values.

  Can be one of:
    - ScalarArithFn
    - ScalarTypeFn
    - ScalarArg
    - ScalarConst
    - ScalarIndex
    - ScalarSymbolicCast
  """
  yaml_tag = "!ScalarExpression"

  def __init__(self,
               arith_fn: Optional[ScalarArithFn] = None,
               type_fn: Optional[ScalarTypeFn] = None,
               scalar_arg: Optional[ScalarArg] = None,
               scalar_const: Optional[ScalarConst] = None,
               scalar_index: Optional[ScalarIndex] = None):
    if (bool(arith_fn) + bool(type_fn) + bool(scalar_arg) + bool(scalar_const) +
        bool(scalar_index)) != 1:
      raise ValueError("One of 'arith_fn', 'type_fn', 'scalar_arg', "
                       "'scalar_const', 'scalar_index', must be specified")
    self.arith_fn = arith_fn
    self.type_fn = type_fn
    self.scalar_arg = scalar_arg
    self.scalar_const = scalar_const
    self.scalar_index = scalar_index

  def to_yaml_custom_dict(self):
    if self.arith_fn:
      return dict(
          arith_fn=dict(
              fn_name=self.arith_fn.fn_name,
              operands=list(self.arith_fn.operands),
          ))
    if self.type_fn:
      # Note that even though operands must be arity 1, we write it the
      # same way as for apply because it allows handling code to be more
      # generic vs having a special form.
      return dict(
          type_fn=dict(
              fn_name=self.type_fn.fn_name,
              type_var=self.type_fn.type_var.name,
              operands=[self.type_fn.operand],
          ))
    elif self.scalar_arg:
      return dict(scalar_arg=self.scalar_arg.arg)
    elif self.scalar_const:
      return dict(scalar_const=self.scalar_const.value)
    elif self.scalar_index:
      return dict(scalar_index=self.scalar_index.dim)
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
