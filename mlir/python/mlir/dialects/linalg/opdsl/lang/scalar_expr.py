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

from .comprehension import *
from .types import *
from .yaml_helper import *

__all__ = [
    "ScalarAssign",
    "ScalarFn",
    "ScalarArg",
    "ScalarConst",
    "ScalarIndex",
    "ScalarExpression",
]


class ScalarFn:
  """A type of ScalarExpression that applies a function."""

  def __init__(self, kind: "FunctionKind", fn_name: Optional[str],
               attr_name: Optional[str], type_var: Optional["TypeVar"],
               operands: Sequence["ScalarExpression"]):
    if bool(fn_name) + bool(attr_name) != 1:
      raise ValueError("One of 'fn_name', 'attr_name' must be specified")
    self.kind = kind
    self.fn_name = fn_name
    self.attr_name = attr_name
    self.type_var = type_var
    self.operands = operands

  def expr(self) -> "ScalarExpression":
    return ScalarExpression(scalar_fn=self)

  def __repr__(self):
    name = self.fn_name if self.fn_name else self.attr_name
    return (f"ScalarFn<{self.kind.name}.{name}>(type_var={self.type_var}, "
            f"operands=[{', '.join(self.operands)}])")


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
    - ScalarFn
    - ScalarArg
    - ScalarConst
    - ScalarIndex
  """
  yaml_tag = "!ScalarExpression"

  def __init__(self,
               scalar_fn: Optional[ScalarFn] = None,
               scalar_arg: Optional[ScalarArg] = None,
               scalar_const: Optional[ScalarConst] = None,
               scalar_index: Optional[ScalarIndex] = None):
    if (bool(scalar_fn) + bool(scalar_arg) + bool(scalar_const) +
        bool(scalar_index)) != 1:
      raise ValueError("One of 'scalar_fn', 'scalar_arg', 'scalar_const', or "
                       "'scalar_index' must be specified")
    self.scalar_fn = scalar_fn
    self.scalar_arg = scalar_arg
    self.scalar_const = scalar_const
    self.scalar_index = scalar_index

  def to_yaml_custom_dict(self):
    if self.scalar_fn:
      scalar_fn_dict = dict(kind=self.scalar_fn.kind.name.lower())
      if self.scalar_fn.fn_name:
        scalar_fn_dict["fn_name"] = self.scalar_fn.fn_name
      if self.scalar_fn.attr_name:
        scalar_fn_dict["attr_name"] = self.scalar_fn.attr_name
      if self.scalar_fn.type_var:
        scalar_fn_dict["type_var"] = self.scalar_fn.type_var.name
      scalar_fn_dict["operands"] = list(self.scalar_fn.operands)
      return dict(scalar_fn=scalar_fn_dict)
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
