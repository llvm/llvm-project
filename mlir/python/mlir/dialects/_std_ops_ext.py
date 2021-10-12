#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
  from ..ir import *
  from .builtin import FuncOp
  from ._ods_common import get_default_loc_context as _get_default_loc_context

  from typing import Any, List, Optional, Union
except ImportError as e:
  raise RuntimeError("Error loading imports from extension module") from e


class ConstantOp:
  """Specialization for the constant op class."""

  def __init__(self, result: Type, value: Attribute, *, loc=None, ip=None):
    super().__init__(result, value, loc=loc, ip=ip)

  @property
  def type(self):
    return self.results[0].type


class CallOp:
  """Specialization for the call op class."""

  def __init__(self,
               calleeOrResults: Union[FuncOp, List[Type]],
               argumentsOrCallee: Union[List, FlatSymbolRefAttr, str],
               arguments: Optional[List] = None,
               *,
               loc=None,
               ip=None):
    """Creates an call operation.

    The constructor accepts three different forms:

      1. A function op to be called followed by a list of arguments.
      2. A list of result types, followed by the name of the function to be
         called as string, following by a list of arguments.
      3. A list of result types, followed by the name of the function to be
         called as symbol reference attribute, followed by a list of arguments.

    For example

        f = builtin.FuncOp("foo", ...)
        std.CallOp(f, [args])
        std.CallOp([result_types], "foo", [args])

    In all cases, the location and insertion point may be specified as keyword
    arguments if not provided by the surrounding context managers.
    """

    # TODO: consider supporting constructor "overloads", e.g., through a custom
    # or pybind-provided metaclass.
    if isinstance(calleeOrResults, FuncOp):
      if not isinstance(argumentsOrCallee, list):
        raise ValueError(
            "when constructing a call to a function, expected " +
            "the second argument to be a list of call arguments, " +
            f"got {type(argumentsOrCallee)}")
      if arguments is not None:
        raise ValueError("unexpected third argument when constructing a call" +
                         "to a function")

      super().__init__(
          calleeOrResults.type.results,
          FlatSymbolRefAttr.get(
              calleeOrResults.name.value,
              context=_get_default_loc_context(loc)),
          argumentsOrCallee,
          loc=loc,
          ip=ip)
      return

    if isinstance(argumentsOrCallee, list):
      raise ValueError("when constructing a call to a function by name, " +
                       "expected the second argument to be a string or a " +
                       f"FlatSymbolRefAttr, got {type(argumentsOrCallee)}")

    if isinstance(argumentsOrCallee, FlatSymbolRefAttr):
      super().__init__(
          calleeOrResults, argumentsOrCallee, arguments, loc=loc, ip=ip)
    elif isinstance(argumentsOrCallee, str):
      super().__init__(
          calleeOrResults,
          FlatSymbolRefAttr.get(
              argumentsOrCallee, context=_get_default_loc_context(loc)),
          arguments,
          loc=loc,
          ip=ip)
