#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Dict, List

from contextlib import contextmanager
import functools
import inspect
import threading

from mlir import ir
from .comprehension import *
from .config import *
from .emitter import *

_CONTEXT = threading.local()


@contextmanager
def bind_op_def(model: LinalgOpDef):
  if hasattr(_CONTEXT, "current_op_def"):
    raise ValueError("Cannot recursively define an operation")
  _CONTEXT.current_op_def = model
  try:
    yield model
  finally:
    del _CONTEXT.current_op_def


def current_op_def() -> LinalgOpDef:
  try:
    return _CONTEXT.current_op_def
  except AttributeError:
    raise ValueError(
        "Attempt to access the current op definition being defined "
        "but none is set. Did you mean to call this in an op definition?")


class DefinedOpCallable:
  """Callable that wraps any defined op function."""

  def __init__(self, op_name: str, model: LinalgOpDef):
    self.op_name = op_name
    self.model = model

  def __call__(self, *ins: ir.Value, outs: Sequence[ir.Value], **kwargs):
    """Emits the corresponding op definition as IR.

    Most arguments are passed through to the underlying emitter. The following
    keyword argument is interpreted here:
      emit_generic: Emits a generic form as appropriate (default True). If
        False, a named form is emitted (which must have been built in to the
        compiler).
    """
    emit_generic = kwargs.pop("emit_generic", False)
    if not isinstance(emit_generic, bool):
      raise ValueError(f"The named argument 'emit_generic' needs to be "
                       f" of type bool but got {type(emit_generic)}")

    op_configs = LinalgOpConfig.from_linalg_op_def(
        self.model, context=ir.Context.current)

    if len(op_configs) != 1:
      # TODO: Support composite ops.
      raise NotImplementedError(
          f"Emission of composite linalg ops not supported: {op_configs}")

    ctx = ir.Context.current
    linalgDialect = ctx.get_dialect_descriptor("linalg")
    fully_qualified_name = "linalg." + self.op_name
    emit_generic = (
        emit_generic or not ctx.is_registered_operation(fully_qualified_name))

    op_config = op_configs[0]
    if op_config.structured_op:
      if emit_generic:
        return emit_generic_structured_op(
            op_config.structured_op, *ins, outs=outs, **kwargs)
      else:
        return emit_named_structured_op(
            op_config.structured_op,
            self.op_name,
            self.model.metadata.cpp_class_name,
            *ins,
            outs=outs,
            **kwargs)

    raise NotImplementedError(
        f"Emission of linalg op type not supported: {op_config}")


def linalg_structured_op(dsl_func=None,
                         *,
                         op_name=None,
                         op_class_name=None) -> DefinedOpCallable:
  if dsl_func is None:
    # Curry the keyword args in for delayed application.
    return functools.partial(
        tc_def_op, op_name=op_name, op_class_name=op_class_name)
  # Determine default names by introspecting the function.
  if op_name is None:
    op_name = dsl_func.__name__
  if op_class_name is None:
    # Camel case it.
    op_class_name = f"{''.join(x.title() for x in op_name.split('_'))}Op"

  tc_model = LinalgOpDef(
      name=op_name, cpp_class_name=op_class_name, doc=inspect.getdoc(dsl_func))

  # Extract arguments and TensorDefs from the signature.
  dsl_func_args = list()
  sig = inspect.signature(dsl_func)
  for param_name, param in sig.parameters.items():
    param_default = param.default
    if isinstance(param_default, (TensorDef, ScalarDef, AttributeDef)):
      tc_model.add_operand(param_name, param_default.operand_def)
    else:
      raise ValueError(f"@tc_def_op function parameters must be defaulted as "
                       f"TensorDef(...), ScalarDef(...), or AttributeDef(...): "
                       f"Found {param_name}: {param_default}")
    dsl_func_args.append(param_default)

  # Invoke the DSL func to finish populating the model.
  with bind_op_def(tc_model):
    dsl_func(*dsl_func_args)

  # TODO: The returned callable should be an IR emitter but that is not
  # upstreamed yet.
  return DefinedOpCallable(op_name, tc_model)


def implements(*interfaces: OpInterfaceDef):
  current_op_def().metadata.implements.extend(interfaces)
