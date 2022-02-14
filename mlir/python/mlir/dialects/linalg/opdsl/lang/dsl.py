#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Dict, List, Sequence, Union

from contextlib import contextmanager
import functools
import inspect
import threading

from ..... import ir
from ...._ods_common import get_op_result_or_value as _get_op_result_or_value, get_op_results_or_values as _get_op_results_or_values
from .comprehension import *
from .config import *
from .emitter import *

_CONTEXT = threading.local()

StructuredOpOuts = Union[ir.Operation, ir.OpView, ir.OpResultList,
                         Sequence[Union[ir.Value, ir.Operation, ir.OpView]]]


@contextmanager
def bind_op_def(op_def: LinalgOpDef):
  if hasattr(_CONTEXT, "current_op_def"):
    raise ValueError("Cannot recursively define an operation")
  _CONTEXT.current_op_def = op_def
  try:
    yield op_def
  finally:
    del _CONTEXT.current_op_def


def current_op_def() -> LinalgOpDef:
  try:
    return _CONTEXT.current_op_def
  except AttributeError:
    raise ValueError(
        "Attempt to access the current op definition being defined "
        "but none is set. Did you mean to call this in an op definition?")


def _prepare_structured_op_outs(outs: StructuredOpOuts) -> ValueList:
  if isinstance(outs, (ir.Operation, ir.OpView)):
    return _get_op_results_or_values(outs)
  elif isinstance(outs, ir.OpResultList):
    return outs

  return [_get_op_result_or_value(o) for o in outs]


class DefinedOpCallable:
  """Callable that wraps any defined op function."""

  def __init__(self, op_name: str, op_def: LinalgOpDef):
    self.op_name = op_name
    self.op_def = op_def

  def __call__(self, *ins: Union[ir.Operation, ir.OpView, ir.Value],
               outs: StructuredOpOuts, **kwargs):
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
        self.op_def, context=ir.Context.current)

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
    out_values = _prepare_structured_op_outs(outs)
    in_values = [_get_op_result_or_value(i) for i in ins]
    if op_config.structured_op:
      if emit_generic:
        return emit_generic_structured_op(
            op_config.structured_op, *in_values, outs=out_values, **kwargs)
      else:
        return emit_named_structured_op(
            op_config.structured_op,
            self.op_name,
            self.op_def.metadata.cpp_class_name,
            *in_values,
            outs=out_values,
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
        linalg_structured_op, op_name=op_name, op_class_name=op_class_name)
  # Determine default names by introspecting the function.
  if op_name is None:
    op_name = dsl_func.__name__
  if op_class_name is None:
    # Camel case it.
    op_class_name = f"{''.join(x.title() for x in op_name.split('_'))}Op"

  op_def = LinalgOpDef(
      name=op_name, cpp_class_name=op_class_name, doc=inspect.getdoc(dsl_func))

  # Extract arguments and TensorDefs from the signature.
  dsl_func_args = list()
  sig = inspect.signature(dsl_func)
  for param_name, param in sig.parameters.items():
    param_default = param.default
    if isinstance(param_default, (TensorDef, ScalarDef, IndexAttrDef)):
      op_def.add_operand(param_name, param_default.operand_def)
    else:
      raise ValueError(
          f"@linalg_structured_op function parameters must be defaulted as "
          f"TensorDef(...), ScalarDef(...), or IndexAttrDef(...): "
          f"Found {param_name}: {param_default}")
    dsl_func_args.append(param_default)

  # Invoke the DSL func to finish populating the op definition.
  with bind_op_def(op_def):
    dsl_func(*dsl_func_args)

  # TODO: The returned callable should be an IR emitter but that is not
  # upstreamed yet.
  return DefinedOpCallable(op_name, op_def)


def implements(*interfaces: OpInterfaceDef):
  current_op_def().metadata.implements.extend(interfaces)


def domain(*dimensions: DimDef):
  if current_op_def().domain:
    raise ValueError(f"Expected only one set of domain dimensions per operator")
  if any(not isinstance(dim, DimDef) for dim in dimensions):
    raise ValueError(f"Expected dimensions of type DimDef but got {dimensions}")
  current_op_def().domain.extend(dimensions)
