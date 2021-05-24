#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Dict, Sequence

from mlir.ir import *
from mlir.dialects import linalg
from mlir.dialects import std
# TODO: resolve name collision for Linalg functionality that is injected inside
# the _mlir.dialects.linalg directly via pybind.
from _mlir.dialects.linalg import fill_builtin_region

from .scalar_expr import *
from .config import *

__all__ = [
    "emit_generic_structured_op",
    "emit_named_structured_op",
]


def isa(cls: Type, ty: Type):
  try:
    cls(ty)
    return True
  except ValueError:
    return False


def prepare_common_structured_op(op_config: LinalgStructuredOpConfig,
                                 *ins: Value, outs: Sequence[Value],
                                 captures: Sequence[Value]):
  all_arg_defs = op_config.ordered_tensor_args
  in_arg_defs = [arg for arg in all_arg_defs if arg.usage == "input"]
  out_arg_defs = [arg for arg in all_arg_defs if arg.usage == "output"]
  capture_arg_defs = op_config.ordered_capture_args

  # Verify outs and captures are sequences.
  if not isinstance(outs, Sequence):
    raise ValueError(f"Expected named argument outs to have type Sequence "
                     f"but got {type(outs)}")
  if not isinstance(captures, Sequence):
    raise ValueError(f"Expected named argument captures to have type Sequence "
                     f"but got {type(outs)}")

  # Arity validation.
  if len(ins) != len(in_arg_defs):
    raise ValueError(f"Expected {len(in_arg_defs)} inputs but got "
                     f"{len(ins)} for {op_config}")
  if outs and len(outs) != len(out_arg_defs):
    raise ValueError(f"Expected {len(out_arg_defs)} outputs but got "
                     f"{len(outs)} for {op_config}")
  if captures and len(captures) != len(capture_arg_defs):
    raise ValueError(f"Expected {len(capture_arg_defs)} captures but got "
                     f"{len(captures)} for {op_config}")

  outs, out_types = _infer_structured_outs(op_config, in_arg_defs, ins,
                                           out_arg_defs, outs)

  result_types = [t for t in out_types if isa(RankedTensorType, t)]

  # Initialize the type dictionary with the predefined types.
  type_mapping = dict()  # type: Dict[str, Type]
  type_mapping["F32"] = F32Type.get()
  type_mapping["F64"] = F64Type.get()
  type_mapping["I32"] = IntegerType.get_signless(32)
  type_mapping["I64"] = IntegerType.get_signless(64)

  # Extract type vars for input/output based types.
  for arg_def, arg_element_type in zip(
      in_arg_defs + out_arg_defs,
      _get_shaped_element_types_from_values(*ins, *outs)):
    _add_type_mapping(arg_def.tensor_def.type_var.name, arg_element_type,
                      type_mapping)

  # Extract type vars for captures and compute capture argument mapping.
  capture_arg_mapping = dict()  # type: Dict[str, Value]
  for arg_def, capture_value in zip(capture_arg_defs, captures):
    _add_type_mapping(arg_def.capture_def.type_var.name, capture_value.type,
                      type_mapping)
    capture_arg_mapping[arg_def.capture_def.capture_name] = capture_value

  # Emit the generic op.
  # TODO: Support emission of pure memref form.
  indexing_maps_attr = ArrayAttr.get([
      AffineMapAttr.get(am)
      # TODO: linalg verification does not currently allow symbols.
      # Compress them for now.
      for am in AffineMap.compress_unused_symbols(op_config.indexing_maps,
                                                  Context.current)
  ])
  iterator_types_attr = ArrayAttr.get(
      [StringAttr.get(s) for s in op_config.iterator_types])

  return (all_arg_defs, in_arg_defs, out_arg_defs, outs, result_types,
          type_mapping, capture_arg_mapping, indexing_maps_attr,
          iterator_types_attr)


def emit_generic_structured_op(op_config: LinalgStructuredOpConfig,
                               *ins: Value,
                               outs: Sequence[Value] = (),
                               captures: Sequence[Value] = ()):
  all_arg_defs, in_arg_defs, out_arg_defs, outs, result_types, type_mapping, \
  capture_arg_mapping, indexing_maps_attr, iterator_types_attr = \
     prepare_common_structured_op(op_config, *ins, outs = outs,
                                  captures=captures)

  generic_op = linalg.GenericOp(
      result_tensors=result_types,
      inputs=ins,
      outputs=outs,
      indexing_maps=indexing_maps_attr,
      iterator_types=iterator_types_attr,
      doc=None,  # TODO: Make optional.
      library_call=None)  # TODO: Make optional.

  # Construct the body.
  block_arg_names = _get_tensor_def_names(*in_arg_defs, *out_arg_defs)
  block_arg_types = _get_shaped_element_types_from_values(*ins, *outs)
  block = generic_op.regions[0].blocks.append(*block_arg_types)
  block_arg_mapping = dict(zip(block_arg_names, block.arguments))
  with InsertionPoint(block):
    body_builder = _BodyBuilder(type_mapping, block_arg_mapping,
                                capture_arg_mapping)
    for assignment in op_config.assignments:
      body_builder.assign(assignment)
    body_builder.yield_outputs(*_get_tensor_def_names(*out_arg_defs))

  if len(result_types) == 1:
    return generic_op.result
  else:
    return generic_op.results


def emit_named_structured_op(op_config: LinalgStructuredOpConfig,
                             op_name: str,
                             op_class_name: str,
                             *ins: Value,
                             outs: Sequence[Value] = (),
                             captures: Sequence[Value] = ()):
  all_arg_defs, in_arg_defs, out_arg_defs, outs, result_types, type_mapping, \
  capture_arg_mapping, indexing_maps_attr, iterator_types_attr = \
     prepare_common_structured_op(op_config, *ins, outs = outs,
                                  captures = captures)

  # If we get here, there must exist a builtin class `op_class_name`.
  ctx = Context.current
  fully_qualified_name = "linalg." + op_name
  if (not ctx.is_registered_operation(fully_qualified_name) or
      not op_class_name in linalg.__dict__.keys()):
    raise NotImplementedError(
        f"Unknown named op_name / op_class_name: {op_name} / {op_class_name}")

  named_op = getattr(linalg, op_class_name)(ins, outs, result_types)
  linalgDialect = ctx.get_dialect_descriptor("linalg")
  fill_builtin_region(linalgDialect, named_op.operation)
  # Note: mlir-linalg-ods-yaml-gen.cpp uses a special linalg.memoized_indexing_maps
  # attribute that the non-yaml path does not. The non-yaml path hardcodes the
  # indexing_maps in C++ directly.
  named_op.operation.attributes[
      "linalg.memoized_indexing_maps"] = indexing_maps_attr
  # iterator_types are hardcoded in C++ both in the yaml and non-yaml path.

  if len(result_types) == 1:
    return named_op.result
  else:
    return named_op.results


class _BodyBuilder:
  """Constructs a structured op body by evaluating assignments."""

  def __init__(self, type_mapping: Dict[str, Type],
               block_arg_mapping: Dict[str, Value],
               capture_arg_mapping: Dict[str, Value]):
    self.type_mapping = type_mapping
    self.block_arg_mapping = block_arg_mapping
    self.capture_arg_mapping = capture_arg_mapping
    self.yield_mapping = dict()  # type: Dict[str, Value]

  def assign(self, assignment: ScalarAssign):
    if assignment.arg in self.yield_mapping:
      raise ValueError(
          f"Multiple assignments to the same argument are forbidden: "
          f"{assignment}")
    self.yield_mapping[assignment.arg] = self.expression(assignment.value)

  def expression(self, expr: ScalarExpression) -> Value:
    if expr.scalar_arg:
      try:
        return self.block_arg_mapping[expr.scalar_arg.arg]
      except KeyError:
        raise ValueError(f"Argument {expr.scalar_arg.arg} is not bound for "
                         f"this structured op.")
    elif expr.scalar_capture:
      try:
        return self.capture_arg_mapping[expr.scalar_capture.capture]
      except KeyError:
        raise ValueError(
            f"Capture {expr.scalar_capture.capture} is not bound for "
            f"this structured op.")
    elif expr.scalar_const:
      value_attr = Attribute.parse(expr.scalar_const.value)
      return std.ConstantOp(value_attr.type, value_attr).result
    elif expr.scalar_index:
      dim_attr = IntegerAttr.get(
          IntegerType.get_signless(64), expr.scalar_index.dim)
      return linalg.IndexOp(IndexType.get(), dim_attr).result
    elif expr.scalar_apply:
      try:
        fn = getattr(self, f"_eval_{expr.scalar_apply.fn_name}")
      except AttributeError:
        raise ValueError(
            f"Function '{expr.scalar_apply.fn_name}' is not a known "
            "scalar body function")
      operand_values = [
          self.expression(operand) for operand in expr.scalar_apply.operands
      ]
      return fn(*operand_values)
    elif expr.symbolic_cast:
      operand_value = self.expression(expr.symbolic_cast.operand)
      return self.cast(expr.symbolic_cast.to_type.name, operand_value)
    raise NotImplementedError(f"Unimplemented scalar body expression: {expr}")

  def cast(self, type_var_name: str, operand: Value) -> Value:
    try:
      to_type = self.type_mapping[type_var_name]
    except KeyError:
      raise ValueError(f"Unbound type variable '{type_var_name}' ("
                       f"expected one of {self.type_mappings.keys()}")
    if operand.type == to_type:
      return operand
    if _is_integer_type(to_type):
      return self._cast_to_integer(to_type, operand)
    elif _is_floating_point_type(to_type):
      return self._cast_to_floating_point(to_type, operand)

  def _cast_to_integer(self, to_type: Type, operand: Value) -> Value:
    to_width = IntegerType(to_type).width
    operand_type = operand.type
    if _is_floating_point_type(operand_type):
      return std.FPToSIOp(to_type, operand).result
    if _is_index_type(operand_type):
      return std.IndexCastOp(to_type, operand).result
    # Assume integer.
    from_width = IntegerType(operand_type).width
    if to_width > from_width:
      return std.SignExtendIOp(to_type, operand).result
    elif to_width < from_width:
      return std.TruncateIOp(to_type, operand).result
    raise ValueError(f"Unable to cast body expression from {operand_type} to "
                     f"{to_type}")

  def _cast_to_floating_point(self, to_type: Type, operand: Value) -> Value:
    operand_type = operand.type
    if _is_integer_type(operand_type):
      return std.SIToFPOp(to_type, operand).result
    # Assume FloatType.
    to_width = _get_floating_point_width(to_type)
    from_width = _get_floating_point_width(operand_type)
    if to_width > from_width:
      return std.FPExtOp(to_type, operand).result
    elif to_width < from_width:
      return std.FPTruncOp(to_type, operand).result
    raise ValueError(f"Unable to cast body expression from {operand_type} to "
                     f"{to_type}")

  def yield_outputs(self, *output_names: str):
    output_values = []
    for n in output_names:
      try:
        output_values.append(self.yield_mapping[n])
      except KeyError:
        raise ValueError(f"Body assignments do not assign all outputs: "
                         f"missing '{n}'")
    linalg.YieldOp(output_values)

  def _eval_add(self, lhs: Value, rhs: Value) -> Value:
    if _is_floating_point_type(lhs.type):
      return std.AddFOp(lhs.type, lhs, rhs).result
    if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
      return std.AddIOp(lhs.type, lhs, rhs).result
    raise NotImplementedError("Unsupported 'add' operand: {lhs}")

  def _eval_sub(self, lhs: Value, rhs: Value) -> Value:
    if _is_floating_point_type(lhs.type):
      return std.SubFOp(lhs.type, lhs, rhs).result
    if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
      return std.SubIOp(lhs.type, lhs, rhs).result
    raise NotImplementedError("Unsupported 'sub' operand: {lhs}")

  def _eval_mul(self, lhs: Value, rhs: Value) -> Value:
    if _is_floating_point_type(lhs.type):
      return std.MulFOp(lhs.type, lhs, rhs).result
    if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
      return std.MulIOp(lhs.type, lhs, rhs).result
    raise NotImplementedError("Unsupported 'mul' operand: {lhs}")


def _infer_structured_outs(op_config: LinalgStructuredOpConfig,
                           in_arg_defs: Sequence[TensorDefConfig],
                           ins: Sequence[Value],
                           out_arg_defs: Sequence[TensorDefConfig],
                           outs: Sequence[Value]):
  """Infers implicit outs and output types.

  Respects existing contents of outs if not empty.

  Returns:
    normalized outs, output types
  """
  # If outs were explicitly provided, we accept them verbatim.
  if outs:
    return outs, [out.type for out in outs]

  raise NotImplementedError(f"Output tensor inference not yet supported for "
                            "structured ops")


def _get_shaped_element_types_from_values(*values: Value) -> Sequence[Type]:
  types = []
  for v in values:
    try:
      t = ShapedType(v.type)
    except Exception as e:
      raise ValueError(f"Expected ShapedType but got {v}") from e
    types.append(t.element_type)
  return types


def _get_tensor_def_names(
    *tensor_def_configs: TensorDefConfig) -> Sequence[str]:
  return [tdc.tensor_def.tensor_name for tdc in tensor_def_configs]


def _add_type_mapping(name: str, type: Type, type_mapping: Dict[str, Type]):
  if name in type_mapping:
    if type_mapping[name] != type:
      raise ValueError(f"Cannot overwrite type mapping {name} = "
                       f"{type_mapping[name]} by type {type}")
  type_mapping[name] = type


def _is_floating_point_type(t: Type) -> bool:
  # TODO: Create a FloatType in the Python API and implement the switch
  # there.
  return (F64Type.isinstance(t) or F32Type.isinstance(t) or
          F16Type.isinstance(t) or BF16Type.isinstance(t))


def _is_integer_type(t: Type) -> bool:
  return IntegerType.isinstance(t)


def _is_index_type(t: Type) -> bool:
  return IndexType.isinstance(t)


def _get_floating_point_width(t: Type) -> int:
  # TODO: Create a FloatType in the Python API and implement the switch
  # there.
  if F64Type.isinstance(t):
    return 64
  if F32Type.isinstance(t):
    return 32
  if F16Type.isinstance(t):
    return 16
  if BF16Type.isinstance(t):
    return 16
  raise NotImplementedError(f"Unhandled floating point type switch {t}")
