#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Represents configured ops as emitted for code generation.

Classes in this module generally are directly serializable to YAML for use
by the code generator.

TODO: These should just be dumb containers or serialization code but they
currently encode too many details of how the language is interpreted. Move this
to helpers on the comprehension objects themselves.
"""

from typing import Dict, Optional

from ..... import ir as _ir
from .comprehension import *
from .yaml_helper import *

__all__ = ["LinalgStructuredOpConfig", "LinalgOpConfig", "OperandDefConfig"]


def _serialize_affine_map(affine_map: _ir.AffineMap) -> str:
  with affine_map.context:
    # Affine map printing/parsing is via an AffineMap attr.
    attr = _ir.AffineMapAttr.get(affine_map)
    return str(attr)


class TensorUseConfig:
  """Wrapper around a TensorUse with additional context-bound state."""

  def __init__(self, tensor_use: TensorUse, indexing_map: _ir.AffineMap):
    self.tensor_use = tensor_use
    self.indexing_map = indexing_map

  def __repr__(self):
    return f"Use({self.tensor_use}, indexing_map={self.indexing_map})"


class OperandDefConfig(YAMLObject):
  """Wrapper containing an operand definition with additional state."""
  yaml_tag = "!LinalgOperandDefConfig"

  def __init__(self,
               operand_def: OperandDef,
               shape_map: Optional[_ir.AffineMap] = None,
               index_attr_map: Optional[_ir.AffineMap] = None):
    self.operand_def = operand_def
    self.shape_map = shape_map  # type: Optional[_ir.AffineMap]
    self.index_attr_map = index_attr_map  # type: Optional[_ir.AffineMap]
    self.indexing_map = None  # type: Optional[_ir.AffineMap]

  @property
  def name(self) -> str:
    return self.operand_def.name

  @property
  def kind(self) -> OperandKind:
    return self.operand_def.kind

  @property
  def type_var(self) -> TypeVar:
    return self.operand_def.type_var

  def to_yaml_custom_dict(self):
    self_dict = dict(name=self.name, kind=self.operand_def.kind.name.lower())
    if self.type_var:
      self_dict["type_var"] = self.type_var.name
    if self.shape_map:
      self_dict["shape_map"] = _serialize_affine_map(self.shape_map)
    if self.index_attr_map:
      self_dict["index_attr_map"] = _serialize_affine_map(self.index_attr_map)
    if self.operand_def.default_indices:
      self_dict["default_indices"] = self.operand_def.default_indices
    if self.operand_def.default_fn:
      self_dict["default_fn"] = self.operand_def.default_fn
    return self_dict

  def __repr__(self):
    return (f"OperandDefConfig({self.operand_def}, "
            f"shape_map={self.shape_map}, "
            f"index_attr_map={self.index_attr_map}, "
            f"indexing_map={self.indexing_map})")


class LinalgIndexingMapsConfig(YAMLObject):
  """Abstracts the style of indexing maps that the op exports.

  Presently only static (tied to the op name) indexing maps are supported. In
  the future, it is expected that we will have additional variants:
    - Dynamic based on attributes
    - Dynamic based on operands
  Each is expected to require a different variant of specification.
  """
  yaml_tag = "!LinalgIndexingMapsConfig"

  def __init__(self,
               static_indexing_maps: Optional[Sequence[_ir.AffineMap]] = None):
    self.static_indexing_maps = static_indexing_maps

  def to_yaml_custom_dict(self):
    if self.static_indexing_maps is not None:
      return dict(static_indexing_maps=[
          _serialize_affine_map(m) for m in self.static_indexing_maps
      ])
    raise ValueError(
        f"LinalgIndexingMapsConfig must have one type of indexing map"
        f"(got none)")


class LinalgStructuredOpConfig(YAMLObject):
  """Configuration for metadata sufficient to construct a linalg named op."""

  yaml_tag = "!LinalgStructuredOpConfig"

  def __init__(self,
               comprehension: Comprehension,
               domain: Sequence[DimDef],
               registered_operands: Sequence[OperandDef],
               context: Optional[_ir.Context] = None):
    self.context = context if context is not None else _ir.Context()
    self.affine_state = AffineBuildState()
    self.writes = list()  # type: List[Tuple[TensorUse, TensorExpression]]
    self.operands = dict()  # type: Dict[OperandDef, OperandDefConfig]
    self.uses = dict()  # type: Dict[TensorUse, TensorUseConfig]

    # Compute the ordered set of writes and collect the tensor, capture, dims,
    # and index uses.
    collected_tensor_uses = set()
    collected_scalar_uses = set()
    collected_dim_uses = set()
    collected_indices = set()
    for write_use, read_use in zip(comprehension.definitions,
                                   comprehension.values):
      self.writes.append((write_use, read_use))

    for write_use, read_use in self.writes:
      collected_tensor_uses.add(write_use)
      read_use.collect_tensor_uses(collected_tensor_uses)
      read_use.collect_scalar_uses(collected_scalar_uses)
      read_use.collect_dim_uses(collected_dim_uses)
      write_use.collect_dim_uses(collected_dim_uses)
      read_use.collect_indices(collected_indices)

    # Set domain to the sorted list of uses if no domain annotation is given.
    if not domain:
      domain = sorted(collected_dim_uses, key=lambda dim: dim.dimname)

    # Verify the domain dimensions match the used dimensions.
    if (len(domain) != len(collected_dim_uses) or
        any(dim not in collected_dim_uses for dim in domain)):
      raise ValueError(f"Expected the annotated domain dimensions {domain} to "
                       f"match the set of dimension used by the tensor "
                       f"comprehension {collected_dim_uses}")

    # Instantiate the dimensions in the given order.
    with self.context:
      local_state = AffineBuildState(
          global_state=self.affine_state, allow_new_symbols=False)
      for dim in domain:
        dim.build(state=local_state)

    # Collect all attribute definitions.
    collected_attr_defs = list()
    for operand in registered_operands:
      if operand.is_attribute():
        collected_attr_defs.append(operand)

    # Collect all tensors with manual indexing annotation.
    collected_index_defs = list()
    for operand in registered_operands:
      if operand.index_dims:
        if any(dim not in collected_dim_uses for dim in operand.index_dims):
          raise ValueError(f"Expected all index dims {operand.index_dims} of "
                           f"operand {operand.name} to have uses.")
        collected_index_defs.append(operand)

    # Collect the operand definitions of all tensor/scalar uses, attributes, and
    # shape-only tensors.
    all_operand_defs = list()
    for use in collected_tensor_uses:
      all_operand_defs.append(use.operand_def)
    for use in collected_scalar_uses:
      all_operand_defs.append(use.operand_def)
    for definition in collected_attr_defs:
      all_operand_defs.append(definition)
    for definition in collected_index_defs:
      all_operand_defs.append(definition)

    # Add all operands in registration order to ensure the symbols are
    # registered in the order they appear.
    all_operand_defs = sorted(
        all_operand_defs, key=lambda operand_def: operand_def.registered_index)
    for operand_def in all_operand_defs:
      self.add_operand(operand_def)

    # Add all shape-only tensor index_dim annotations and all tensor uses.
    for definition in collected_index_defs:
      self.add_indexed_operand(definition)
    for use in collected_tensor_uses:
      self.add_tensor_use(use)

    # Normalize all shape and indexing maps now that full count of dims and
    # symbols are known.
    for cuse in self.uses.values():
      cuse.indexing_map = self._normalize_affine_map(cuse.indexing_map)
    for definition in collected_index_defs:
      self.operands[definition].indexing_map = self._normalize_affine_map(
          self.operands[definition].indexing_map)
    for operand_config in self.operands.values():
      if operand_config.shape_map:
        operand_config.shape_map = self._normalize_affine_map(
            operand_config.shape_map, with_dims=False)
      if operand_config.index_attr_map:
        operand_config.index_attr_map = self._normalize_affine_map(
            operand_config.index_attr_map, with_dims=False)

    # Now for each write use, propagate the indexing maps from the use to the
    # tensor, ensuring that there are not conflicts.
    for write_use, _ in self.writes:
      write_tensor_config = self.operands[write_use.operand_def]
      if write_tensor_config.indexing_map:
        raise ValueError(
            f"Unexpected multi-write to a single tensor: {write_tensor_config}")
      write_tensor_config.indexing_map = self.uses[write_use].indexing_map

    # For each read use, propagate the indexing maps from the use to the
    # tensor, ensuring that there are not conflicts.
    for _, read_expr in self.writes:
      read_uses = set()  # type: Set[TensorUse]
      read_expr.collect_tensor_uses(read_uses)
      for read_use in read_uses:
        read_operand_config = self.operands[read_use.operand_def]
        if (read_operand_config.indexing_map and
            read_operand_config.indexing_map !=
            self.uses[read_use].indexing_map):
          raise ValueError(
              f"Unexpected multi-read of a tensor with different accesses:"
              f"{read_operand_config} vs {read_use}")
        read_operand_config.indexing_map = self.uses[read_use].indexing_map

    # Set the indexing map of all scalar uses to the empty map.
    for operand_config in self.operands.values():
      if operand_config.operand_def.kind == OperandKind.SCALAR:
        operand_config.indexing_map = self._get_scalar_map()

    # Check all registered tensor and scalar operands have an indexing map.
    for operand in registered_operands:
      if operand.is_attribute():
        continue
      if not (operand in self.operands and self.operands[operand].indexing_map):
        raise ValueError(f"Failed to compute an indexing map for operand "
                         f"{operand.name}")

    # Collect reduction dims and ensure all the same.
    all_reduction_dims = set(comprehension.all_reduction_dims)
    if len(all_reduction_dims) != 1:
      raise ValueError(
          f"All writes within a generic must have the same reduction "
          f"dims. Got: {all_reduction_dims}")
    self.reduction_dims = next(iter(all_reduction_dims))

    # Check the index dimension exists and resolve.
    for index in collected_indices:
      if index.dim_def.dimname not in self.affine_state.all_dims:
        raise ValueError(
            f"The dimension {index.dim.dimname} is not part of the iteration "
            f"domain {self.affine_state.all_dims}")
      index.resolve_dimension_name(self.affine_state)

    # Generate the scalar assignments (used to build a body).
    self.assignments = [
        ScalarAssign(write_use.tensor_name, read_expr.to_scalar_expression())
        for write_use, read_expr in self.writes
    ]

  @property
  def ordered_operands(self) -> Sequence[OperandDefConfig]:
    return sorted(
        self.operands.values(),
        key=lambda operand: operand.operand_def.registered_index)

  @property
  def ordered_dims(self) -> Sequence[Tuple[str, int]]:
    """Gets the ordered list of dim bindings (symbolic name, position).

    TODO: The original parser relies on parse ordering to arrive at the
    iterator types, but that ordering is not defined on the Python side, so
    this may be ambiguous.
    """
    return list(self.affine_state.all_dims.items())

  @property
  def indexing_maps(self) -> Sequence[_ir.AffineMap]:
    return [o.indexing_map for o in self.ordered_operands if o.indexing_map]

  @property
  def iterator_types(self) -> Sequence[str]:

    def get_type(symbolic_name, position):
      for reduction_dim_expr in self.reduction_dims:
        if reduction_dim_expr.dimname == symbolic_name:
          return "reduction"
      return "parallel"

    return [get_type(*dim) for dim in self.ordered_dims]

  def add_operand(self, operand_def: OperandDef):
    if operand_def in self.operands:
      return
    if not (operand_def.is_tensor() or
            operand_def.kind == OperandKind.INDEX_ATTR):
      self.operands[operand_def] = OperandDefConfig(operand_def)
      return
    with self.context:
      local_state = AffineBuildState(
          global_state=self.affine_state, allow_new_dims=False)
      exprs = []
      for expr in operand_def.size_exprs:
        exprs.append(expr.build(state=local_state))
      assert local_state.local_dim_count == 0
      affine_map = _ir.AffineMap.get(
          dim_count=0, symbol_count=local_state.symbol_count, exprs=exprs)
      if operand_def.kind == OperandKind.INDEX_ATTR:
        self.operands[operand_def] = OperandDefConfig(
            operand_def, index_attr_map=affine_map)
      else:
        self.operands[operand_def] = OperandDefConfig(
            operand_def, shape_map=affine_map)

  def add_indexed_operand(self, operand_def: OperandDef):
    with self.context:
      local_state = AffineBuildState(
          global_state=self.affine_state, allow_new_symbols=False)
      exprs = []
      for expr in operand_def.index_dims:
        exprs.append(expr.build(state=local_state))
      self.operands[operand_def].indexing_map = _ir.AffineMap.get(
          dim_count=local_state.dim_count,
          symbol_count=local_state.symbol_count,
          exprs=exprs)

  def add_tensor_use(self, tensor_use: TensorUse):
    if tensor_use in self.uses:
      return
    with self.context:
      local_state = AffineBuildState(
          global_state=self.affine_state, allow_new_symbols=False)
      exprs = []
      for expr in tensor_use.indices:
        exprs.append(expr.build(state=local_state))
      indexing_map = _ir.AffineMap.get(
          dim_count=local_state.dim_count,
          symbol_count=local_state.symbol_count,
          exprs=exprs)

      use_config = TensorUseConfig(tensor_use, indexing_map)
      self.uses[tensor_use] = use_config

  def _get_scalar_map(self) -> _ir.AffineMap:
    """Create an empty affine map used to index a scalar."""
    with self.context:
      return _ir.AffineMap.get(
          dim_count=self.affine_state.dim_count,
          symbol_count=self.affine_state.symbol_count,
          exprs=list())

  def _normalize_affine_map(self,
                            affine_map: _ir.AffineMap,
                            with_dims: bool = True) -> _ir.AffineMap:
    """Normalizes an indexing map to have the max known symbols and dims."""
    with self.context:
      return _ir.AffineMap.get(
          dim_count=self.affine_state.dim_count if with_dims else 0,
          symbol_count=self.affine_state.symbol_count,
          exprs=list(affine_map.results))

  def to_yaml_custom_dict(self):
    self_dict = dict(args=self.ordered_operands)
    # TODO: Refactor the hierarchy internally when supporting more
    # than static (preserving this serialized form).
    self_dict["indexing_maps"] = LinalgIndexingMapsConfig(
        static_indexing_maps=self.indexing_maps)
    self_dict["iterator_types"] = self.iterator_types
    self_dict["assignments"] = self.assignments
    return self_dict

  def __repr__(self):
    lines = [f"LinalgGenericOpConfig(reduction_dims={self.reduction_dims},"]
    lines.append("operands=[")
    for def_config in self.ordered_operands:
      lines.append(f"  {repr(def_config)}")
    lines.append("], indexing_maps=[")
    for m in self.indexing_maps:
      lines.append(f"  {repr(m)}")
    lines.append(f"], iterator_types=[")
    for t in self.iterator_types:
      lines.append(f"  {t}")
    lines.append("])")
    return "\n".join(lines)


class LinalgOpConfig(YAMLObject):
  """Container for any supported linalg op type.

  This includes the concrete type by name for ease of parsing by systems
  that ignore tags.
  """
  yaml_tag = "!LinalgOpConfig"

  def __init__(self,
               metadata: OpMetadataDef,
               *,
               structured_op: Optional[LinalgStructuredOpConfig] = None):
    self.metadata = metadata
    self.structured_op = structured_op

  def to_yaml_custom_dict(self):
    self_dict = dict(metadata=self.metadata,)
    if self.structured_op:
      self_dict["structured_op"] = self.structured_op
    return self_dict

  @staticmethod
  def from_linalg_op_def(
      op_def: LinalgOpDef,
      context: Optional[_ir.Context] = None) -> Sequence["LinalgOpConfig"]:
    """Expands a LinalgOpDef into corresponding Linalg configured ops."""
    # TODO: Many LinalgOpDef patterns need to expand to multiple generics.
    assert len(op_def.comprehensions) == 1, "Only one comprehension supported"
    return [
        LinalgOpConfig(
            op_def.metadata,
            structured_op=LinalgStructuredOpConfig(
                op_def.comprehensions[0], op_def.domain,
                op_def.registered_operands.values(), context)),
    ]

  def __repr__(self):
    return (f"LinalgOpConfig(metadata={self.metadata},\n"
            f"structured_op={self.structured_op})")
