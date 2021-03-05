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

from typing import Any, Dict, Optional

from mlir import ir as _ir

from .comprehension import *
from .yaml_helper import *

__all__ = [
    "LinalgStructuredOpConfig",
    "LinalgOpConfig",
]


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


class TensorDefConfig(YAMLObject):
  """Wrapper around a TensorDef with additional context-bound state."""
  yaml_tag = "LinalgTensorDef"

  def __init__(self, tensor_def: TensorDef, shape_map: _ir.AffineMap):
    self.tensor_def = tensor_def
    self.shape_map = shape_map
    self.indexing_map = None  # type: Optional[_ir.AffineMap]

  def to_yaml_custom_dict(self):

    def get_usage():
      if self.tensor_def.output:
        return "output"
      else:
        return "input"

    return dict(
        name=self.tensor_def.tensor_name,
        usage=get_usage(),
        shape=_serialize_affine_map(self.shape_map),
        element_type_var=self.tensor_def.type_var.name,
    )

  def __repr__(self):
    return f"Def({self.tensor_def}, shape_map={self.shape_map}, indexing_map={self.indexing_map})"


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
  """Configuration for metadata sufficient to construct a linalg single
  contraction named op."""

  yaml_tag = "!LinalgStructuredOpConfig"

  def __init__(self,
               comprehension: Comprehension,
               context: Optional[_ir.Context] = None):
    self.context = context if context is not None else _ir.Context()
    self.affine_state = AffineBuildState()
    self.writes = list()  # type: List[Tuple[TensorUse, TensorExpression]]
    self.tensor_args = dict()  # type: Dict[TensorDef, TensorDefConfig]
    self.uses = dict()  # type: Dict[TensorUse, TensorUseConfig]

    # Compute the ordered set of writes.
    collected_uses = set()
    for write_use, read_use in zip(comprehension.definitions,
                                   comprehension.values):
      self.writes.append((write_use, read_use))

    for write_use, read_use in self.writes:
      collected_uses.add(write_use)
      read_use.collect_uses(collected_uses)

    # Need to add all definitions before uses, so process twice.
    for use in collected_uses:
      self.add_tensor_arg(use.tensor_def)
    for use in collected_uses:
      self.add_use(use)

    # Now normalize all defs and uses indexing maps now that full count of
    # dims and symbols are known.
    for cuse in self.uses.values():
      cuse.indexing_map = self._normalize_affine_map(cuse.indexing_map)
    for cdef in self.tensor_args.values():
      cdef.shape_map = self._normalize_affine_map(cdef.shape_map,
                                                  with_dims=False)

    # Now for each write use, propagate the indexing maps from the use to the
    # tensor, ensuring that there are not conflicts.
    for write_use, _ in self.writes:
      write_tensor_def = self.tensor_args[write_use.tensor_def]
      if write_tensor_def.indexing_map:
        raise ValueError(
            f"Unexpected multi-write to a single tensor: {write_tensor_def}")
      write_tensor_def.indexing_map = self.uses[write_use].indexing_map

    # For each read use, propagate the indexing maps from the use to the
    # tensor, ensuring that there are not conflicts.
    for _, read_expr in self.writes:
      read_uses = set()  # type: Set[TensorUse]
      read_expr.collect_uses(read_uses)
      for read_use in read_uses:
        read_tensor_def = self.tensor_args[read_use.tensor_def]
        if (read_tensor_def.indexing_map and
            read_tensor_def.indexing_map != self.uses[read_use].indexing_map):
          raise ValueError(
              f"Unexpected multi-read of a tensor with different accesses:"
              f"{read_tensor_def} vs {read_use}")
        read_tensor_def.indexing_map = self.uses[read_use].indexing_map

    # Sanity check that all defs have an indexing map.
    assert all(d.indexing_map for d in self.tensor_args.values()), (
        f"Missing indexing map on TensorDef: {self.tensor_args}")

    # Collect reduction dims and ensure all the same.
    all_reduction_dims = set(comprehension.all_reduction_dims)
    if len(all_reduction_dims) != 1:
      raise ValueError(
          f"All writes within a generic must have the same reduction "
          f"dims. Got: {all_reduction_dims}")
    self.reduction_dims = next(iter(all_reduction_dims))

    # Generate the scalar assignments (used to build a body).
    self.assignments = [
        ScalarAssign(write_use.tensor_name, read_expr.to_scalar_expression())
        for write_use, read_expr in self.writes
    ]

  @property
  def ordered_tensor_args(self) -> Sequence[TensorDefConfig]:
    return sorted(self.tensor_args.values(),
                  key=lambda tdc: tdc.tensor_def.registered_index)

  @property
  def ordered_tensor_uses(self) -> Sequence[TensorUseConfig]:
    return sorted(self.uses.values(),
                  key=lambda tuc: tuc.tensor_use.tensor_def.registered_index)

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
    return [use.indexing_map for use in self.ordered_tensor_uses]

  @property
  def iterator_types(self) -> Sequence[str]:

    def get_type(symbolic_name, position):
      for reduction_dim_expr in self.reduction_dims:
        if reduction_dim_expr.dimname == symbolic_name:
          return "reduction"
      return "parallel"

    return [get_type(*dim) for dim in self.ordered_dims]

  def add_tensor_arg(self, tensor_def: TensorDef):
    if tensor_def in self.tensor_args:
      return
    with self.context:
      local_state = AffineBuildState(global_state=self.affine_state,
                                     allow_new_dims=False)
      exprs = []
      for expr in tensor_def.shape:
        exprs.append(expr.build(state=local_state))
      assert local_state.local_dim_count == 0
      indexing_map = _ir.AffineMap.get(dim_count=0,
                                       symbol_count=local_state.symbol_count,
                                       exprs=exprs)

      def_config = TensorDefConfig(tensor_def, indexing_map)
      self.tensor_args[tensor_def] = def_config

  def add_use(self, tensor_use: TensorUse):
    if tensor_use in self.uses:
      return
    with self.context:
      local_state = AffineBuildState(global_state=self.affine_state,
                                     allow_new_symbols=False)
      exprs = []
      for expr in tensor_use.indices:
        exprs.append(expr.build(state=local_state))
      assert local_state.local_symbol_count == 0
      indexing_map = _ir.AffineMap.get(dim_count=local_state.dim_count,
                                       symbol_count=local_state.symbol_count,
                                       exprs=exprs)

      use_config = TensorUseConfig(tensor_use, indexing_map)
      self.uses[tensor_use] = use_config

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
    self_dict = dict(
        args=self.ordered_tensor_args,
        # TODO: Refactor the hierarchy internally when supporting more
        # than static (preserving this serialized form).
        indexing_maps=LinalgIndexingMapsConfig(
            static_indexing_maps=self.indexing_maps),
        iterator_types=self.iterator_types,
        assignments=self.assignments,
    )
    return self_dict

  def __repr__(self):
    lines = [f"LinalgGenericOpConfig(reduction_dims={self.reduction_dims},"]
    lines.append("tensor_args=[")
    for def_config in self.ordered_tensor_args:
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
      tc_op_def: LinalgOpDef,
      context: Optional[_ir.Context] = None) -> Sequence["LinalgOpConfig"]:
    """Expands a LinalgOpDef into corresponding Linalg configured ops."""
    # TODO: Many LinalgOpDef patterns need to expand to multiple generics.
    assert len(
        tc_op_def.comprehensions) == 1, "Only one comprehension supported"
    return [
        LinalgOpConfig(tc_op_def.metadata,
                       structured_op=LinalgStructuredOpConfig(
                           tc_op_def.comprehensions[0], context)),
    ]

  def __repr__(self):
    return (f"LinalgOpConfig(metadata={self.metadata},\n"
            f"structured_op={self.structured_op})")
