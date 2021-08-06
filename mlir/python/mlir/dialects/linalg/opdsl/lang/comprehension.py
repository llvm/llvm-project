#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Model classes representing a tensor comprehension.

These classes model the language more at an AST level as evaluated. Reasoning
about it typically involves processing this form into config objects that
represent actual op definitions (i.e. YAML).
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from enum import Enum

from ..... import ir as _ir
from .affine import *
from .scalar_expr import *
from .types import *
from .yaml_helper import *

# Type aliases.
AffineDimList = Dict[str, _ir.AffineExpr]


class TensorExpression:
  """An expression that can appear on the RHS of a comprehension."""

  def to_scalar_expression(self) -> ScalarExpression:
    raise NotImplementedError()

  def visit_tensor_exprs(self, callback):
    """Visits all tensor expression reachable by the expression."""
    callback(self)

  def collect_dim_uses(self, uses: Set["DimDef"]):
    """Collects all DimDefs reachable through this expression."""
    results = set()

    def visit_dim_def(dim_def):
      if isinstance(dim_def, DimDef):
        uses.add(dim_def)

    def visit_affine_exprs(expr):
      if isinstance(expr, TensorUse):
        for ind in expr.indices:
          ind.visit_affine_exprs(visit_dim_def)
      if isinstance(expr, ReduceApply):
        for ind in expr.reduce.reduce_dims:
          ind.visit_affine_exprs(visit_dim_def)

    self.visit_tensor_exprs(visit_affine_exprs)

  def collect_tensor_uses(self, uses: Set["TensorUse"]):
    """Collects all TensorUses reachable through this expression."""

    def visit_tensor_use(expr):
      if isinstance(expr, TensorUse):
        uses.add(expr)

    self.visit_tensor_exprs(visit_tensor_use)

  def collect_indices(self, indices: Set["index"]):
    """Collects all index accesses reachable through this expression."""

    def visit_index(expr):
      if isinstance(expr, index):
        indices.add(expr)

    self.visit_tensor_exprs(visit_index)

  def collect_scalar_uses(self, uses: Set["ScalarDef"]):
    """Collects all ScalarDefs reachable through this expression."""

    def visit_scalar_def(expr):
      if isinstance(expr, ScalarDef):
        uses.add(expr)

    self.visit_tensor_exprs(visit_scalar_def)

  def __add__(self, rhs: "TensorExpression") -> "TensorExpression":
    return PrimFn.add(self, rhs)

  def __mul__(self, rhs) -> "TensorExpression":
    return PrimFn.mul(self, rhs)

  def __sub__(self, rhs) -> "TensorExpression":
    return PrimFn.sub(self, rhs)

  def __hash__(self):
    return hash(id(self))


class TensorUse(TensorExpression):
  """A used tensor represented by its (tensor_name, indices).

  Note that forming a comprehension via direct assignment is performed through
  __setitem__ on the TensorDef level. However, performing a reduction with
  compound ops (+=, *=, etc) is done by doing a:
    TensorDef.__getitem__
    TensorUse.__iadd__
    TensorDef.__setitem__
  """

  def __init__(self, operand_def: "OperandDef",
               indices: Sequence[AffineExprDef]):
    self.operand_def = operand_def
    self.indices = tuple(indices)

  def to_scalar_expression(self) -> ScalarExpression:
    return ScalarArg(self.tensor_name).expr()

  @property
  def tensor_name(self) -> str:
    name = self.operand_def.name
    assert name is not None, "TensorDef not attached"
    return name

  def __iadd__(self, rhs: TensorExpression) -> TensorExpression:
    return ReduceFn.add(*self._compute_reduce_dims(rhs))(rhs)

  def _compute_reduce_dims(self, rhs: TensorExpression) -> Set[DimDef]:
    """For implicit reductions, computes default reduction dims.

    Assumes that the rhs is the expression being reduced and self is being
    reduced into. Any indices referenced on the rhs and not in self are
    considered reduction dims and will be ordered as encountered on the rhs.
    """
    rhs_dims = set()
    lhs_dims = set()
    rhs.collect_dim_uses(rhs_dims)
    self.collect_dim_uses(lhs_dims)
    return rhs_dims - lhs_dims

  def __repr__(self):
    return f"{self.tensor_name}[{', '.join([repr(i) for i in self.indices])}]"


class OperandKind(Enum):
  InputTensor = 0
  Scalar = 1
  OutputTensor = 2
  Attribute = 3


class OperandDef:
  """Definition of an operand passed to an operation.

  Keep the meta information of Tensor, Scalar, and Attribute operands and
  provide the shared registration functionality.
  """

  def __init__(self,
               kind: OperandKind,
               type_var: TypeVar,
               size_exprs: Optional[Sequence[AffineExprDef]] = None,
               index_dims: Optional[Sequence[DimDef]] = None):
    if not isinstance(type_var, TypeVar):
      raise ValueError(
          f"OperandDef requires a TypeVar but got {repr(type_var)}")
    self.owner = None  # type: Optional["LinalgOpDef"]
    self.type_var = type_var
    self.size_exprs = size_exprs
    self.index_dims = index_dims
    self.kind = kind
    self.name = None  # type: Optional[str]
    self.registered_index = -1  # type: int

  def attach(self, index: int, name: str, owner: "LinalgOpDef"):
    if self.owner:
      raise ValueError(f"OperandDef already registered with op: {self}")
    self.registered_index = index
    self.name = name
    self.owner = owner

  def __hash__(self):
    return hash(id(self))

  def __repr__(self):
    return (f"{self.name}:OperandDef(kind={self.kind.name}, "
            f"type={repr(self.type_var)}, size_exprs={self.size_exprs}), "
            f"index_dims={self.index_dims})")


class TensorDef:
  """Tensor operand definition.

  Tensor operands are indexed using the associated indexing_map when forwarded
  to the body of the structured op. A unique name identifies the tensor operands
  and an index determines their position in the operation's parameter list. A
  tensor definition takes type, a shape, and an optional flag to mark output
  tensors. Additionally, a tuple of index dimensions may be used to map the
  tensor to the loop dimensions of the operation. This mapping is needed to
  compute the indexing map of shape-only tensors that have no uses.
  """

  def __init__(self,
               type_var: TypeVar,
               *shape: AffineExprDef,
               index_dims: Optional[Sequence[DimDef]] = None,
               output: bool = False):
    if index_dims and len(shape) != len(index_dims):
      raise ValueError(f"Expected the shape rank {len(shape)} to match the "
                       f"number of index_dims {len(index_dims)}")
    if index_dims and any(not isinstance(dim, DimDef) for dim in index_dims):
      raise ValueError(f"TensorDef requires index dims of type DimDef but "
                       f"got {index_dims}")
    kind = OperandKind.OutputTensor if output else OperandKind.InputTensor
    self.operand_def = OperandDef(
        kind, type_var, size_exprs=shape, index_dims=index_dims)

  def __getitem__(self, dims) -> TensorUse:
    assert self.operand_def.owner, "TensorDef is not attached to an op"
    state = AffineBuildState(
        global_state=self.operand_def.owner._affine_state,
        allow_new_symbols=False)
    if not isinstance(dims, tuple):
      dims = (dims,)  # Handle single subscript case.
    # Special case: (None) is a 0d-scalar use.
    if dims == (None,):
      dims = ()

    exprs = []
    for expr_def in dims:
      if not isinstance(expr_def, AffineExprDef):
        raise KeyError(
            "A TensorDef can only be subscripted by a tuple of affine dims")
      exprs.append(expr_def)
    return TensorUse(self.operand_def, exprs)

  def __setitem__(self, dims, value):
    """Creates a new 1:1 comprehension by binding this tensor to an expression.

    Note that due to the way assignment works in Python, we have to capture
    direct assignment as a setitem on the TensorDef.
    """
    if not isinstance(value, TensorExpression):
      raise ValueError(f"Only TensorExpressions can be assigned to TensorDefs. "
                       f"Got: {repr(value)}")
    use = self[dims]
    comp = Comprehension((use, value))
    self.operand_def.owner.comprehensions.append(comp)


class ScalarDef(TensorExpression):
  """Scalar operand definition.

  Scalar operands are forwarded to the body of the structured op as they are.
  A unique name identifies the scalars and an index determines their position in
  the operation's parameter list.
  """

  def __init__(self, type_var: TypeVar):
    self.operand_def = OperandDef(OperandKind.Scalar, type_var)

  @property
  def scalar_name(self) -> str:
    name = self.operand_def.name
    assert name is not None, "ScalarDef not attached"
    return name

  def to_scalar_expression(self) -> ScalarExpression:
    return ScalarArg(self.scalar_name).expr()


class AttributeDef:
  """Index Attribute definition.

  Index attributes provide a way to define and set symbols that can be used in
  indexing expressions. Every attribute specifies a tuple of symbols that at
  compile-time are replaced by integer values.
  """
  yaml_tag = "!LinalgAttributeDef"

  def __init__(self, *sizes: SymbolDef):
    if any(not isinstance(size, SymbolDef) for size in sizes):
      raise ValueError(f"AttributeDef requires sizes of type SymbolDef but got "
                       f"{sizes}")
    self.operand_def = OperandDef(OperandKind.Attribute, I64, size_exprs=sizes)


class Comprehension:
  """Represents a single comprehension."""

  def __init__(self, *bindings: Tuple[TensorUse, TensorExpression]):
    self.definitions = list()  # List[TensorUse]
    self.values = list()  # List[TensorExpression]

    # Find the lhs to reduction rhs.
    for assign, value in bindings:
      if isinstance(value, ReduceApply):
        if value.lhs:
          raise ValueError(f"Reduction expression already assigns: {value}")
        value.lhs = assign
      self.definitions.append(assign)
      self.values.append(value)

  @property
  def all_reduction_dims(self) -> Set[Tuple[DimDef, ...]]:
    """Gets the reduction dims for the comprehension or None."""
    result = set()
    for use in self.values:
      if isinstance(use, ReduceApply):
        result.add(use.reduce.reduce_dims)
      else:
        result.add(tuple())
    return result

  def __repr__(self):
    if len(self.definitions) > 1:
      defs_repr = f"({', '.join(repr(d) for d in self.definitions)})"
      values_repr = f"({', '.join(repr(v) for v in self.values)})"
    else:
      defs_repr = f"{repr(self.definitions[0])}"
      values_repr = f"{repr(self.values[0])}"

    return f"{defs_repr} = {values_repr}"


class PrimFnType:
  """Primitive operations."""

  def __init__(self, prim_name: str):
    self.prim_name = prim_name

  def __call__(self, *args):
    return PrimApply(self, args)

  def reduce(self, *reduce_dims: DimDef):
    """Shortcut to create a Reduce operation from this primitive."""
    return ReduceFnType(self, *reduce_dims)

  def __repr__(self):
    return f"{self.prim_name}"


class PrimFn:
  add = PrimFnType("add")
  exp = PrimFnType("exp")
  log = PrimFnType("log")
  mul = PrimFnType("mul")
  max = PrimFnType("max")
  min = PrimFnType("min")
  sub = PrimFnType("sub")


class ReduceFnType:
  """A reduction operator that reduces into its LHS from its RHS."""

  def __init__(self, operator: PrimFnType, *reduce_dims: DimDef):
    """Initializes the ReduceFn with a primitive function and dims."""
    if not isinstance(operator, PrimFnType):
      raise ValueError(f"Reduce expected a Prim operator but got {operator}")
    self.operator = operator
    self.reduce_dims = tuple(reduce_dims)

  def __call__(self, *args: TensorExpression):
    return ReduceApply(self, args)

  def __repr__(self):
    return (f"reduce_{self.operator.prim_name}"
            f"({', '.join(repr(d) for d in self.reduce_dims)})")


class ReduceFn:
  add = PrimFn.add.reduce
  mul = PrimFn.mul.reduce
  max = PrimFn.max.reduce
  min = PrimFn.min.reduce


class PrimApply(TensorExpression):
  """Application of a primitive."""

  def __init__(self, prim: PrimFnType, args: Sequence[TensorExpression]):
    self.prim = prim
    self.args = tuple(args)

  def to_scalar_expression(self) -> ScalarExpression:
    return ScalarApplyFn(self.prim.prim_name,
                         *[arg.to_scalar_expression() for arg in self.args
                          ]).expr()

  def visit_tensor_exprs(self, callback):
    super().visit_tensor_exprs(callback)
    for arg in self.args:
      arg.visit_tensor_exprs(callback)

  def __repr__(self):
    return f"{repr(self.prim)}({', '.join(repr(a) for a in self.args)})"


class const(TensorExpression):
  """Returns the given constant floating point or integer value."""

  def __init__(self, value: Any):
    with _ir.Context():
      if isinstance(value, float):
        self.value = str(_ir.FloatAttr.get_f64(float(value)))
      elif isinstance(value, int):
        self.value = str(
            _ir.IntegerAttr.get(_ir.IntegerType.get_signless(64), int(value)))
      else:
        raise ValueError(f"const requires int or float but got {type(value)}")

  def to_scalar_expression(self) -> ScalarExpression:
    return ScalarConst(self.value).expr()

  def __repr__(self):
    return f"const({self.type_var}, {self.value})"


class index(TensorExpression):
  """Returns the iteration index for a given dimension name.

  Resolves the given dimension name to obtain its position in the iteration
  domain of the operation.
  """

  def __init__(self, dim: DimDef):
    self.dim_def = dim
    self.dim = -1

  def resolve_dimension_name(self, affine_state: AffineBuildState):
    self.dim = affine_state.get_dim(self.dim_def.dimname)

  def to_scalar_expression(self) -> ScalarExpression:
    assert self.dim != -1, "Dimension name not resolved"
    return ScalarIndex(self.dim).expr()

  def __repr__(self):
    return f"index({repr(self.dim)})"


class cast(TensorExpression):
  """Casts the element type to a type (typically symbolic TypeVar)."""

  def __init__(self, to_type: TypeVar, operand: TensorExpression):
    self.to_type = to_type
    self.operand = operand

  def to_scalar_expression(self) -> ScalarExpression:
    return ScalarSymbolicCast(self.to_type,
                              self.operand.to_scalar_expression()).expr()

  def visit_tensor_exprs(self, callback):
    super().visit_tensor_exprs(callback)
    self.operand.visit_tensor_exprs(callback)

  def __repr__(self):
    return f"cast({self.to_type}, {repr(self.operand)})"


class ReduceApply(TensorExpression):
  """Application of a reduction.

  This captures the lhs separately (initial value) separately from the rhs.
  """

  def __init__(self, reduce: ReduceFnType, args: Sequence[TensorExpression]):
    self.reduce = reduce
    self.lhs = None  # type: Optional[TensorUse]
    self.args = tuple(args)

  def to_scalar_expression(self) -> ScalarExpression:
    if self.lhs is None:
      raise ValueError(f"Cannot scalarize a ReduceApply that has not been "
                       f"bound to its lhs: {self}")
    full_args = [self.lhs.to_scalar_expression()
                ] + [arg.to_scalar_expression() for arg in self.args]
    return ScalarApplyFn(self.reduce.operator.prim_name, *full_args).expr()

  def visit_tensor_exprs(self, callback):
    for arg in self.args:
      arg.visit_tensor_exprs(callback)

  def __repr__(self):
    return f"{repr(self.reduce)}({', '.join(repr(a) for a in self.args)})"


class OpInterfaceDef:
  """An interface that an op implements."""

  def __init__(self, cpp_name: str):
    self.cpp_name = cpp_name


ContractionOpInterface = OpInterfaceDef("LinalgContractionOpInterface")


class OpMetadataDef(YAMLObject):
  """Metadata about the op (generally not behavior impacting)."""
  yaml_tag = "!LinalgOpMetadata"

  def __init__(self, name: str, cpp_class_name: Optional[str],
               doc: Optional[str]):
    self.name = name
    self.cpp_class_name = cpp_class_name if cpp_class_name is not None else name
    self.doc = doc
    self.implements = []  # type: List[OpInterfaceDef]

  def to_yaml_custom_dict(self):
    d = dict(
        name=self.name,
        cpp_class_name=self.cpp_class_name,
        doc=self.doc,
    )
    if self.implements:
      d["implements"] = [intr.cpp_name for intr in self.implements]
    return d


class LinalgOpDef:
  """Definition of a linalg op."""

  def __init__(self,
               name: str,
               cpp_class_name: Optional[str] = None,
               doc: Optional[str] = None):
    self.metadata = OpMetadataDef(
        name=name, cpp_class_name=cpp_class_name, doc=doc)
    self.registered_operands = dict()  # type: Dict[str, OperandDef]
    self.domain = list()  # type: List[DimDef]
    self.comprehensions = list()  # type: List[Comprehension]
    self._affine_state = AffineBuildState()

  def add_operand(self, name: str, operand: OperandDef):
    """Registers an operand."""
    if name in self.registered_operands:
      raise ValueError(f"The operand {name} is already registered "
                       f"to {self.registered_operands['name']}")
    # Ensure output tensors are registered after input tensors and scalars and
    # attributes are registered after all other operand types.
    registered_kinds = [
        operand.kind.value for operand in self.registered_operands.values()
    ]
    if registered_kinds:
      maximum = max(registered_kinds)
      if maximum > operand.kind.value and maximum > OperandKind.Scalar.value:
        raise ValueError(
            f"The operand {name} of kind {operand.kind.name} is registered "
            f"after an operand of kind {OperandKind(maximum).name}")
    operand.attach(len(self.registered_operands), name, self)
    self.registered_operands[name] = operand

  def __repr__(self):
    lines = [
        f"LinalgOpDef({self.metadata.name} -> {self.metadata.cpp_class_name},"
    ]
    for name, operand in self.registered_operands.items():
      lines.append(f"  {operand}")
    if self.comprehensions:
      lines[-1] += " {"
      for comprehension in self.comprehensions:
        lines.append(f"    {comprehension}")
      lines.append("}")
    return "\n".join(lines)
