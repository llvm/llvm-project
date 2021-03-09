#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Model classes representing a tensor comprehension.

These classes model the language more at an AST level as evaluated. Reasoning
about it typically involves processing this form into config objects that
represent actual op definitions (i.e. YAML).
"""

from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from mlir import ir as _ir

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

  def visit_affine_exprs(self, callback):
    """Visits all affine expressions reachable by the expression."""
    pass

  def _get_all_dim_defs(self) -> Set[DimDef]:
    """Recursively gets all DimDef affine expressions that are referenced."""
    results = set()

    def visitor(affine_expr):
      if isinstance(affine_expr, DimDef):
        results.add(affine_expr)

    self.visit_affine_exprs(visitor)
    return results

  def collect_uses(self, uses: Set["TensorUse"]):
    """Collects all TensorUses reachable through this expression."""
    pass

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

  def __init__(self, tensor_def: "TensorDef", indices: Sequence[AffineExprDef]):
    self.tensor_def = tensor_def
    self.indices = tuple(indices)

  def to_scalar_expression(self) -> ScalarExpression:
    assert self.tensor_def.tensor_name is not None
    return ScalarArg(self.tensor_def.tensor_name).expr()

  @property
  def tensor_name(self) -> str:
    n = self.tensor_def.tensor_name
    assert n is not None, "TensorDef not attached"
    return n

  def visit_affine_exprs(self, callback):
    for ind in self.indices:
      ind.visit_affine_exprs(callback)

  def collect_uses(self, uses: Set["TensorUse"]):
    uses.add(self)

  def __iadd__(self, rhs: TensorExpression) -> TensorExpression:
    return ReduceFn.add(*self._compute_reduce_dims(rhs))(rhs)

  def _compute_reduce_dims(self, rhs: TensorExpression) -> Set[DimDef]:
    """For implicit reductions, computes default reduction dims.

    Assumes that the rhs is the expression being reduced and self is being
    reduced into. Any indices referenced on the rhs and not in self are
    considered reduction dims and will be ordered as encountered on the rhs.
    """
    rhs_dims = rhs._get_all_dim_defs()
    lhs_dims = self._get_all_dim_defs()
    return rhs_dims - lhs_dims

  def __repr__(self):
    return f"{self.tensor_name}[{', '.join([repr(i) for i in self.indices])}]"


class TensorDef:
  """Bookkeeping of a single registered tensor, held in dict by name."""

  def __init__(self,
               type_var: TypeVar,
               *shape: AffineExprDef,
               indexing_map: Optional[_ir.AffineMap] = None,
               output: bool = False):
    if not isinstance(type_var, TypeVar):
      raise ValueError(f"TensorDef requires a TypeVar. Got: {repr(type_var)}")
    self.owner = None  # type: Optional["LinalgOpDef"]
    self.type_var = type_var
    self.shape = shape
    self.indexing_map = indexing_map
    self.output = output
    self.tensor_name = None  # type: Optional[str]
    self.registered_index = -1  # type: int

  @property
  def rank(self) -> int:
    """The rank of the tensor."""
    return len(self.shape)

  def attach(self, index: int, tensor_name: str, owner: "LinalgOpDef"):
    if self.owner:
      raise ValueError(f"TensorDef already registered with op: {self}")
    self.registered_index = index
    self.tensor_name = tensor_name
    self.owner = owner

  def __getitem__(self, dims) -> TensorUse:
    assert self.owner, "TensorDef is not attached to an op"
    state = AffineBuildState(global_state=self.owner._affine_state,
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
    return TensorUse(self, exprs)

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
    self.owner.comprehensions.append(comp)

  def __hash__(self):
    return hash(id(self))

  def __repr__(self):
    output = "OUTPUT " if self.output else ""
    return (f"{self.tensor_name}:TensorDef({output}{repr(self.type_var)}, "
            f"shape={self.shape})")


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
  sub = PrimFnType("sub")


class ReduceFnType:
  """A reduction operator that reduces into its LHS from its RHS."""

  def __init__(self, operator: PrimFnType, *reduce_dims: DimDef):
    """Initializes the ReduceFn with a primitive function and dims."""
    if not isinstance(operator, PrimFnType):
      raise ValueError(f"Reduce expected a Prim operator. Got: {operator}")
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


class PrimApply(TensorExpression):
  """Application of a primitive."""

  def __init__(self, prim: PrimFnType, args: Sequence[TensorExpression]):
    self.prim = prim
    self.args = tuple(args)

  def to_scalar_expression(self) -> ScalarExpression:
    return ScalarApplyFn(self.prim.prim_name,
                         *[arg.to_scalar_expression() for arg in self.args
                          ]).expr()

  def visit_affine_exprs(self, callback):
    for arg in self.args:
      arg.visit_affine_exprs(callback)

  def collect_uses(self, uses: Set["TensorUse"]):
    for arg in self.args:
      arg.collect_uses(uses)

  def __repr__(self):
    return f"{repr(self.prim)}({', '.join(repr(a) for a in self.args)})"


class cast(TensorExpression):
  """Casts the element type to a type (typically symbolic TypeVar)."""

  def __init__(self, to_type: TypeVar, operand: TensorExpression):
    self.to_type = to_type
    self.operand = operand

  def to_scalar_expression(self) -> ScalarExpression:
    return ScalarSymbolicCast(self.to_type,
                              self.operand.to_scalar_expression()).expr()

  def visit_affine_exprs(self, callback):
    self.operand.visit_affine_exprs(callback)

  def collect_uses(self, uses: Set["TensorUse"]):
    self.operand.collect_uses(uses)

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

  def visit_affine_exprs(self, callback):
    for ind in self.reduce.reduce_dims:
      ind.visit_affine_exprs(callback)
    for arg in self.args:
      arg.visit_affine_exprs(callback)

  def collect_uses(self, uses: Set["TensorUse"]):
    for arg in self.args:
      arg.collect_uses(uses)

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

  def __init__(self, name: str, cpp_op_name: Optional[str], doc: Optional[str]):
    self.name = name
    self.cpp_op_name = cpp_op_name if cpp_op_name is not None else name
    self.doc = doc
    self.implements = []  # type: List[OpInterfaceDef]

  def to_yaml_custom_dict(self):
    d = dict(
        name=self.name,
        cpp_op_name=self.cpp_op_name,
        doc=self.doc,
    )
    if self.implements:
      d["implements"] = [intr.cpp_name for intr in self.implements]
    return d


class LinalgOpDef:
  """Definition of a linalg op."""

  def __init__(self,
               name: str,
               cpp_op_name: Optional[str] = None,
               doc: Optional[str] = None):
    self.metadata = OpMetadataDef(name=name, cpp_op_name=cpp_op_name, doc=doc)
    self.registered_tensors = dict()  # type: Dict[str, TensorDef]
    self.comprehensions = list()  # type: List[Comprehension]
    self._affine_state = AffineBuildState()

  @property
  def inputs(self) -> Sequence[TensorDef]:
    return [t for t in self.registered_tensors.values() if not t.output]

  @property
  def outputs(self) -> Sequence[TensorDef]:
    return [t for t in self.registered_tensors.values() if t.output]

  def add_tensor(self, tensor_name: str, tensor: TensorDef):
    """Registers a tensor."""
    if tensor_name in self.registered_tensors:
      raise ValueError(f"Tensor {tensor_name} is already registered "
                       f"to {self.registered_tensors['tensor_name']}")
    tensor.attach(len(self.registered_tensors), tensor_name, self)
    self.registered_tensors[tensor_name] = tensor

  def tensor(self, name):
    """Gets a registered tensor by name."""
    try:
      return self.registered_tensors[name]
    except KeyError:
      raise KeyError(f"Tensor {name} is not registered")

  def __repr__(self):
    lines = [
        f"LinalgOpDef({self.metadata.name} -> {self.metadata.cpp_op_name},"
    ]
    for name, tensor in self.registered_tensors.items():
      lines.append(f"  {tensor}")
    if self.comprehensions:
      lines[-1] += " {"
      for comprehension in self.comprehensions:
        lines.append(f"    {comprehension}")
      lines.append("}")
    return "\n".join(lines)
