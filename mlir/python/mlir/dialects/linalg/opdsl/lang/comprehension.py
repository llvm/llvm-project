#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Model classes representing a tensor comprehension.

These classes model the language more at an AST level as evaluated. Reasoning
about it typically involves processing this form into config objects that
represent actual op definitions (i.e. YAML).
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
from enum import Enum

from ..... import ir as _ir
from .affine import *
from .scalar_expr import *
from .types import *
from .yaml_helper import *

###############################################################################
# Tensor expression nodes.
###############################################################################


class TensorExpression:
  """An expression that can appear on the RHS of a comprehension."""

  def to_scalar_expression(self) -> ScalarExpression:
    raise NotImplementedError()

  def visit_tensor_exprs(self, callback: Callable[["TensorExpression"], None]):
    """Visits all tensor expression reachable by the expression."""
    callback(self)

  def collect_dim_uses(self, uses: Set["DimDef"]):
    """Collects all DimDefs reachable through this expression."""

    def visit_dim_def(dim_def: AffineExprDef):
      if isinstance(dim_def, DimDef):
        uses.add(dim_def)

    def visit_affine_exprs(expr: "TensorExpression"):
      if isinstance(expr, TensorUse):
        for ind in expr.indices:
          ind.visit_affine_exprs(visit_dim_def)
      if isinstance(expr, TensorReduceFn):
        for ind in expr.reduce_fn.reduce_dims:
          ind.visit_affine_exprs(visit_dim_def)

    self.visit_tensor_exprs(visit_affine_exprs)

  def collect_tensor_uses(self, uses: Set["TensorUse"]):
    """Collects all TensorUses reachable through this expression."""

    def visit_tensor_use(expr: "TensorExpression"):
      if isinstance(expr, TensorUse):
        uses.add(expr)

    self.visit_tensor_exprs(visit_tensor_use)

  def collect_indices(self, indices: Set["index"]):
    """Collects all index accesses reachable through this expression."""

    def visit_index(expr: "TensorExpression"):
      if isinstance(expr, index):
        indices.add(expr)

    self.visit_tensor_exprs(visit_index)

  def collect_scalar_uses(self, uses: Set["ScalarDef"]):
    """Collects all ScalarDefs reachable through this expression."""

    def visit_scalar_def(expr: "TensorExpression"):
      if isinstance(expr, ScalarDef):
        uses.add(expr)

    self.visit_tensor_exprs(visit_scalar_def)

  def __add__(self, rhs: "TensorExpression") -> "TensorExpression":
    return BinaryFn.add(self, rhs)

  def __mul__(self, rhs) -> "TensorExpression":
    return BinaryFn.mul(self, rhs)

  def __sub__(self, rhs) -> "TensorExpression":
    return BinaryFn.sub(self, rhs)

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
    assert name is not None, "TensorDef not registered with an op"
    return name

  def _compute_reduce_dims(self, rhs: TensorExpression) -> Set[DimDef]:
    # Computes the reduction dims for implicit reductions. Assumes that the rhs
    # is the expression being reduced and self is being reduced into. Any
    # indices referenced on the rhs and not in self are considered reduction
    # dims and will be ordered as encountered on the rhs.
    rhs_dims = set()
    lhs_dims = set()
    rhs.collect_dim_uses(rhs_dims)
    self.collect_dim_uses(lhs_dims)
    return rhs_dims - lhs_dims

  def __iadd__(self, rhs: TensorExpression) -> "TensorReduceFn":
    return ReduceFnUse(BinaryFn.add, None, *self._compute_reduce_dims(rhs))(rhs)

  def __repr__(self):
    return (f"{self.operand_def.name}"
            f"[{', '.join([repr(i) for i in self.indices])}]")


class TensorFn(TensorExpression):
  """Application of a tensor function."""

  def __init__(self, kind: "FunctionKind", name: Optional[str],
               operand_def: Optional["OperandDef"], type_var: Optional[TypeVar],
               args: Sequence[TensorExpression]):
    if bool(name) + bool(operand_def) != 1:
      raise ValueError("One of 'name', 'operand_def' must be specified")
    self.name = name
    self.kind = kind
    self.operand_def = operand_def
    self.type_var = type_var
    self.args = args

  def to_scalar_expression(self) -> ScalarExpression:
    if self.operand_def:
      assert self.operand_def.name, "TensorFn not registered with an op"
    attr_name = self.operand_def.name if self.operand_def else None
    args = [arg.to_scalar_expression() for arg in self.args]
    return ScalarFn(self.kind, self.name, attr_name, self.type_var, args).expr()

  def visit_tensor_exprs(self, callback: Callable[["TensorExpression"], None]):
    super().visit_tensor_exprs(callback)
    for arg in self.args:
      arg.visit_tensor_exprs(callback)

  def __repr__(self):
    name = self.operand_def.name if self.operand_def else self.name
    return (f"{self.kind.name}.{name}(type_var={self.type_var}, "
            f"args={', '.join(repr(a) for a in self.args)})")


class TensorReduceFn(TensorExpression):
  """Application of a reduction function.

  This captures the lhs (initial value) separately from the rhs.
  """

  def __init__(self, reduce_use: "ReduceFnUse",
               args: Sequence[TensorExpression]):
    self.reduce_use = reduce_use
    self.lhs = None  # type: Optional[TensorUse]
    self.args = args

  def to_scalar_expression(self) -> ScalarExpression:
    if self.lhs is None:
      raise ValueError(f"Cannot scalarize a TensorReduceFn that has not been "
                       f"bound to its lhs: {self}")
    full_args = [self.lhs.to_scalar_expression()
                ] + [arg.to_scalar_expression() for arg in self.args]
    fn_name = None
    attr_name = None
    if self.reduce_use.binary_fn:
      fn_name = self.reduce_use.binary_fn.fn_name
    if self.reduce_use.binary_attr:
      attr_name = self.reduce_use.binary_attr.operand_def.name
    return ScalarFn(FunctionKind.BINARY, fn_name, attr_name, None,
                    full_args).expr()

  def visit_tensor_exprs(self, callback: Callable[["TensorExpression"], None]):
    for arg in self.args:
      arg.visit_tensor_exprs(callback)

  def __repr__(self):
    return f"{repr(self.reduce_use)}({', '.join(repr(a) for a in self.args)})"


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
    return f"const({self.value})"


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


###############################################################################
# Function types and function definitions.
###############################################################################


class FunctionKind(Enum):
  UNARY = 0
  BINARY = 1
  TYPE = 2


class UnaryFnType:
  """Unary function.

  A unary function takes one tensor expression and returns the
  function evaluation result.
  """

  def __init__(self, fn_name: str):
    self.fn_name = fn_name

  def __call__(self, arg: TensorExpression) -> "TensorFn":
    return TensorFn(FunctionKind.UNARY, self.fn_name, None, None, [arg])

  def __repr__(self):
    return f"{self.fn_name}"


class UnaryFn:
  """Unary function namespace."""
  exp = UnaryFnType("exp")
  log = UnaryFnType("log")


class BinaryFnType:
  """Binary function.

  A binary function takes two tensor expressions and returns the
  function evaluation result.
  """

  def __init__(self, fn_name: str):
    self.fn_name = fn_name

  def __call__(self, arg0: TensorExpression,
               arg1: TensorExpression) -> "TensorFn":
    return TensorFn(FunctionKind.BINARY, self.fn_name, None, None, [arg0, arg1])

  def __repr__(self):
    return f"{self.fn_name}"


class BinaryFn:
  """Binary function namespace.

  As the integer types are signless, signedness is implement by different
  functions that treat integers as signed or unsigned values.

  Examples:
  - max -> `arith.MaxSIOp`
  - max_unsinged -> `arith.MaxUIOp`
  """
  add = BinaryFnType("add")
  sub = BinaryFnType("sub")
  mul = BinaryFnType("mul")
  max_signed = BinaryFnType("max_signed")
  min_signed = BinaryFnType("min_signed")
  max_unsigned = BinaryFnType("max_unsigned")
  min_unsigned = BinaryFnType("min_unsigned")


class TypeFnType:
  """Type conversion function.

  A type conversion function takes a target type and a tensor expression and
  returns the casted tensor expression.
  """

  def __init__(self, fn_name: str):
    self.fn_name = fn_name

  def __call__(self, type_var: TypeVar, arg: TensorExpression) -> "TensorFn":
    return TensorFn(FunctionKind.TYPE, self.fn_name, None, type_var, [arg])

  def __repr__(self):
    return f"{self.fn_name}"


class TypeFn:
  """Type conversion function namespace.

  As the integer types are signless, signedness is implement by different cast
  functions that treat integers as signed (`cast_signed`) or unsigned
  (`cast_unsigned`) values.

  Examples:
  - cast_signed(I32 -> I64) -> `arith.ExtSIOp`
  - cast_unsigned(I32 -> I64) -> `arith.ExtUIOp`
  """
  cast_signed = TypeFnType("cast_signed")
  cast_unsigned = TypeFnType("cast_unsigned")


class ReduceFnUse:
  """Reduction function use.

  A reduction use specifies the reduction function and dimensions.
  """

  def __init__(self, binary_fn: Optional[BinaryFnType],
               binary_attr: Optional["BinaryFnAttrDef"], *reduce_dims: DimDef):
    if bool(binary_fn) + bool(binary_attr) != 1:
      raise ValueError("One of 'binary_fn', 'binary_attr' must be specified")
    self.binary_fn = binary_fn
    self.binary_attr = binary_attr
    self.reduce_dims = reduce_dims

  def __call__(self, *args: TensorExpression) -> "TensorReduceFn":
    return TensorReduceFn(self, args)

  def __repr__(self):
    fn = self.binary_fn if self.binary_fn else self.binary_attr
    return (
        f"reduce_{repr(fn)}({', '.join(repr(d) for d in self.reduce_dims)})")


class ReduceFnType:
  """Reduction function.

  A binary function that reduces its RHS into its LHS.
  """

  def __init__(self, binary_fn: BinaryFnType):
    if not isinstance(binary_fn, BinaryFnType):
      raise ValueError(f"Reduce expected a BinaryFnType but got {binary_fn}")
    self.binary_fn = binary_fn

  def __getitem__(self, reduce_dims: Tuple[DimDef]) -> ReduceFnUse:
    return ReduceFnUse(self.binary_fn, None, *reduce_dims)

  def __repr__(self):
    return f"reduce_{repr(self.binary_fn)}"


class ReduceFn:
  add = ReduceFnType(BinaryFn.add)
  mul = ReduceFnType(BinaryFn.mul)
  max_signed = ReduceFnType(BinaryFn.max_signed)
  min_signed = ReduceFnType(BinaryFn.min_signed)
  max_unsigned = ReduceFnType(BinaryFn.max_unsigned)
  min_unsigned = ReduceFnType(BinaryFn.min_unsigned)


###############################################################################
# Operand definitions.
###############################################################################


class OperandKind(Enum):
  INPUT_TENSOR = 0
  SCALAR = 1
  OUTPUT_TENSOR = 2
  INDEX_ATTR = 3
  UNARY_FN_ATTR = 4
  BINARY_FN_ATTR = 5
  TYPE_FN_ATTR = 6


class OperandDef:
  """Definition of an operand passed to an operation.

  Keep the meta information of Tensor, Scalar, and Attribute operands and
  provide the shared registration functionality.
  """

  def __init__(self,
               kind: OperandKind,
               type_var: Optional[TypeVar] = None,
               size_exprs: Optional[Sequence[AffineExprDef]] = None,
               index_dims: Optional[Sequence[DimDef]] = None,
               default_indices: Optional[Sequence[int]] = None,
               default_fn: Optional[str] = None):
    if type_var and not isinstance(type_var, TypeVar):
      raise ValueError(
          f"OperandDef requires a TypeVar but got {repr(type_var)}")
    self.owner = None  # type: Optional["LinalgOpDef"]
    self.type_var = type_var
    self.size_exprs = size_exprs
    self.index_dims = index_dims
    self.default_indices = default_indices
    self.default_fn = default_fn
    self.kind = kind
    self.name = None  # type: Optional[str]
    self.registered_index = -1  # type: int

  def attach(self, index: int, name: str, owner: "LinalgOpDef"):
    if self.owner:
      raise ValueError(f"OperandDef already registered with an op: {self}")
    self.registered_index = index
    self.name = name
    self.owner = owner

  def is_input(self) -> bool:
    return (self.kind == OperandKind.SCALAR or
            self.kind == OperandKind.INPUT_TENSOR)

  def is_tensor(self) -> bool:
    return (self.kind == OperandKind.INPUT_TENSOR or
            self.kind == OperandKind.OUTPUT_TENSOR)

  def is_attribute(self) -> bool:
    return (self.kind == OperandKind.INDEX_ATTR or
            self.kind == OperandKind.UNARY_FN_ATTR or
            self.kind == OperandKind.BINARY_FN_ATTR or
            self.kind == OperandKind.TYPE_FN_ATTR)

  def __hash__(self):
    return hash(id(self))

  def __repr__(self):
    return (f"{self.name}:OperandDef(kind={self.kind.name}, "
            f"type={repr(self.type_var)}, size_exprs={self.size_exprs}, "
            f"index_dims={self.index_dims}, "
            f"default_indices={self.default_indices}, "
            f"default_fn={self.default_fn})")


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
    kind = OperandKind.OUTPUT_TENSOR if output else OperandKind.INPUT_TENSOR
    self.operand_def = OperandDef(
        kind, type_var=type_var, size_exprs=shape, index_dims=index_dims)

  def __getitem__(self, dims: Sequence[AffineExprDef]) -> TensorUse:
    assert self.operand_def.owner, "TensorDef is not registered with an op"
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

  def __setitem__(self, dims: Sequence[AffineExprDef], value: TensorExpression):
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
    self.operand_def = OperandDef(OperandKind.SCALAR, type_var=type_var)

  @property
  def scalar_name(self) -> str:
    name = self.operand_def.name
    assert name is not None, "ScalarDef not registered with an op"
    return name

  def to_scalar_expression(self) -> ScalarExpression:
    return ScalarArg(self.scalar_name).expr()


class IndexAttrDef:
  """Index attribute definition.

  Index attributes provide a way to define and set symbols that can be used in
  indexing expressions. Every attribute specifies a tuple of symbols that at
  compile-time are replaced by integer values as well as their default values.
  """

  def __init__(self, *sizes: SymbolDef, default: Sequence[int]):
    if any(not isinstance(size, SymbolDef) for size in sizes):
      raise ValueError(f"IndexAttrDef requires sizes of type SymbolDef "
                       f"but got {sizes}")
    if any(not isinstance(default_val, int) for default_val in default):
      raise ValueError(f"IndexAttrDef requires default values of type int "
                       f"but got {default}")
    if len(sizes) != len(default):
      raise ValueError(f"IndexAttrDef expects {len(sizes)} default values "
                       f"but got {len(default)}")
    self.operand_def = OperandDef(
        OperandKind.INDEX_ATTR, size_exprs=sizes, default_indices=default)


class UnaryFnAttrDef:
  """Unary function attribute definition.

  Unary function attributes provide a way to make the arithmetic computation
  parametrizable. Every attribute specifies a default unary function
  that may be overwritten at operation instantiation time.
  """

  def __init__(self, default: "UnaryFnType"):
    if not isinstance(default, UnaryFnType):
      raise ValueError(f"UnaryFnAttrDef requires default of type UnaryFnType "
                       f"but got {default}")
    self.operand_def = OperandDef(
        OperandKind.UNARY_FN_ATTR, default_fn=default.fn_name)

  def __call__(self, arg: TensorExpression) -> TensorFn:
    return TensorFn(FunctionKind.UNARY, None, self.operand_def, None, [arg])


class BinaryFnAttrDef:
  """Binary function attribute definition.

  Binary function attributes provide a way to make the arithmetic computation
  parametrizable. Every attribute specifies a default binary function
  that may be overwritten at operation instantiation time.
  """

  def __init__(self, default: "BinaryFnType"):
    if not isinstance(default, BinaryFnType):
      raise ValueError(f"BinaryFnAttrDef requires default of type BinaryFnType "
                       f"but got {default}")
    self.operand_def = OperandDef(
        OperandKind.BINARY_FN_ATTR, default_fn=default.fn_name)

  def __call__(self, arg0: TensorExpression,
               arg1: TensorExpression) -> TensorFn:
    return TensorFn(FunctionKind.BINARY, None, self.operand_def, None,
                    [arg0, arg1])

  def __getitem__(self, reduce_dims: Tuple[DimDef]) -> ReduceFnUse:
    return ReduceFnUse(None, self, *reduce_dims)


class TypeFnAttrDef:
  """Type conversion function attribute definition.

  Type conversion function attributes provide a way to make type conversions
  parameterizable. Every attribute specifies a default type conversion function
  that may be overwritten at operation instantiation time.
  """

  def __init__(self, default: "TypeFnType"):
    if not isinstance(default, TypeFnType):
      raise ValueError(f"TypeFnAttrDef requires default of type TypeFnType "
                       f"but got {default}")
    self.operand_def = OperandDef(
        OperandKind.TYPE_FN_ATTR, default_fn=default.fn_name)

  def __call__(self, type_var: TypeVar, arg: TensorExpression) -> TensorFn:
    return TensorFn(FunctionKind.TYPE, None, self.operand_def, type_var, [arg])


###############################################################################
# Operation definition.
###############################################################################


class Comprehension:
  """Represents a single comprehension."""

  def __init__(self, *bindings: Tuple[TensorUse, TensorExpression]):
    self.definitions = list()  # List[TensorUse]
    self.values = list()  # List[TensorExpression]

    # Find the lhs to reduction rhs.
    for assign, value in bindings:
      if isinstance(value, TensorReduceFn):
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
      if isinstance(use, TensorReduceFn):
        result.add(use.reduce_use.reduce_dims)
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


class OpInterfaceDef:
  """An interface that an op implements."""

  def __init__(self, cpp_name: str):
    self.cpp_name = cpp_name


ContractionOpInterface = OpInterfaceDef("LinalgContractionOpInterface")
ConvolutionOpInterface = OpInterfaceDef("LinalgConvolutionOpInterface")


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
    structured_op_methods = [
        "inputs", "outputs", "result_tensors", "region", "iterator_types",
        "indexing_maps", "getRegionBuilder", "getLibraryCallName"
    ]
    if operand.is_attribute() and name in structured_op_methods:
      raise ValueError(f"The attribute name {name} conflicts with a structured "
                       f"op method name")
    # Ensure output tensors are registered after input tensors and scalars and
    # attributes are registered after all other operand types.
    if operand.is_input() and any(
        not op_def.is_input() for op_def in self.registered_operands.values()):
      raise ValueError(f"Input {name} registered after an output or attribute")
    if operand.kind == OperandKind.OUTPUT_TENSOR and any(
        op_def.is_attribute() for op_def in self.registered_operands.values()):
      raise ValueError(f"Output {name} registered after an attribute")
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
