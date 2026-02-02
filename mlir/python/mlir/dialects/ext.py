#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import (
    Dict,
    List,
    Union,
    Tuple,
    Any,
    Optional,
    Callable,
    TypeVar,
    get_origin,
    get_args,
)
from collections.abc import Sequence
from dataclasses import dataclass
from inspect import Parameter, Signature
from types import UnionType
from . import irdl
from ._ods_common import _cext, segmented_accessor
from .irdl import Variadicity
from ..passmanager import PassManager

ir = _cext.ir

__all__ = [
    "Dialect",
    "Operand",
    "Result",
]

Operand = ir.Value
Result = ir.OpResult


class ConstraintLoweringContext:
    def __init__(self):
        self._cache: Dict[str, ir.Value] = {}

    def lower(self, type_) -> ir.Value:
        """
        Lower a type hint (e.g. `Any`, `IntegerType[32]`, `IntegerAttr | StringAttr`) into IRDL ops.
        """

        if type(type_) is TypeVar:
            if type_.__name__ in self._cache:
                return self._cache[type_.__name__]
            v = self._lower(type_.__bound__ or Any)
            self._cache[type_.__name__] = v
        else:
            v = self._lower(type_)
        return v

    def _lower(self, type_) -> ir.Value:
        origin = get_origin(type_)
        if origin is UnionType or origin is Union:
            return irdl.any_of(self.lower(arg) for arg in get_args(type_))
        elif type_ is Any:
            return irdl.any()
        elif isinstance(type_, TypeVar):
            return self.lower(type_)
        elif origin and issubclass(origin, ir.Type):
            # `origin.get` is to construct an instance of MLIR type.
            t = origin.get(*get_args(type_))
            return irdl.is_(ir.TypeAttr.get(t))
        elif origin and issubclass(origin, ir.Attribute):
            # `origin.get` is to construct an instance of MLIR attribute.
            attr = origin.get(*get_args(type_))
            return irdl.is_(attr)
        elif issubclass(type_, ir.Type):
            return irdl.base(base_name=f"!{type_.type_name}")
        elif issubclass(type_, ir.Attribute):
            return irdl.base(base_name=f"#{type_.attr_name}")

        raise TypeError(f"unsupported type in constraints: {type_}")


def infer_type(type_) -> Optional[Callable[[], ir.Type]]:
    """
    A function to infer ir.Type from type annotation.
    Returns a callable that returns the inferred ir.Type,
    or None if the type cannot be inferred.
    We use callables so that MLIR contexts are not required
    while calling this function.
    """

    origin = get_origin(type_)
    if origin and issubclass(origin, ir.Type):
        # `origin.get` is to construct an instance of MLIR type/attribute.
        return lambda: origin.get(*get_args(type_))
    elif isinstance(type_, TypeVar):
        return infer_type(type_.__bound__)
    return None


@dataclass
class FieldDef:
    """
    Base class for kinds of fields that can occur in an `Operation`'s definition.
    """

    name: str
    constraint: Any
    variadicity: Variadicity

    @staticmethod
    def from_type_hint(name, type_) -> "FieldDef":
        variadicity = Variadicity.single
        if inner := match_optional(type_):
            variadicity = Variadicity.optional
            type_ = inner
        elif get_origin(type_) is Sequence:
            variadicity = Variadicity.variadic
            type_ = get_args(type_)[0]

        origin = get_origin(type_)
        if origin is ir.OpResult:
            return ResultDef(name, get_args(type_)[0], variadicity)
        elif origin is ir.Value:
            return OperandDef(name, get_args(type_)[0], variadicity)
        elif issubclass(origin or type_, ir.Attribute):
            return AttributeDef(name, type_, variadicity)
        raise TypeError(f"unsupported type in operation definition: {type_}")


@dataclass
class OperandDef(FieldDef):
    pass


@dataclass
class ResultDef(FieldDef):
    pass


@dataclass
class AttributeDef(FieldDef):
    def __post_init__(self):
        if self.variadicity != Variadicity.single:
            raise ValueError("optional attribute is not supported in IRDL")


def partition_fields(
    fields: List[FieldDef],
) -> Tuple[List[OperandDef], List[AttributeDef], List[ResultDef]]:
    operands = [i for i in fields if isinstance(i, OperandDef)]
    attrs = [i for i in fields if isinstance(i, AttributeDef)]
    results = [i for i in fields if isinstance(i, ResultDef)]
    return operands, attrs, results


def normalize_value_range(
    value_range: Union[ir.OpOperandList, ir.OpResultList],
    variadicity: Variadicity,
) -> ir.Value | ir.OpOperandList | ir.OpResultList | None:
    if variadicity == Variadicity.single:
        return value_range[0]
    if variadicity == Variadicity.optional:
        return value_range[0] if len(value_range) > 0 else None
    return value_range


def match_optional(type_) -> Optional[Any]:
    """
    Try to match type hint like `Optional[T]`, `T | None` or `None | T`.
    Returns the `T` inside `Optional[T]` if matched.
    Returns `None` if not matched.
    """

    origin = get_origin(type_)
    args = get_args(type_)
    if (
        (origin is Union or origin is UnionType)
        and len(args) == 2
        and type(None) in args
    ):
        return args[0] if args[1] is type(None) else args[1]

    return None


class Operation(ir.OpView):
    """
    Base class of Python-defined operation.

    NOTE: Usually you don't need to use it directly.
    Use `Dialect` and `.Operation` of `Dialect` subclasses instead.
    """

    @classmethod
    def __init_subclass__(cls, *, name: str = None, **kwargs):
        """
        This method is to perform all magic to make a `Operation` subclass works like a dataclass, like:
        - generate the method to emit IRDL operations,
        - generate `__init__` method as an operation builder function,
        - generate operand, result and attribute accessors
        """

        super().__init_subclass__(**kwargs)

        fields = []

        for base in cls.__bases__:
            if hasattr(base, "_fields"):
                fields.extend(base._fields)
        for key, value in cls.__annotations__.items():
            field = FieldDef.from_type_hint(key, value)
            fields.append(field)

        cls._fields = fields

        # for subclasses without "name" parameter,
        # just treat them as normal classes
        if not name:
            return

        op_name = name
        cls._op_name = op_name
        dialect_name = cls._dialect_name
        dialect_obj = cls._dialect_obj

        cls._generate_class_attributes(dialect_name, op_name, fields)
        cls._generate_init_method(fields)
        operands, attrs, results = partition_fields(fields)
        cls._generate_attr_properties(attrs)
        cls._generate_operand_properties(operands)
        cls._generate_result_properties(results)

        dialect_obj.operations.append(cls)

    @staticmethod
    def _variadicity_to_segment(variadicity: Variadicity) -> int:
        return {Variadicity.variadic: -1, Variadicity.optional: 0}.get(variadicity, 1)

    @staticmethod
    def _generate_segments(
        operands_or_results: List[Union[OperandDef, ResultDef]],
    ) -> List[int]:
        if any(i.variadicity != Variadicity.single for i in operands_or_results):
            return [
                Operation._variadicity_to_segment(i.variadicity)
                for i in operands_or_results
            ]
        return None

    @staticmethod
    def _generate_init_signature(
        fields: List[FieldDef], can_infer_types: bool
    ) -> Signature:
        result_args = (
            [] if can_infer_types else [i for i in fields if isinstance(i, ResultDef)]
        )
        # results are placed at the beginning of the parameter list,
        # but operands and attributes can appear in any relative order.
        args = result_args + [i for i in fields if not isinstance(i, ResultDef)]
        positional_args = [
            i.name for i in args if i.variadicity != Variadicity.optional
        ]
        optional_args = [i.name for i in args if i.variadicity == Variadicity.optional]

        params = [Parameter("self", Parameter.POSITIONAL_ONLY)]
        for i in positional_args:
            params.append(Parameter(i, Parameter.POSITIONAL_OR_KEYWORD))
        for i in optional_args:
            params.append(Parameter(i, Parameter.KEYWORD_ONLY, default=None))
        params.append(Parameter("loc", Parameter.KEYWORD_ONLY, default=None))
        params.append(Parameter("ip", Parameter.KEYWORD_ONLY, default=None))

        return Signature(params)

    @classmethod
    def _generate_init_method(cls, fields: List[FieldDef]) -> None:
        operands, attrs, results = partition_fields(fields)
        inferred_types = [infer_type(i.constraint) for i in results]

        # we infer result types only when all result types can be inferred
        # and all results are single (not optional or variadic)
        can_infer_types = all(inferred_types) and all(
            i.variadicity == Variadicity.single for i in results
        )

        init_sig = cls._generate_init_signature(fields, can_infer_types)

        def __init__(*args, **kwargs):
            bound = init_sig.bind(*args, **kwargs)
            bound.apply_defaults()
            args = bound.arguments

            _operands = [args[operand.name] for operand in operands]
            _results = (
                [t() for t in inferred_types]
                if can_infer_types
                else [args[result.name] for result in results]
            )
            _attributes = dict(
                (attr.name, args[attr.name])
                for attr in attrs
                if args[attr.name] is not None
            )
            _regions = None
            _ods_successors = None
            self = args["self"]
            super(Operation, self).__init__(
                self.OPERATION_NAME,
                self._ODS_REGIONS,
                self._ODS_OPERAND_SEGMENTS,
                self._ODS_RESULT_SEGMENTS,
                attributes=_attributes,
                results=_results,
                operands=_operands,
                successors=_ods_successors,
                regions=_regions,
                loc=args["loc"],
                ip=args["ip"],
            )

        __init__.__signature__ = init_sig
        cls.__init__ = __init__

    @classmethod
    def _generate_class_attributes(
        cls, dialect_name: str, op_name: str, fields: List[FieldDef]
    ) -> None:
        operands, attrs, results = partition_fields(fields)

        operand_segments = cls._generate_segments(operands)
        result_segments = cls._generate_segments(results)

        cls.OPERATION_NAME = f"{dialect_name}.{op_name}"
        cls._ODS_REGIONS = (0, True)
        cls._ODS_OPERAND_SEGMENTS = operand_segments
        cls._ODS_RESULT_SEGMENTS = result_segments

    @classmethod
    def _generate_attr_properties(cls, attrs: List[AttributeDef]) -> None:
        for attr in attrs:
            setattr(
                cls,
                attr.name,
                property(lambda self, name=attr.name: self.attributes[name]),
            )

    @classmethod
    def _generate_operand_properties(cls, operands: List[OperandDef]) -> None:
        for i, operand in enumerate(operands):
            if cls._ODS_OPERAND_SEGMENTS:

                def getter(self, i=i, operand=operand):
                    operand_range = segmented_accessor(
                        self.operation.operands,
                        self.operation.attributes["operandSegmentSizes"],
                        i,
                    )
                    return normalize_value_range(operand_range, operand.variadicity)

                setattr(cls, operand.name, property(getter))
            else:
                setattr(cls, operand.name, property(lambda self, i=i: self.operands[i]))

    @classmethod
    def _generate_result_properties(cls, results: List[ResultDef]) -> None:
        for i, result in enumerate(results):
            if cls._ODS_RESULT_SEGMENTS:

                def getter(self, i=i, result=result):
                    result_range = segmented_accessor(
                        self.operation.results,
                        self.operation.attributes["resultSegmentSizes"],
                        i,
                    )
                    return normalize_value_range(result_range, result.variadicity)

                setattr(cls, result.name, property(getter))
            else:
                setattr(cls, result.name, property(lambda self, i=i: self.results[i]))

    @classmethod
    def _emit_operation(cls) -> None:
        ctx = ConstraintLoweringContext()
        operands, attrs, results = partition_fields(cls._fields)

        op = irdl.operation_(cls._op_name)
        with ir.InsertionPoint(op.body):
            if operands:
                irdl.operands_(
                    [ctx.lower(i.constraint) for i in operands],
                    [i.name for i in operands],
                    [i.variadicity for i in operands],
                )
            if attrs:
                irdl.attributes_(
                    [ctx.lower(i.constraint) for i in attrs],
                    [i.name for i in attrs],
                )
            if results:
                irdl.results_(
                    [ctx.lower(i.constraint) for i in results],
                    [i.name for i in results],
                    [i.variadicity for i in results],
                )


class Dialect(ir.Dialect):
    """
    Base class of a Python-defined dialect.

    It can be used like the following example:
    ```python
    class MyInt(Dialect, name="myint"):
        pass

    i32 = IntegerType[32]

    class ConstantOp(MyInt.Operation, name="constant"):
        value: IntegerAttr
        cst: Result[i32]

    class AddOp(MyInt.Operation, name="add"):
        lhs: Operand[i32]
        rhs: Operand[i32]
        res: Result[i32]
    ```
    """

    @classmethod
    def __init_subclass__(cls, name: str, **kwargs):
        cls.name = name
        cls.DIALECT_NAMESPACE = name
        cls.operations = []
        cls.Operation = type(
            "Operation",
            (Operation,),
            {"_dialect_obj": cls, "_dialect_name": name},
        )

    @classmethod
    def _emit_dialect(cls) -> None:
        d = irdl.dialect(cls.name)
        with ir.InsertionPoint(d.body):
            for op in cls.operations:
                op._emit_operation()

    @classmethod
    def _emit_module(cls) -> ir.Module:
        m = ir.Module.create()
        with ir.InsertionPoint(m.body):
            cls._emit_dialect()

        return m

    @classmethod
    def load(cls) -> None:
        if hasattr(cls, "_mlir_module"):
            raise RuntimeError(f"Dialect {cls.name} is already loaded.")

        mlir_module = cls._emit_module()

        pm = PassManager()
        pm.add("canonicalize, cse")
        pm.run(mlir_module.operation)

        irdl.load_dialects(mlir_module)

        _cext.register_dialect(cls)

        for op in cls.operations:
            _cext.register_operation(cls)(op)

        cls._mlir_module = mlir_module
