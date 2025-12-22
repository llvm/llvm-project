#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Dict, List, Union, Callable, Tuple, ClassVar
from dataclasses import dataclass
from inspect import Parameter, Signature
from types import SimpleNamespace
from abc import ABC, abstractmethod
from contextlib import nullcontext
from ...dialects import irdl
from .._ods_common import _cext, segmented_accessor
from . import Variadicity

ir = _cext.ir

__all__ = [
    "Variadicity",
    "Is",
    "AnyOf",
    "AllOf",
    "Any",
    "BaseName",
    "BaseRef",
    "Operand",
    "Result",
    "Attribute",
    "Dialect",
]


class ConstraintExpr(ABC):
    @abstractmethod
    def _lower(self, ctx: "ConstraintLoweringContext") -> ir.Value:
        pass

    def __or__(self, other: "ConstraintExpr") -> "ConstraintExpr":
        return AnyOf(self, other)

    def __and__(self, other: "ConstraintExpr") -> "ConstraintExpr":
        return AllOf(self, other)


class ConstraintLoweringContext:
    def __init__(self):
        # Cache so that the same ConstraintExpr instance reuses its SSA value.
        self._cache: Dict[int, ir.Value] = {}

    def lower(self, expr: ConstraintExpr) -> ir.Value:
        key = id(expr)
        if key in self._cache:
            return self._cache[key]
        v = expr._lower(self)
        self._cache[key] = v
        return v


class Is(ConstraintExpr):
    def __init__(self, val: Callable[..., Union[ir.Attribute, ir.Type]]):
        self.val = val
        self.args = []
        self.kwargs = {}

    def __call__(self, *args, **kwargs) -> "Is":
        self.args.extend(args)
        self.kwargs.update(kwargs)
        return self

    def __class_getitem__(
        cls, val: Callable[..., Union[ir.Attribute, ir.Type]]
    ) -> "Is":
        return cls(val)

    def _lower(self, ctx: ConstraintLoweringContext) -> ir.Value:
        # for most attributes and types, they are created via `.get` method,
        # here we can just omit the `.get` suffix for convenience
        if isinstance(self.val, type) and hasattr(self.val, "get"):
            self.val = self.val.get

        val = self.val(*self.args, **self.kwargs)

        if isinstance(val, ir.Type):
            val = ir.TypeAttr.get(val)

        return irdl.is_(val)


class AnyOf(ConstraintExpr):
    def __init__(self, *exprs: ConstraintExpr):
        self.exprs = exprs

    def _lower(self, ctx: ConstraintLoweringContext) -> ir.Value:
        return irdl.any_of(ctx.lower(expr) for expr in self.exprs)


class AllOf(ConstraintExpr):
    def __init__(self, *exprs: ConstraintExpr):
        self.exprs = exprs

    def _lower(self, ctx: ConstraintLoweringContext) -> ir.Value:
        return irdl.all_of(ctx.lower(expr) for expr in self.exprs)


class Any(ConstraintExpr):
    def _lower(self, ctx: ConstraintLoweringContext) -> ir.Value:
        return irdl.any()


class BaseName(ConstraintExpr):
    def __init__(self, name: str):
        self.name = name

    def _lower(self, ctx: ConstraintLoweringContext) -> ir.Value:
        return irdl.base(base_name=self.name)


class BaseRef(ConstraintExpr):
    def __init__(self, ref):
        self.ref = ref

    def _lower(self, ctx: ConstraintLoweringContext) -> ir.Value:
        return irdl.base(base_ref=self.ref)


class FieldDef:
    pass


@dataclass
class Operand(FieldDef):
    constraint: ConstraintExpr
    variadicity: Variadicity = Variadicity.single


@dataclass
class Result(FieldDef):
    constraint: ConstraintExpr
    variadicity: Variadicity = Variadicity.single


@dataclass
class Attribute(FieldDef):
    constraint: ConstraintExpr
    variadicity: ClassVar[Variadicity] = Variadicity.single


def partition_fields(
    fields: List[FieldDef],
) -> Tuple[List[Operand], List[Attribute], List[Result]]:
    operands = [i for i in fields if isinstance(i, Operand)]
    attrs = [i for i in fields if isinstance(i, Attribute)]
    results = [i for i in fields if isinstance(i, Result)]
    return operands, attrs, results


def normalize_value_range(
    value_range: Union[ir.OpOperandList, ir.OpResultList],
    variadicity: Variadicity,
):
    if variadicity == Variadicity.single:
        return value_range[0]
    if variadicity == Variadicity.optional:
        return value_range[0] if len(value_range) > 0 else None
    return value_range


class Operation(ir.OpView):
    @classmethod
    def __init_subclass__(cls, *, name: str = None, **kwargs):
        super().__init_subclass__(**kwargs)

        # for subclasses without "name" parameter,
        # just treat them as normal classes
        if not name:
            return

        op_name = name
        cls._op_name = op_name
        dialect_name = cls._dialect_name
        dialect_obj = cls._dialect_obj

        fields = []
        cls._fields = fields

        for base in reversed(cls.__mro__):
            for key, value in base.__dict__.items():
                if isinstance(value, FieldDef):
                    setattr(value, "name", key)
                    fields.append(value)

        cls._generate_class_attributes(dialect_name, op_name, fields)
        cls._generate_init_method(fields)
        operands, attrs, results = partition_fields(fields)
        cls._generate_attr_properties(attrs)
        cls._generate_operand_properties(operands)
        cls._generate_result_properties(results)

        dialect_obj.operations.append(cls)

    @staticmethod
    def _variadicity_to_segment(variadicity: Variadicity) -> int:
        if variadicity == Variadicity.variadic:
            return -1
        if variadicity == Variadicity.optional:
            return 0
        return 1

    @staticmethod
    def _generate_segments(
        operands_or_results: List[Union[Operand, Result]],
    ) -> List[int]:
        if any(i.variadicity != Variadicity.single for i in operands_or_results):
            return [
                Operation._variadicity_to_segment(i.variadicity)
                for i in operands_or_results
            ]
        return None

    @staticmethod
    def _generate_init_signature(fields: List[FieldDef]) -> Signature:
        # results are placed at the beginning of the parameter list,
        # but operands and attributes can appear in any relative order.
        args = [i for i in fields if isinstance(i, Result)] + [
            i for i in fields if not isinstance(i, Result)
        ]
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
        init_sig = cls._generate_init_signature(fields)
        operands, attrs, results = partition_fields(fields)

        def __init__(*args, **kwargs):
            bound = init_sig.bind(*args, **kwargs)
            bound.apply_defaults()
            args = bound.arguments

            _operands = [args[operand.name] for operand in operands]
            _results = [args[result.name] for result in results]
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
    def _generate_attr_properties(cls, attrs: List[Attribute]) -> None:
        for attr in attrs:
            setattr(
                cls,
                attr.name,
                property(lambda self, name=attr.name: self.attributes[name]),
            )

    @classmethod
    def _generate_operand_properties(cls, operands: List[Operand]) -> None:
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
    def _generate_result_properties(cls, results: List[Result]) -> None:
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
        if hasattr(cls, "mlir_module"):
            raise RuntimeError(f"Dialect {cls.name} is already loaded.")

        mlir_module = cls._emit_module()
        irdl.load_dialects(mlir_module)

        _cext.register_dialect(cls)

        for op in cls.operations:
            _cext.register_operation(cls)(op)

        cls.mlir_module = mlir_module
