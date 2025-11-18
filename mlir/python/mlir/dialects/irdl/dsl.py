#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ...dialects import irdl as _irdl
from .._ods_common import (
    _cext as _ods_cext,
    segmented_accessor as _ods_segmented_accessor,
)
from . import Variadicity
from typing import Dict, List, Union, Callable, Tuple
from dataclasses import dataclass
from inspect import Parameter as _Parameter, Signature as _Signature
from types import SimpleNamespace as _SimpleNameSpace

_ods_ir = _ods_cext.ir


class ConstraintExpr:
    def _lower(self, ctx: "ConstraintLoweringContext") -> _ods_ir.Value:
        raise NotImplementedError()

    def __or__(self, other: "ConstraintExpr") -> "ConstraintExpr":
        return AnyOf(self, other)

    def __and__(self, other: "ConstraintExpr") -> "ConstraintExpr":
        return AllOf(self, other)


class ConstraintLoweringContext:
    def __init__(self):
        # Cache so that the same ConstraintExpr instance reuses its SSA value.
        self._cache: Dict[int, _ods_ir.Value] = {}

    def lower(self, expr: ConstraintExpr) -> _ods_ir.Value:
        key = id(expr)
        if key in self._cache:
            return self._cache[key]
        v = expr._lower(self)
        self._cache[key] = v
        return v


class Is(ConstraintExpr):
    def __init__(self, attr: _ods_ir.Attribute):
        self.attr = attr

    def _lower(self, ctx: ConstraintLoweringContext) -> _ods_ir.Value:
        return _irdl.is_(self.attr)


class IsType(Is):
    def __init__(self, typ: _ods_ir.Type):
        super().__init__(_ods_ir.TypeAttr.get(typ))


class AnyOf(ConstraintExpr):
    def __init__(self, *exprs: ConstraintExpr):
        self.exprs = exprs

    def _lower(self, ctx: ConstraintLoweringContext) -> _ods_ir.Value:
        return _irdl.any_of(ctx.lower(expr) for expr in self.exprs)


class AllOf(ConstraintExpr):
    def __init__(self, *exprs: ConstraintExpr):
        self.exprs = exprs

    def _lower(self, ctx: ConstraintLoweringContext) -> _ods_ir.Value:
        return _irdl.all_of(ctx.lower(expr) for expr in self.exprs)


class Any(ConstraintExpr):
    def _lower(self, ctx: ConstraintLoweringContext) -> _ods_ir.Value:
        return _irdl.any()


class BaseName(ConstraintExpr):
    def __init__(self, name: str):
        self.name = name

    def _lower(self, ctx: ConstraintLoweringContext) -> _ods_ir.Value:
        return _irdl.base(base_name=self.name)


class BaseRef(ConstraintExpr):
    def __init__(self, ref):
        self.ref = ref

    def _lower(self, ctx: ConstraintLoweringContext) -> _ods_ir.Value:
        return _irdl.base(base_ref=self.ref)


class FieldDef:
    def __set_name__(self, owner, name: str):
        self.name = name


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

    def __post_init__(self):
        # just for unified processing,
        # currently optional attribute is not supported by IRDL
        self.variadicity = Variadicity.single


@dataclass
class Operation:
    dialect_name: str
    name: str
    # We store operands and attributes into one list to maintain relative orders
    # among them for generating OpView class.
    operands_and_attrs: List[Union[Operand, Attribute]]
    results: List[Result]

    def _emit(self) -> None:
        op = _irdl.operation_(self.name)
        ctx = ConstraintLoweringContext()

        operands = [i for i in self.operands_and_attrs if isinstance(i, Operand)]
        attrs = [i for i in self.operands_and_attrs if isinstance(i, Attribute)]

        with _ods_ir.InsertionPoint(op.body):
            if operands:
                _irdl.operands_(
                    [ctx.lower(i.constraint) for i in operands],
                    [i.name for i in operands],
                    [i.variadicity for i in operands],
                )
            if attrs:
                _irdl.attributes_(
                    [ctx.lower(i.constraint) for i in attrs],
                    [i.name for i in attrs],
                )
            if self.results:
                _irdl.results_(
                    [ctx.lower(i.constraint) for i in self.results],
                    [i.name for i in self.results],
                    [i.variadicity for i in self.results],
                )

    def _make_op_view_and_builder(self) -> Tuple[type, Callable]:
        operands = [i for i in self.operands_and_attrs if isinstance(i, Operand)]
        attrs = [i for i in self.operands_and_attrs if isinstance(i, Attribute)]

        def variadicity_to_segment(variadicity: Variadicity) -> int:
            if variadicity == Variadicity.variadic:
                return -1
            if variadicity == Variadicity.optional:
                return 0
            return 1

        operand_segments = None
        if any(i.variadicity != Variadicity.single for i in operands):
            operand_segments = [variadicity_to_segment(i.variadicity) for i in operands]

        result_segments = None
        if any(i.variadicity != Variadicity.single for i in self.results):
            result_segments = [
                variadicity_to_segment(i.variadicity) for i in self.results
            ]

        args = self.results + self.operands_and_attrs
        positional_args = [
            i.name for i in args if i.variadicity != Variadicity.optional
        ]
        optional_args = [i.name for i in args if i.variadicity == Variadicity.optional]

        params = [_Parameter("self", _Parameter.POSITIONAL_ONLY)]
        for i in positional_args:
            params.append(_Parameter(i, _Parameter.POSITIONAL_OR_KEYWORD))
        for i in optional_args:
            params.append(_Parameter(i, _Parameter.KEYWORD_ONLY, default=None))
        params.append(_Parameter("loc", _Parameter.KEYWORD_ONLY, default=None))
        params.append(_Parameter("ip", _Parameter.KEYWORD_ONLY, default=None))

        sig = _Signature(params)
        op = self

        class _OpView(_ods_ir.OpView):
            OPERATION_NAME = f"{op.dialect_name}.{op.name}"
            _ODS_REGIONS = (0, True)
            _ODS_OPERAND_SEGMENTS = operand_segments
            _ODS_RESULT_SEGMENTS = result_segments

            def __init__(*args, **kwargs):
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                args = bound.arguments

                _operands = [args[operand.name] for operand in operands]
                _results = [args[result.name] for result in op.results]
                _attributes = dict(
                    (attr.name, args[attr.name])
                    for attr in attrs
                    if args[attr.name] is not None
                )
                _regions = None
                _ods_successors = None
                self = args["self"]
                super(_OpView, self).__init__(
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

            __init__.__signature__ = sig

        for attr in attrs:
            setattr(
                _OpView,
                attr.name,
                property(lambda self, name=attr.name: self.attributes[name]),
            )

        def value_range_getter(
            value_range: Union[_ods_ir.OpOperandList, _ods_ir.OpResultList],
            variadicity: Variadicity,
        ):
            if variadicity == Variadicity.single:
                return value_range[0]
            if variadicity == Variadicity.optional:
                return value_range[0] if len(value_range) > 0 else None
            return value_range

        for i, operand in enumerate(operands):
            if operand_segments:

                def getter(self, i=i, operand=operand):
                    operand_range = _ods_segmented_accessor(
                        self.operation.operands,
                        self.operation.attributes["operandSegmentSizes"],
                        i,
                    )
                    return value_range_getter(operand_range, operand.variadicity)

                setattr(_OpView, operand.name, property(getter))
            else:
                setattr(
                    _OpView, operand.name, property(lambda self, i=i: self.operands[i])
                )
        for i, result in enumerate(self.results):
            if result_segments:

                def getter(self, i=i, result=result):
                    result_range = _ods_segmented_accessor(
                        self.operation.results,
                        self.operation.attributes["resultSegmentSizes"],
                        i,
                    )
                    return value_range_getter(result_range, result.variadicity)

                setattr(_OpView, result.name, property(getter))
            else:
                setattr(
                    _OpView, result.name, property(lambda self, i=i: self.results[i])
                )

        def _builder(*args, **kwargs) -> _OpView:
            return _OpView(*args, **kwargs)

        _builder.__signature__ = _Signature(params[1:])

        return _OpView, _builder


class Dialect:
    def __init__(self, name: str):
        self.name = name
        self.operations: List[Operation] = []
        self.namespace = _SimpleNameSpace()

    def _emit(self) -> None:
        d = _irdl.dialect(self.name)
        with _ods_ir.InsertionPoint(d.body):
            for op in self.operations:
                op._emit()

    def _make_module(self) -> _ods_ir.Module:
        with _ods_ir.Location.unknown():
            m = _ods_ir.Module.create()
            with _ods_ir.InsertionPoint(m.body):
                self._emit()
        return m

    def _make_dialect_class(self) -> type:
        class _Dialect(_ods_ir.Dialect):
            DIALECT_NAMESPACE = self.name

        return _Dialect

    def load(self) -> _SimpleNameSpace:
        _irdl.load_dialects(self._make_module())
        dialect_class = self._make_dialect_class()
        _ods_cext.register_dialect(dialect_class)
        for op in self.operations:
            _ods_cext.register_operation(dialect_class)(op.op_view)
        return self.namespace

    def op(self, name: str) -> Callable[[type], type]:
        def decorator(cls: type) -> type:
            operands_and_attrs: List[Union[Operand, Attribute]] = []
            results: List[Result] = []

            for field in cls.__dict__.values():
                if isinstance(field, Operand) or isinstance(field, Attribute):
                    operands_and_attrs.append(field)
                elif isinstance(field, Result):
                    results.append(field)

            op_def = Operation(self.name, name, operands_and_attrs, results)
            op_view, builder = op_def._make_op_view_and_builder()
            setattr(op_def, "op_view", op_view)
            setattr(op_def, "builder", builder)
            self.operations.append(op_def)
            self.namespace.__dict__[cls.__name__] = op_view
            op_view.__name__ = cls.__name__
            self.namespace.__dict__[name.replace(".", "_")] = builder
            return cls

        return decorator
