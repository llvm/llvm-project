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
from enum import Enum
from . import irdl
from ._ods_common import _cext, segmented_accessor
from .irdl import Variadicity
from ..passmanager import PassManager
from contextlib import nullcontext

ir = _cext.ir

__all__ = [
    "Dialect",
    "Operation",
    "Operand",
    "Result",
    "Region",
    "Type",
    "Attribute",
    "result",
    "infer_result",
    "operand",
    "attribute",
]

Operand = ir.Value
Result = ir.OpResult
Region = ir.Region


def construct_instance(origin, args):
    if not issubclass(origin, ir.Type | ir.Attribute):
        raise TypeError(f"unsupported type in constraints: {origin}")

    # `origin.get` is to construct an instance of MLIR type or attribute.
    return origin.get(
        *(
            (
                construct_instance(get_origin(arg), get_args(arg))
                if get_origin(arg)
                else arg
            )
            for arg in args
        )
    )


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
        elif origin and issubclass(origin, Type | Attribute):
            return irdl.parametric(
                base_type=[origin._dialect_name, origin._name],
                args=[self.lower(arg) for arg in get_args(type_)],
            )
        elif origin and issubclass(origin, ir.Type):
            t = construct_instance(origin, get_args(type_))
            return irdl.is_(ir.TypeAttr.get(t))
        elif origin and issubclass(origin, ir.Attribute):
            attr = construct_instance(origin, get_args(type_))
            return irdl.is_(attr)
        elif issubclass(type_, Type | Attribute):
            return irdl.base(base_ref=[type_._dialect_name, type_._name])
        elif issubclass(type_, ir.Type):
            return irdl.base(base_name=f"!{type_.type_name}")
        elif issubclass(type_, ir.Attribute):
            return irdl.base(base_name=f"#{type_.attr_name}")

        raise TypeError(f"unsupported type in constraints: {type_}")


@dataclass
class FieldSpecifier:
    type_: Any = None
    infer_type: bool = False
    default_is_none: bool = False
    default_factory: Optional[Callable[[], Any]] = None
    kw_only: bool = False

    @property
    def param_kind(self):
        if self.default_is_none or self.default_factory or self.infer_type:
            return ParameterKind.KEYWORD_ONLY_WITH_DEFAULT
        if self.kw_only:
            return ParameterKind.KEYWORD_ONLY_WITHOUT_DEFAULT
        return ParameterKind.POSITIONAL_OR_KEYWORD


def result(
    *,
    default_factory: Optional[Callable[[], Any]] = None,
    kw_only: bool = False,
) -> Result:
    """
    A field specifier for `Result` definitions.
    """

    return FieldSpecifier(
        type_=Result,
        default_factory=default_factory,
        kw_only=kw_only,
    )


def infer_result() -> Result:
    """
    A field specifier for `Result` definitions with type inference enabled.
    """

    return FieldSpecifier(
        type_=Result,
        infer_type=True,
    )


def operand(
    *,
    kw_only: bool = False,
) -> Operand:
    """
    A field specifier for `Operand` definitions.
    """

    return FieldSpecifier(
        type_=Operand,
        kw_only=kw_only,
    )


def attribute(
    *,
    default_factory: Optional[Callable[[], Any]] = None,
    kw_only: bool = False,
) -> ir.Attribute:
    """
    A field specifier for attribute definitions.
    """

    return FieldSpecifier(
        type_=Attribute,
        default_factory=default_factory,
        kw_only=kw_only,
    )


def infer_type_impl(type_) -> Callable[[], ir.Type]:
    """
    A function to infer ir.Type from type annotation.
    Returns a callable that returns the inferred ir.Type.
    We use callables so that MLIR contexts are not required
    while calling this function.
    """

    origin = get_origin(type_)
    if origin and issubclass(origin, ir.Type | ir.Attribute):
        args = [
            infer_type_impl(arg) if get_origin(arg) else lambda: arg
            for arg in get_args(type_)
        ]
        return lambda: origin.get(*[arg() for arg in args])
    elif isinstance(type_, TypeVar):
        return infer_type_impl(type_.__bound__)
    raise TypeError(f"unsupported type for inferring: {type_}")


class ParameterKind(Enum):
    POSITIONAL_OR_KEYWORD = 1
    KEYWORD_ONLY_WITHOUT_DEFAULT = 2
    KEYWORD_ONLY_WITH_DEFAULT = 3


@dataclass
class FieldDef:
    """
    Base class for kinds of fields that can occur in an `Operation`'s definition.
    """

    name: str
    variadicity: Variadicity
    constraint: Any

    param_kind: ParameterKind = ParameterKind.POSITIONAL_OR_KEYWORD

    @staticmethod
    def from_type_hint(name, type_, specifier) -> "FieldDef":
        variadicity = Variadicity.single
        if inner := match_optional(type_):
            variadicity = Variadicity.optional
            type_ = inner
        elif get_origin(type_) is Sequence:
            variadicity = Variadicity.variadic
            type_ = get_args(type_)[0]

        origin = get_origin(type_)
        if origin is ir.OpResult:
            if specifier.type_ and specifier.type_ is not Result:
                raise TypeError(
                    f"only `result` field specifier can be used for result fields"
                )
            constraint = get_args(type_)[0]
            return ResultDef(
                name,
                variadicity,
                constraint,
                param_kind=specifier.param_kind,
                default_factory=specifier.default_factory,
                default_is_none=specifier.default_is_none,
                infer_type=(
                    infer_type_impl(constraint) if specifier.infer_type else None
                ),
            )
        elif origin is ir.Value:
            if specifier.type_ and specifier.type_ is not Operand:
                raise TypeError(
                    f"only `operand` field specifier can be used for operand fields"
                )
            return OperandDef(
                name,
                variadicity,
                get_args(type_)[0],
                param_kind=specifier.param_kind,
                default_is_none=specifier.default_is_none,
            )
        elif type_ is ir.Region:
            if specifier.type_ and specifier.type_ is not Region:
                raise TypeError(
                    f"this field specifier can not be used for region fields"
                )
            return RegionDef(name, variadicity, Any)

        if specifier.type_ and specifier.type_ is not Attribute:
            raise TypeError(
                f"only `attribute` field specifier can be used for attribute fields"
            )
        return AttributeDef(
            name,
            variadicity,
            type_,
            param_kind=specifier.param_kind,
            default_factory=specifier.default_factory,
        )


@dataclass
class OperandDef(FieldDef):
    default_is_none: bool = False

    def __post_init__(self):
        if self.variadicity != Variadicity.optional and self.default_is_none:
            raise ValueError(f"only optional operand can be set to None")


@dataclass
class ResultDef(FieldDef):
    infer_type: Callable[[], ir.Type] | None = None
    default_factory: Optional[Callable[[], Any]] = None
    default_is_none: bool = False

    def __post_init__(self):
        if self.variadicity != Variadicity.optional and self.default_is_none:
            raise ValueError(f"only optional result can be set to None")

        if self.infer_type and self.variadicity != Variadicity.single:
            raise ValueError(
                f"type of variadic or optional result '{self.name}' cannot be inferred"
            )

    def process_type(self, type_):
        if type_:
            return type_

        if self.infer_type:
            return self.infer_type()

        if self.default_factory:
            return self.default_factory()

        return None


@dataclass
class AttributeDef(FieldDef):
    default_factory: Optional[Callable[[], Any]] = None

    def __post_init__(self):
        if self.variadicity != Variadicity.single:
            raise ValueError("optional attribute is not currently supported")
        if (
            self.param_kind == ParameterKind.KEYWORD_ONLY_WITH_DEFAULT
            and not self.default_factory
        ):
            raise ValueError(f"only optional attribute can be set to None")

    def process_attr(self, attr):
        if attr:
            return attr

        if self.default_factory:
            return self.default_factory()

        return None


@dataclass
class RegionDef(FieldDef):
    def __post_init__(self):
        if self.variadicity != Variadicity.single:
            raise ValueError("optional region is not currently supported")


def partition_fields(
    fields: List[FieldDef],
) -> Tuple[List[OperandDef], List[AttributeDef], List[ResultDef], List[RegionDef]]:
    operands = [i for i in fields if isinstance(i, OperandDef)]
    attrs = [i for i in fields if isinstance(i, AttributeDef)]
    results = [i for i in fields if isinstance(i, ResultDef)]
    regions = [i for i in fields if isinstance(i, RegionDef)]
    return operands, attrs, results, regions


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
    Base class of Python-defined operations.

    The following example shows two ways to define operations via this class:
    ```python
    class MyOp(MyDialect.Operation, name=..):
      ...

    class MyOp(Operation, dialect=MyDialect, name=..):
      ...
    ```
    """

    def __init__(*args, **kwargs):
        raise TypeError(
            "This class is a template and cannot be instantiated directly. "
            "Please use a subclass that defines the operation."
        )

    @classmethod
    def __init_subclass__(
        cls,
        *,
        name: str | None = None,
        traits: list[type] | None = None,
        dialect: type | None = None,
        **kwargs,
    ):
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
            # if the class variable is not defined, we treat it as a default specifier;
            # if it is assigned with `None`, we treat it as a specifier with `default_is_none=True`.
            # e.g. x : int         # default specifier
            #      y : int = None  # specifier with default_is_none=True
            specifier = cls.__dict__.get(key, FieldSpecifier()) or FieldSpecifier(
                default_is_none=True
            )
            # treat all other values as invalid
            if not isinstance(specifier, FieldSpecifier):
                raise TypeError(
                    f"the field specifier of field '{key}' is not supported"
                )
            field = FieldDef.from_type_hint(key, value, specifier)
            fields.append(field)

        cls._fields = fields

        traits = traits or []

        for base in cls.__bases__:
            if hasattr(base, "_traits"):
                traits = base._traits + traits

        cls._traits = traits

        if dialect:
            if hasattr(cls, "_dialect_obj"):
                raise RuntimeError(
                    f"This operation has already been attached to dialect '{cls._dialect_obj.DIALECT_NAMESPACE}'."
                )
            cls._dialect_obj = dialect

        # for subclasses without "name" parameter,
        # just treat them as normal classes
        if not name:
            return

        if not hasattr(cls, "_dialect_obj"):
            raise RuntimeError(
                "Operation subclasses must either inherit from a Dialect's Operation subclass "
                "or provide the dialect as a class keyword argument."
            )

        op_name = name
        cls._op_name = op_name
        dialect_name = cls._dialect_obj.DIALECT_NAMESPACE
        dialect_obj = cls._dialect_obj

        cls._generate_class_attributes(dialect_name, op_name, fields)
        cls._generate_init_method(fields)
        operands, attrs, results, regions = partition_fields(fields)
        cls._generate_attr_properties(attrs)
        cls._generate_operand_properties(operands)
        cls._generate_result_properties(results)
        cls._generate_region_properties(regions)

        cls.Adaptor = type(
            "Adaptor",
            (OperationAdator,),
            dict(),
            operation=cls,
        )

        dialect_obj.operations.append(cls)

    @staticmethod
    def _variadicity_to_segment(variadicity: Variadicity) -> int:
        return {Variadicity.variadic: -1, Variadicity.optional: 0}.get(variadicity, 1)

    @staticmethod
    def _generate_segments(
        operands_or_results: List[Union[OperandDef, ResultDef]],
    ) -> List[int] | None:
        if any(i.variadicity != Variadicity.single for i in operands_or_results):
            return [
                Operation._variadicity_to_segment(i.variadicity)
                for i in operands_or_results
            ]
        return None

    @staticmethod
    def _generate_init_signature(fields: List[FieldDef]) -> Signature:
        args = [i for i in fields if not isinstance(i, RegionDef)]

        params = [Parameter("self", Parameter.POSITIONAL_ONLY)]

        for i in args:
            match i.param_kind:
                case ParameterKind.POSITIONAL_OR_KEYWORD:
                    params.append(Parameter(i.name, Parameter.POSITIONAL_OR_KEYWORD))
                case ParameterKind.KEYWORD_ONLY_WITH_DEFAULT:
                    params.append(
                        Parameter(i.name, Parameter.KEYWORD_ONLY, default=None)
                    )
                case ParameterKind.KEYWORD_ONLY_WITHOUT_DEFAULT:
                    params.append(Parameter(i.name, Parameter.KEYWORD_ONLY))

        params.append(Parameter("loc", Parameter.KEYWORD_ONLY, default=None))
        params.append(Parameter("ip", Parameter.KEYWORD_ONLY, default=None))

        return Signature(params)

    @classmethod
    def _generate_init_method(cls, fields: List[FieldDef]) -> None:
        operands, attrs, results, regions = partition_fields(fields)

        init_sig = cls._generate_init_signature(fields)

        def __init__(*args, **kwargs):
            bound = init_sig.bind(*args, **kwargs)
            bound.apply_defaults()
            args = bound.arguments

            _operands = [args[operand.name] for operand in operands]
            _results = [result.process_type(args[result.name]) for result in results]
            _attributes = dict(
                (attr.name, attr.process_attr(args[attr.name])) for attr in attrs
            )
            _regions = len(regions) or None
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
        operands, attrs, results, regions = partition_fields(fields)

        operand_segments = cls._generate_segments(operands)
        result_segments = cls._generate_segments(results)

        cls.OPERATION_NAME = f"{dialect_name}.{op_name}"
        cls._ODS_REGIONS = (len(regions), True)
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
    def _generate_region_properties(cls, regions: List[RegionDef]) -> None:
        for i, region in enumerate(regions):
            setattr(
                cls,
                region.name,
                property(lambda self, i=i: self.regions[i]),
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
    def _attach_traits(cls) -> None:
        for trait in cls._traits:
            trait.attach(cls.OPERATION_NAME)

        if hasattr(cls, "verify_invariants") or hasattr(
            cls, "verify_region_invariants"
        ):
            ir.DynamicOpTrait.attach(cls.OPERATION_NAME, cls)

    @classmethod
    def _emit_operation(cls) -> None:
        ctx = ConstraintLoweringContext()
        operands, attrs, results, regions = partition_fields(cls._fields)

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
            if regions:
                irdl.regions_(
                    [irdl.region([]) for _ in regions],
                    [i.name for i in regions],
                )


class OperationAdator(ir.OpAdaptor):
    @classmethod
    def __init_subclass__(cls, *, operation: type):
        cls.OPERATION_NAME = operation.OPERATION_NAME
        cls._operation_cls = operation

        operands, attrs, results, regions = partition_fields(operation._fields)

        for attr in attrs:
            setattr(
                cls,
                attr.name,
                property(lambda self, name=attr.name: self.attributes[name]),
            )

        for i, operand in enumerate(operands):
            if operation._ODS_OPERAND_SEGMENTS:

                def getter(self, i=i, operand=operand):
                    operand_range = segmented_accessor(
                        self.operands,
                        self.attributes["operandSegmentSizes"],
                        i,
                    )
                    return normalize_value_range(operand_range, operand.variadicity)

                setattr(cls, operand.name, property(getter))
            else:
                setattr(cls, operand.name, property(lambda self, i=i: self.operands[i]))


@dataclass
class ParamDef:
    name: str
    constraint: Any


class Type(ir.DynamicType):
    """
    Base class of Python-defined types.

    The following example shows two ways to define types via this class:
    ```python
    class MyType(MyDialect.Type, name=..):
      ...

    class MyType(Type, dialect=MyDialect, name=..):
      ...
    ```
    """

    @classmethod
    def __init_subclass__(
        cls,
        *,
        name: str | None = None,
        dialect: type | None = None,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)

        fields = []

        for base in cls.__bases__:
            if hasattr(base, "_fields"):
                fields.extend(base._fields)
        for key, value in cls.__annotations__.items():
            field = ParamDef(key, value)
            fields.append(field)

        cls._fields = fields

        if dialect:
            if hasattr(cls, "_dialect_obj"):
                raise RuntimeError(
                    f"This type has already been attached to dialect '{cls._dialect_obj.DIALECT_NAMESPACE}'."
                )
            cls._dialect_obj = dialect

        # for subclasses without "name" parameter,
        # just treat them as normal classes
        if not name:
            return

        if not hasattr(cls, "_dialect_obj"):
            raise RuntimeError(
                "Type subclasses must either inherit from a Dialect's Type subclass "
                "or provide the dialect as a class keyword argument."
            )

        cls._name = name
        cls._dialect_name = cls._dialect_obj.DIALECT_NAMESPACE
        cls.type_name = f"{cls._dialect_name}.{name}"

        for i, field in enumerate(cls._fields):
            setattr(
                cls,
                field.name,
                property(lambda self, i=i: self.params[i]),
            )

        cls._dialect_obj.types.append(cls)

    @classmethod
    def get(cls, *args, context=None):
        args = [
            ir.TypeAttr.get(arg, context) if isinstance(arg, ir.Type) else arg
            for arg in args
        ]
        return cls(ir.DynamicType.get(cls.type_name, args, context=context))

    @classmethod
    def _emit_type(cls) -> None:
        ctx = ConstraintLoweringContext()

        t = irdl.type_(cls._name)
        with ir.InsertionPoint(t.body):
            irdl.parameters(
                [ctx.lower(f.constraint) for f in cls._fields],
                [f.name for f in cls._fields],
            )


class Attribute(ir.DynamicAttr):
    """
    Base class of Python-defined attributes.

    The following example shows two ways to define attributes via this class:
    ```python
    class MyAttr(MyDialect.Attribute, name=..):
      ...

    class MyAttr(Attribute, dialect=MyDialect, name=..):
      ...
    ```
    """

    @classmethod
    def __init_subclass__(
        cls,
        *,
        name: str | None = None,
        dialect: type | None = None,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)

        fields = []

        for base in cls.__bases__:
            if hasattr(base, "_fields"):
                fields.extend(base._fields)
        for key, value in cls.__annotations__.items():
            field = ParamDef(key, value)
            fields.append(field)

        cls._fields = fields

        if dialect:
            if hasattr(cls, "_dialect_obj"):
                raise RuntimeError(
                    f"This attribute has already been attached to dialect '{cls._dialect_obj.DIALECT_NAMESPACE}'."
                )
            cls._dialect_obj = dialect

        # for subclasses without "name" parameter,
        # just treat them as normal classes
        if not name:
            return

        if not hasattr(cls, "_dialect_obj"):
            raise RuntimeError(
                "Attribute subclasses must either inherit from a Dialect's Attribute subclass "
                "or provide the dialect as a class keyword argument."
            )

        cls._name = name
        cls._dialect_name = cls._dialect_obj.DIALECT_NAMESPACE
        cls.attr_name = f"{cls._dialect_name}.{name}"

        for i, field in enumerate(cls._fields):
            setattr(
                cls,
                field.name,
                property(lambda self, i=i: self.params[i]),
            )

        cls._dialect_obj.attributes.append(cls)

    @classmethod
    def get(cls, *args, context=None):
        args = [
            ir.TypeAttr.get(arg, context) if isinstance(arg, ir.Type) else arg
            for arg in args
        ]
        return cls(ir.DynamicAttr.get(cls.attr_name, args, context=context))

    @classmethod
    def _emit_attr(cls) -> None:
        ctx = ConstraintLoweringContext()

        t = irdl.attribute(cls._name)
        with ir.InsertionPoint(t.body):
            irdl.parameters(
                [ctx.lower(f.constraint) for f in cls._fields],
                [f.name for f in cls._fields],
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
        cst: Result[i32] = infer_result()

    class AddOp(MyInt.Operation, name="add"):
        lhs: Operand[i32]
        rhs: Operand[i32]
        res: Result[i32] = infer_result()
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
            dict(),
            dialect=cls,
        )
        cls.types = []
        cls.Type = type(
            "Type",
            (Type,),
            dict(),
            dialect=cls,
        )
        cls.attributes = []
        cls.Attribute = type(
            "Attribute",
            (Attribute,),
            dict(),
            dialect=cls,
        )

    @classmethod
    def _emit_dialect(cls) -> None:
        d = irdl.dialect(cls.name)
        with ir.InsertionPoint(d.body):
            for type_ in cls.types:
                type_._emit_type()
            for attr in cls.attributes:
                attr._emit_attr()
            for op in cls.operations:
                op._emit_operation()

    @classmethod
    def _emit_module(cls) -> ir.Module:
        with ir.Location.unknown() if not ir.Location.current else nullcontext():
            m = ir.Module.create()
            with ir.InsertionPoint(m.body):
                cls._emit_dialect()

        return m

    @classmethod
    def load(
        cls,
        *,
        reload: bool = False,
    ) -> None:
        if hasattr(cls, "_mlir_module") and not reload:
            if cls._mlir_module.context is not ir.Context.current:
                raise RuntimeError(
                    "This dialect was loaded in a different context. "
                    "Please set reload=True to reload the dialect in the current context."
                )
            return

        cls._mlir_module = cls._emit_module()
        pm = PassManager()
        pm.add("canonicalize, cse")
        pm.run(cls._mlir_module.operation)

        irdl.load_dialects(cls._mlir_module)

        for op in cls.operations:
            op._attach_traits()

        _cext.globals._register_dialect_impl(cls.DIALECT_NAMESPACE, cls, replace=reload)

        for type_ in cls.types:
            typeid = ir.DynamicType.lookup_typeid(type_.type_name)
            _cext.register_type_caster(typeid, replace=reload)(type_)

        for attr in cls.attributes:
            typeid = ir.DynamicAttr.lookup_typeid(attr.attr_name)
            _cext.register_type_caster(typeid, replace=reload)(attr)

        for op in cls.operations:
            _cext.register_operation(cls, replace=reload)(op)
            _cext.register_op_adaptor(op, replace=reload)(op.Adaptor)
