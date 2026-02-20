# RUN: env PYTHONUNBUFFERED=1 %PYTHON %s 2>&1 | FileCheck %s

from typing import Sequence

from contextlib import contextmanager

from mlir import ir
from mlir.dialects import index, transform, func, arith, ext
from mlir.dialects.transform import (
    DiagnosedSilenceableFailure,
    AnyOpType,
    AnyValueType,
    AnyParamType,
    structured,
    interpreter,
)


@ext.register_dialect
class MyTransform(ext.Dialect, name="my_transform"):
    pass


def run(emit_schedule):
    print(f"Test: {emit_schedule.__name__}")
    with ir.Context() as ctx, ir.Location.unknown():
        payload = emit_payload()

        MyTransform.load(register=False, reload=True)

        GetNamedAttributeOp.attach_interface_impls(ctx)
        PrintParamOp.attach_interface_impls(ctx)

        # NB: Other newly defined my_transform ops have their interfaces attached
        #     in their respective test functions.
        schedule = emit_schedule()

        interpreter.apply_named_sequence(
            payload,
            _named_seq := schedule.operation.regions[0].blocks[0].operations[0],
            schedule,
        )


# Payload used by all tests
def emit_payload():
    payload_module = ir.Module.create()
    with ir.InsertionPoint(payload_module.body):
        f32 = ir.F32Type.get()

        @func.FuncOp.from_py_func(f32, f32, results=[f32])
        def name_of_func(a, b):
            c = arith.addf(a, b)
            i32 = ir.IntegerType.get_signless(32)
            arith.constant(i32, 42)
            arith.constant(i32, 24)
            func.ReturnOp([c])

    return payload_module


@contextmanager
def schedule_boilerplate():
    schedule = ir.Module.create()
    schedule.operation.attributes["transform.with_named_sequence"] = ir.UnitAttr.get()
    with ir.InsertionPoint(schedule.body):
        named_sequence = transform.NamedSequenceOp(
            "__transform_main",
            [AnyOpType.get()],
            [AnyOpType.get()],
            arg_attrs=[{"transform.consumed": ir.UnitAttr.get()}],
        )
        with ir.InsertionPoint(named_sequence.body):
            yield schedule, named_sequence


# MemoryEffectsOpInterface implementation for TransformOpInterface-implementing ops.
# Used by most ops defined below.
class MemoryEffectsOpInterfaceFallbackModel(ir.MemoryEffectsOpInterface):
    @staticmethod
    def get_effects(op: ir.Operation, effects):
        transform.only_reads_handle(op.op_operands, effects)
        transform.produces_handle(op.results, effects)
        transform.only_reads_payload(effects)


# Demonstration of a TransformOpInterface-implementing op that gets named attributes
# from target ops and produces them as param handles.
@ext.register_operation(MyTransform)
class GetNamedAttributeOp(MyTransform.Operation, name="get_named_attribute"):
    target: ext.Operand[transform.AnyOpType]
    attr_name: ir.StringAttr
    attr_as_param: ext.Result[transform.AnyParamType[()]]

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceFallbackModel.attach(cls.OPERATION_NAME, context=ctx)
        MemoryEffectsOpInterfaceFallbackModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceFallbackModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "GetNamedAttributeOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            target_ops = state.get_payload_ops(op.target)
            associated_attrs = []
            for target_op in target_ops:
                assoc_attr = target_op.attributes.get(op.attr_name.value)
                if assoc_attr is None:
                    return DiagnosedSilenceableFailure.RecoverableFailure
                associated_attrs.append(assoc_attr)
            results.set_params(op.attr_as_param, associated_attrs)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "GetNamedAttributeOp") -> bool:
            return False


@ext.register_operation(MyTransform)
class PrintParamOp(MyTransform.Operation, name="print_param"):
    target: ext.Operand[transform.AnyParamType]
    name: ir.StringAttr

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceFallbackModel.attach(cls.OPERATION_NAME, context=ctx)
        MemoryEffectsOpInterfaceFallbackModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceFallbackModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "PrintParamOp",
            rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            target_attrs = state.get_params(op.target)
            print(f"[[[ IR printer: {op.name.value} ]]]")
            for attr in target_attrs:
                print(attr)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "GetNamedAttributeOp") -> bool:
            return False


# Syntax for an op with one op handle operand and one op handle result.
@ext.register_operation(MyTransform)
class OneOpInOneOpOut(MyTransform.Operation, name="one_op_in_one_op_out"):
    target: ext.Operand[transform.AnyOpType]
    res: ext.Result[transform.AnyOpType[()]]


# CHECK-LABEL: Test: OneOpInOneOpOutTransformOpInterface
@run
def OneOpInOneOpOutTransformOpInterface():
    """Tests a simple passthrough interface implementation.

    Checks that the target ops are correctly identified and passed as results.
    """

    # Define a simple passthrough implementation of the TransformOpInterface for OneOpInOneOpOut.
    class TransformOpInterfaceFallbackModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: OneOpInOneOpOut,
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            target_ops = state.get_payload_ops(op.target)
            target_names = [t.name.value for t in target_ops]
            print(f"OneOpInOneOpOutTransformOpInterface: target_names={target_names}")
            results.set_ops(op.res, target_ops)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: OneOpInOneOpOut) -> bool:
            return False

    # Attach the interface implementation to the op.
    TransformOpInterfaceFallbackModel.attach(OneOpInOneOpOut.OPERATION_NAME)

    # TransformOpInterface-implementing ops are also required to implement MemoryEffectsOpInterface. The above defined fallback model works for this op.
    MemoryEffectsOpInterfaceFallbackModel.attach(OneOpInOneOpOut.OPERATION_NAME)

    with schedule_boilerplate() as (schedule, named_seq):
        func_handle = structured.MatchOp.match_op_names(
            named_seq.bodyTarget, ["func.func"]
        ).result
        # CHECK: OneOpInOneOpOutTransformOpInterface: target_names=['name_of_func']
        out = OneOpInOneOpOut(func_handle).result
        # CHECK: Output handle from OneOpInOneOpOut
        # CHECK-NEXT: func.func @name_of_func
        transform.PrintOp(target=out, name="Output handle from OneOpInOneOpOut")
        transform.YieldOp([out])

    return schedule


# CHECK-LABEL: Test: OneOpInOneOpOutTransformOpInterfaceRewriterImpl
@run
def OneOpInOneOpOutTransformOpInterfaceRewriterImpl():
    """Tests an interface implementation using the rewriter to modify the IR.

    Checks that `arith.constant` ops are replaced by `index.constant` ops and
    that the results are correctly updated.
    """

    class TransformOpInterfaceFallbackModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: OneOpInOneOpOut,
            rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            result_ops = []
            for target_op in state.get_payload_ops(op.target):
                with ir.InsertionPoint(target_op):
                    index_version = index.constant(target_op.value.value)
                result_ops.append(index_version.owner)
                rewriter.replace_op(target_op, [index_version])
            results.set_ops(op.res, result_ops)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: OneOpInOneOpOut) -> bool:
            return False

    # Attach the interface implementation to the op.
    TransformOpInterfaceFallbackModel.attach(OneOpInOneOpOut.OPERATION_NAME)

    # TransformOpInterface-implementing ops are also required to implement MemoryEffectsOpInterface. The above defined fallback model works for this op.
    class MemoryEffectsOpInterfaceFallbackModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation, effects):
            transform.consumes_handle(op.op_operands, effects)
            transform.produces_handle(op.results, effects)
            transform.modifies_payload(effects)

    MemoryEffectsOpInterfaceFallbackModel.attach(OneOpInOneOpOut.OPERATION_NAME)

    with schedule_boilerplate() as (schedule, named_seq):
        func_handle = structured.MatchOp.match_op_names(
            named_seq.bodyTarget, ["func.func"]
        ).result
        csts_handle = structured.MatchOp.match_op_names(
            named_seq.bodyTarget, ["arith.constant"]
        ).result
        # CHECK: Before replacement:
        # CHECK-NOT: index.constant
        # CHECK-DAG: arith.constant 42 : i32
        # CHECK-DAG: arith.constant 24 : i32
        transform.PrintOp(target=func_handle, name="Before replacement:")
        out = OneOpInOneOpOut(csts_handle).result
        # CHECK: After replacement:
        # CHECK-NOT: arith.constant
        # CHECK-DAG: index.constant 42
        # CHECK-DAG: index.constant 24
        transform.PrintOp(target=func_handle, name="After replacement:")
        # CHECK: Output handle from OneOpInOneOpOut:
        # CHECK-NEXT: index.constant 42
        # CHECK-NEXT: index.constant 24
        transform.PrintOp(target=out, name="Output handle from OneOpInOneOpOut:")
        transform.YieldOp([out])

    return schedule


@ext.register_operation(MyTransform)
class OpValParamInParamOpValOut(
    MyTransform.Operation, name="op_val_param_in_param_op_val_out"
):
    # operands
    op_arg: ext.Operand[transform.AnyOpType]
    val_arg: ext.Operand[transform.AnyValueType]
    param_arg: ext.Operand[transform.AnyParamType]
    # results
    param_res: ext.Result[transform.AnyParamType[()]]
    op_res: ext.Result[transform.AnyOpType[()]]
    value_res: ext.Result[transform.AnyValueType[()]]


# CHECK-LABEL: Test: OpValParamInParamOpValOutTransformOpInterface
@run
def OpValParamInParamOpValOutTransformOpInterface():
    """Tests an interface implementation involving Op, Value, and Param types.

    Checks that payload ops, values, and parameters are correctly permuted and
    propagated and accessible from the (permuted) result handles.
    """

    class TransformOpInterfaceFallbackModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: OpValParamInParamOpValOut,
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            ops = state.get_payload_ops(op.op_arg)
            values = state.get_payload_values(op.val_arg)
            params = state.get_params(op.param_arg)
            print(
                f"OpValParamInParamOpValOutTransformOpInterface: ops={len(ops)}, values={len(values)}, params={len(params)}"
            )
            results.set_params(op.param_res, params)
            results.set_ops(op.op_res, ops)
            results.set_values(op.value_res, values)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: OpValParamInParamOpValOut) -> bool:
            return False

    TransformOpInterfaceFallbackModel.attach(OpValParamInParamOpValOut.OPERATION_NAME)

    # TransformOpInterface-implementing ops are also required to implement MemoryEffectsOpInterface. The above defined fallback model works for this op.
    MemoryEffectsOpInterfaceFallbackModel.attach(
        OpValParamInParamOpValOut.OPERATION_NAME
    )

    with schedule_boilerplate() as (schedule, named_seq):
        func_handle = structured.MatchOp.match_op_names(
            named_seq.bodyTarget, ["func.func"]
        ).result
        addf_handle = structured.MatchOp.match_op_names(
            named_seq.bodyTarget, ["arith.addf"]
        ).result
        func_and_addf = transform.MergeHandlesOp([func_handle, addf_handle])
        value_handle = transform.GetResultOp(
            AnyValueType.get(), addf_handle, [0]
        ).result
        param_handle = transform.ParamConstantOp(
            AnyParamType.get(), ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 42)
        ).param

        # CHECK: OpValParamInParamOpValOutTransformOpInterface: ops=2, values=1, params=1
        op_val_param_op = OpValParamInParamOpValOut(
            func_and_addf, value_handle, param_handle
        )
        # CHECK: Ops passed through OpValParamInParamOpValOut:
        # CHECK-NEXT: func.func
        # CHECK: arith.addf
        transform.PrintOp(
            target=op_val_param_op.op_res,
            name="Ops passed through OpValParamInParamOpValOut:",
        )

        # CHECK: Ops defining values passed through OpValParamInParamOpValOut:
        # CHECK-NEXT: arith.addf
        addf_as_res = transform.GetDefiningOp(
            transform.AnyOpType.get(), op_val_param_op.value_res
        ).result
        transform.PrintOp(
            target=addf_as_res,
            name="Ops defining values passed through OpValParamInParamOpValOut:",
        )

        # CHECK: Parameter passed through OpValParamInParamOpValOut:
        # CHECK-NEXT: 42 : i32
        PrintParamOp(
            op_val_param_op.param_res,
            name=ir.StringAttr.get(
                "Parameter passed through OpValParamInParamOpValOut:"
            ),
        )

        transform.YieldOp([op_val_param_op.op_res])
        named_seq.verify()

    return schedule


@ext.register_operation(MyTransform)
class OpsParamsInValuesParamOut(
    MyTransform.Operation, name="ops_params_in_values_param_out"
):
    # operands
    ops: Sequence[ext.Operand[transform.AnyOpType]]
    params: Sequence[ext.Operand[transform.AnyParamType]]
    # results
    values: Sequence[ext.Result[transform.AnyValueType]]
    param: ext.Result[transform.AnyParamType]


# CHECK-LABEL: Test: OpsParamsInValuesParamOutTransformOpInterface
@run
def OpsParamsInValuesParamOutTransformOpInterface():
    """Tests an interface with variadic Op and Param operands and variadic Value results.

    Checks correct handling of multiple handles, parameter aggregation, and
    result generation.
    """

    class TransformOpInterfaceFallbackModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: OpsParamsInValuesParamOut,
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            ops_count = 0
            value_handles = []
            for op_handle in op.ops:
                ops = state.get_payload_ops(op_handle)
                ops_count += len(ops)
                value_handles.append([i for op in ops for i in op.results])

            param_count = 0
            param_sum = 0
            for param_handle in op.params:
                params = state.get_params(param_handle)
                param_count += len(params)
                param_sum += sum(p.value for p in params)

            print(
                f"OpsParamsInValuesParamOutTransformOpInterfaceFallbackModel: op_count={ops_count}, param_count={param_count}"
            )

            assert len(op.values) == len(op.ops)
            for value_res_handle, value_vector in zip(op.values, value_handles):
                results.set_values(value_res_handle, value_vector)
            results.set_params(
                op.param,
                [ir.IntegerAttr.get(ir.IntegerType.get_signless(32), param_sum)],
            )
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: OpsParamsInValuesParamOut) -> bool:
            return False

    TransformOpInterfaceFallbackModel.attach(OpsParamsInValuesParamOut.OPERATION_NAME)

    MemoryEffectsOpInterfaceFallbackModel.attach(
        OpsParamsInValuesParamOut.OPERATION_NAME
    )

    with schedule_boilerplate() as (schedule, named_seq):
        func_handle = structured.MatchOp.match_op_names(
            named_seq.bodyTarget, ["func.func"]
        ).result
        csts_handle = structured.MatchOp.match_op_names(
            named_seq.bodyTarget, ["arith.constant"]
        ).result
        csts_as_param = GetNamedAttributeOp(
            csts_handle, attr_name=ir.StringAttr.get("value")
        ).attr_as_param

        param_handle = transform.ParamConstantOp(
            AnyParamType.get(), ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 123)
        ).param

        # CHECK: OpsParamsInValuesParamOutTransformOpInterfaceFallbackModel: op_count=3, param_count=3
        op = OpsParamsInValuesParamOut(
            [transform.AnyValueType.get()] * 2,
            transform.AnyParamType.get(),
            [func_handle, csts_handle],
            [csts_as_param, param_handle],
        )

        empty_handle = transform.GetDefiningOp(transform.AnyOpType.get(), op.values[0])
        # CHECK: Defining op of value result 0
        transform.PrintOp(
            target=empty_handle.result, name="Defining op of value result 0"
        )
        # NB: no result on the func.func, so output is expected to be empty
        cst1_res, cst2_res = transform.SplitHandleOp(
            [transform.AnyValueType.get()] * 2, op.values[1]
        ).results

        cst1_again = transform.GetDefiningOp(transform.AnyOpType.get(), cst1_res)
        # CHECK-NEXT: Defining op of first constant
        # CHECK-NEXT: arith.constant 42 : i32
        transform.PrintOp(
            target=cst1_again.result, name="Defining op of first constant"
        )
        cst2_again = transform.GetDefiningOp(transform.AnyOpType.get(), cst2_res)
        # CHECK-NEXT: Defining op of second constant
        # CHECK-NEXT: arith.constant 24 : i32
        transform.PrintOp(
            target=cst2_again.result, name="Defining op of second constant"
        )

        # CHECK: Sum of params:
        # CHECK-NEXT: 189 : i32
        PrintParamOp(op.param, name=ir.StringAttr.get("Sum of params:"))

        transform.YieldOp([func_handle])
        named_seq.verify()

    return schedule
