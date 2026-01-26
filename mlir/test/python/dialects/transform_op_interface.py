# RUN: %PYTHON %s | FileCheck %s

from typing import Sequence

from contextlib import contextmanager

from mlir import ir
from mlir.ir import TypeAttr, F32Type, UnitAttr
from mlir.dialects import transform, irdl, func, arith
from mlir.dialects.transform import (
    debug as transform_debug,
    AnyOpType,
    AnyValueType,
    AnyParamType,
    structured,
    interpreter,
)

from mlir.dialects._ods_common import get_op_result_or_value, get_op_results_or_values


def run(emit_schedule):
    with ir.Context(), ir.Location.unknown():
        irdl_module = ir.Module.create()
        with ir.InsertionPoint(irdl_module.body):
            my_transform = irdl.dialect("my_transform")
            with ir.InsertionPoint(my_transform.body):
                GetNamedAttributeOp.emit_irdl()
                OneOpInOneOpOut.emit_irdl()
                OpValParamInParamOpValOut.emit_irdl()
                OpsParamsInValuesParamOut.emit_irdl()
        irdl_module.operation.verify()

        print(irdl_module)
        irdl.load_dialects(irdl_module)

        GetNamedAttributeTransformOpInterfaceFallbackModel.attach(
            "my_transform." + GetNamedAttributeOp.name
        )
        GetNamedAttributeMemoryEffectsOpInterfaceFallbackModel.attach(
            "my_transform." + GetNamedAttributeOp.name
        )

        payload = emit_payload()
        schedule = emit_schedule()

        print("payload:", payload)
        print("schedule:", schedule)
        named_seq = schedule.operation.regions[0].blocks[0].operations[0]

        interpreter.apply_named_sequence(
            payload,
            named_seq,
            schedule,
        )

    del payload
    del schedule


# Payload used by all tests
def emit_payload():
    payload_module = ir.Module.create()
    with ir.InsertionPoint(payload_module.body):

        @func.FuncOp.from_py_func(F32Type.get(), F32Type.get(), results=[F32Type.get()])
        def name_of_func(a, b):
            c = arith.AddFOp(a, b)
            d = arith.constant(F32Type.get(), 42.0)
            e = arith.constant(F32Type.get(), 24.0)
            func.ReturnOp([c.results[0]])

    return payload_module


class GetNamedAttributeOp:
    name = "get_named_attribute"

    def __init__(self, target: AnyOpType, attr_name: str):
        self.op = ir.Operation.create(
            "my_transform.get_named_attribute",
            [AnyParamType.get()],
            [get_op_result_or_value(target)],
            {"attr_name": ir.StringAttr.get(attr_name)},
        )

    @property
    def attr_as_param(self) -> ir.Value:
        return self.op.results[0]

    @classmethod
    def emit_irdl(cls):
        op = irdl.operation_(cls.name)
        with ir.InsertionPoint(op.body):
            op_handle_type = irdl.is_(TypeAttr.get(AnyOpType.get()))
            param_handle_type = irdl.is_(TypeAttr.get(AnyParamType.get()))
            name_handle_kind = irdl.base(base_name="#builtin.string")
            irdl.operands_(
                [op_handle_type],
                ["target"],
                [irdl.Variadicity.single],
            )
            irdl.attributes_([name_handle_kind], ["attr_name"])
            irdl.results_(
                [param_handle_type], ["attr_as_param"], [irdl.Variadicity.single]
            )
        return op


class GetNamedAttributeTransformOpInterfaceFallbackModel(
    transform.TransformOpInterface
):
    @staticmethod
    def apply(
        op_: ir.Operation,
        rewriter: transform.TransformRewriter,
        results: transform.TransformResults,
        state: transform.TransformState,
    ):
        targets = state.get_payload_ops(target := op_.operands[0])
        associated_attrs = []
        for target_op in targets:
            assoc_attr = target_op.attributes.get(op_.attributes["attr_name"].value)
            if assoc_attr is None:
                return transform.DiagnosedSilenceableFailure.RecoverableFailure
            associated_attrs.append(assoc_attr)
        results.set_params(op_.results[0], associated_attrs)
        return transform.DiagnosedSilenceableFailure.Success

    @staticmethod
    def allow_repeated_handle_operands(op_: ir.Operation) -> bool:
        return False


class GetNamedAttributeMemoryEffectsOpInterfaceFallbackModel(
    ir.MemoryEffectsOpInterface
):
    @staticmethod
    def get_effects(op_: ir.Operation, effects):
        transform.only_reads_handle(list(op_.op_operands), effects)
        transform.produces_handle(list(op_.results), effects)


class OneOpInOneOpOut:
    name = "one_op_in_one_op_out"

    def __init__(self, op_arg: AnyOpType):
        self.op = ir.Operation.create(
            "my_transform.one_op_in_one_op_out",
            [AnyOpType.get()],
            [get_op_result_or_value(op_arg)],
        )

    @property
    def result(self):
        return self.op.results[0]

    @classmethod
    def emit_irdl(cls):
        op = irdl.operation_(cls.name)
        with ir.InsertionPoint(op.body):
            op_handle_type = irdl.is_(TypeAttr.get(AnyOpType.get()))
            irdl.operands_(
                [op_handle_type],
                ["arg"],
                [irdl.Variadicity.single],
            )
            irdl.results_([op_handle_type], ["result"], [irdl.Variadicity.single])
        return op


class OpValParamInParamOpValOut:
    name = "op_val_param_in_param_op_val_out"

    def __init__(
        self,
        op_arg: AnyOpType,
        val: AnyValueType,
        param: AnyParamType,
    ):
        self.op = ir.Operation.create(
            "my_transform." + self.name,
            [
                AnyParamType.get(),
                AnyOpType.get(),
                AnyValueType.get(),
            ],
            [
                get_op_result_or_value(op_arg),
                get_op_result_or_value(val),
                get_op_result_or_value(param),
            ],
        )

    @property
    def param_res(self):
        return self.op.results[0]

    @property
    def op_res(self):
        return self.op.results[1]

    @property
    def value_res(self):
        return self.op.results[2]

    @classmethod
    def emit_irdl(cls):
        op = irdl.operation_(cls.name)
        with ir.InsertionPoint(op.body):
            op_handle_type = irdl.is_(TypeAttr.get(AnyOpType.get()))
            value_handle_type = irdl.is_(TypeAttr.get(AnyValueType.get()))
            param_handle_type = irdl.is_(TypeAttr.get(AnyParamType.get()))
            irdl.operands_(
                [op_handle_type, value_handle_type, param_handle_type],
                ["op_arg", "value_arg", "param_arg"],
                [
                    irdl.Variadicity.single,
                    irdl.Variadicity.single,
                    irdl.Variadicity.single,
                ],
            )
            irdl.results_(
                [param_handle_type, op_handle_type, value_handle_type],
                ["param_res", "op_res", "value_res"],
                [
                    irdl.Variadicity.single,
                    irdl.Variadicity.single,
                    irdl.Variadicity.single,
                ],
            )
        return op


class OpsParamsInValuesParamOut:
    name = "ops_params_in_values_param_out"

    def __init__(
        self,
        value_results: Sequence[AnyValueType],
        ops: Sequence[AnyOpType],
        params: Sequence[AnyParamType],
    ):
        def as_i32(x):
            return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), x)

        self.op = ir.Operation.create(
            "my_transform." + self.name,
            list(value_results) + [AnyParamType.get()],
            list(get_op_results_or_values(ops))
            + list(get_op_results_or_values(params)),
            {
                "operandSegmentSizes": ir.DenseI32ArrayAttr.get(
                    [len(ops), len(params)]
                ),
                "resultSegmentSizes": ir.DenseI32ArrayAttr.get([len(value_results)]),
            },
        )

    @property
    def param(self):
        return self.op.results[-1]

    @property
    def values(self):
        return self.op.results[:-1]

    @classmethod
    def emit_irdl(cls):
        op = irdl.operation_(cls.name)
        with ir.InsertionPoint(op.body):
            op_handle_type = irdl.is_(TypeAttr.get(AnyOpType.get()))
            value_handle_type = irdl.is_(TypeAttr.get(AnyValueType.get()))
            param_handle_type = irdl.is_(TypeAttr.get(AnyParamType.get()))
            irdl.operands_(
                [op_handle_type, param_handle_type],
                ["ops", "params"],
                [irdl.Variadicity.variadic, irdl.Variadicity.variadic],
            )
            irdl.results_(
                [value_handle_type, param_handle_type],
                ["value_results", "param"],
                [irdl.Variadicity.variadic, irdl.Variadicity.single],
            )
        return op


@contextmanager
def schedule_boilerplate():
    schedule = ir.Module.create()
    schedule.operation.attributes["transform.with_named_sequence"] = ir.UnitAttr.get()
    with ir.InsertionPoint(schedule.body):
        named_sequence = transform.NamedSequenceOp(
            "__transform_main",
            [AnyOpType.get()],
            [AnyOpType.get()],
            arg_attrs=[{"transform.consumed": UnitAttr.get()}],
        )
        with ir.InsertionPoint(named_sequence.body):
            yield schedule, named_sequence


@run
def OneOpInOneOpOutTransformOpInterface():
    class OneOpInOneOpOutTransformOpInterfaceFallbackModel(
        transform.TransformOpInterface
    ):
        @staticmethod
        def apply(
            op_: ir.Operation,
            rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ):
            targets = state.get_payload_ops(arg := op_.operands[0])
            target_names = [t.opview.name.value for t in targets]
            print(
                f"OneOpInOneOpOutTransformOpInterfaceFallbackModel: target_names={target_names}"
            )
            results.set_ops(result := op_.results[0], targets)
            return transform.DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(op_: ir.Operation) -> bool:
            return False

    OneOpInOneOpOutTransformOpInterfaceFallbackModel.attach(
        "my_transform.one_op_in_one_op_out", ir.Context.current
    )

    # TransformOpInterface-implementing ops are also required to implement MemoryEffectsOpInterface.
    class MemoryEffectsOpInterfaceFallbackModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op_: ir.Operation, effects):
            transform.only_reads_handle(list(op_.op_operands), effects)
            transform.produces_handle(list(op_.results), effects)

    MemoryEffectsOpInterfaceFallbackModel.attach(
        "my_transform.one_op_in_one_op_out", ir.Context.current
    )

    with schedule_boilerplate() as (schedule, named_seq):
        print(f"{named_seq=}")
        func_handle = structured.MatchOp.match_op_names(
            named_seq.bodyTarget, ["func.func"]
        ).result
        func_handle.dump()
        # CHECK: OneOpInOneOpOutTransformOpInterfaceFallbackModel: target_names=['name_of_func']
        out = OneOpInOneOpOut(func_handle).result
        out.dump()
        print(out.owner)
        transform.YieldOp([out])
        named_seq.verify()
        print("named_seq", named_seq)
        print("named_seq.parent", named_seq.parent)

    return schedule


@run
def OpValParamInParamOpValOutTransformOpInterface():
    class OpValParamInParamOpValOutTransformOpInterfaceFallbackModel(
        transform.TransformOpInterface
    ):
        @staticmethod
        def apply(
            op_: ir.Operation,
            rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ):
            ops = state.get_payload_ops(op_.operands[0])
            values = state.get_payload_values(op_.operands[1])
            params = state.get_params(op_.operands[2])
            print(
                f"OpValParamInParamOpValOutTransformOpInterfaceFallbackModel: ops={len(ops)}, values={len(values)}, params={len(params)}"
            )
            results.set_params(op_.results[0], params)
            results.set_ops(op_.results[1], ops)
            results.set_values(op_.results[2], values)
            return transform.DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(op_: ir.Operation) -> bool:
            return False

    OpValParamInParamOpValOutTransformOpInterfaceFallbackModel.attach(
        "my_transform.op_val_param_in_param_op_val_out", ir.Context.current
    )

    # TransformOpInterface-implementing ops are also required to implement MemoryEffectsOpInterface.
    class MemoryEffectsOpInterfaceFallbackModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op_: ir.Operation, effects):
            transform.only_reads_handle(list(op_.op_operands), effects)
            transform.produces_handle(list(op_.results), effects)

    MemoryEffectsOpInterfaceFallbackModel.attach(
        "my_transform.op_val_param_in_param_op_val_out", ir.Context.current
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

        # CHECK: OpValParamInParamOpValOutTransformOpInterfaceFallbackModel: ops=2, values=1, params=1
        op_val_param_op = OpValParamInParamOpValOut(
            func_and_addf, value_handle, param_handle
        )

        transform.YieldOp([op_val_param_op.op_res])
        named_seq.verify()

    return schedule


@run
def OpsParamsInValuesParamOutTransformOpInterface():
    class OpsParamsInValuesParamOutTransformOpInterfaceFallbackModel(
        transform.TransformOpInterface
    ):
        @staticmethod
        def apply(
            op_: ir.Operation,
            rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ):
            # The last operand is the param. All previous ones are ops.
            op_handles, param_handles = [], []
            for operand in op_.operands:
                if isinstance(operand.type, transform.AnyOpType):
                    op_handles.append(operand)
                else:
                    param_handles.append(operand)

            ops_count = 0
            value_handles = []
            for op_handle in op_handles:
                ops = state.get_payload_ops(op_handle)
                ops_count += len(ops)
                value_handles.append(list(op.results[:1] for op in ops))

            param_count = 0
            param_sum = 0
            for param_handle in param_handles:
                params = state.get_params(param_handle)
                param_count += len(params)
                param_sum += sum(p.value for p in params)

            print(
                f"OpsParamInValuesParamOutTransformOpInterfaceFallbackModel: #op_handles={len(op_handles)}, ops_count={ops_count}, #param_handles={len(param_handles)}, param_count={param_count}"
            )

            assert len(op_.results) + 1 == len(op_handles)
            for i in range(len(op_.results) - 1):
                results.set_values(
                    op_.results[i],
                    value_handles[i],
                )
            results.set_params(
                op_.results[-1],
                [ir.IntegerAttr.get(ir.IntegerType.get_signless(32), param_sum)],
            )
            return transform.DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(op_: ir.Operation) -> bool:
            return True

    OpsParamsInValuesParamOutTransformOpInterfaceFallbackModel.attach(
        "my_transform." + OpsParamsInValuesParamOut.name
    )

    class OpsParamsInParamsOutMemoryEffectsOpInterfaceFallbackModel(
        ir.MemoryEffectsOpInterface
    ):
        @staticmethod
        def get_effects(op_: ir.Operation, effects):
            transform.only_reads_handle(list(op_.op_operands), effects)
            transform.produces_handle(list(op_.results), effects)

    OpsParamsInParamsOutMemoryEffectsOpInterfaceFallbackModel.attach(
        "my_transform." + OpsParamsInValuesParamOut.name
    )

    with schedule_boilerplate() as (schedule, named_seq):
        func_handle = structured.MatchOp.match_op_names(
            named_seq.bodyTarget, ["func.func"]
        ).result
        csts_handle = structured.MatchOp.match_op_names(
            named_seq.bodyTarget, ["arith.constant"]
        ).result
        csts_as_param = GetNamedAttributeOp(csts_handle, "value").attr_as_param

        param_handle = transform.ParamConstantOp(
            AnyParamType.get(), ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 123)
        ).param

        # CHECK: OpsParamInParamsOutTransformOpInterfaceFallbackModel: op_count=2, param_count=1
        op = OpsParamsInValuesParamOut(
            [transform.AnyValueType.get()] * 2 + [transform.AnyParamType.get()],
            [func_handle, csts_handle],
            [csts_as_param, param_handle],
        )
        print(op.op)
        # CHECK: Sum of params: 189
        transform_debug.EmitParamAsRemarkOp(op.param, message="Sum of params")

        transform_debug.EmitRemarkAtOp(op.values[0], message="Value results 0")
        transform_debug.EmitRemarkAtOp(op.values[1], message="Value results 1")

        transform.YieldOp([func_handle])
        named_seq.verify()

    return schedule
