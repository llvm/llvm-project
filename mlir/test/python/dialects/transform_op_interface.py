# RUN: %PYTHON %s | FileCheck %s

from contextlib import contextmanager

from mlir import ir
from mlir.ir import TypeAttr, F32Type, UnitAttr
from mlir.dialects import transform, irdl, func
from mlir.dialects.transform import AnyOpType, AnyValueType, AnyParamType, structured, interpreter

from mlir.dialects._ods_common import get_op_result_or_value, get_op_results_or_values


def run(emit_schedule):
    with ir.Context(), ir.Location.unknown():
        irdl_module = ir.Module.create()
        with ir.InsertionPoint(irdl_module.body):
            my_transform = irdl.dialect("my_transform")
            with ir.InsertionPoint(my_transform.body):
                OneOpInOneOpOut.emit_irdl()
                OpValParamInParamOpValOut.emit_irdl()
                OpsParamInParamsOut.emit_irdl()
        irdl_module.operation.verify()

        irdl.load_dialects(irdl_module)

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

        @func.FuncOp.from_py_func(
            F32Type.get(), F32Type.get(), results=[F32Type.get()]
        )
        def name_of_func(a, b):
            func.ReturnOp([b])

    return payload_module


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


class OpsParamInParamsOut:
    name = "ops_param_in_params_out"

    def __init__(
        self,
        ops: list[AnyOpType],
        param: AnyParamType,
    ):
        self.op = ir.Operation.create(
            "my_transform." + self.name,
            [AnyParamType.get()],
            [get_op_results_or_values(ops), get_op_result_or_value(param)],
        )

    @property
    def param_results(self):
        return self.op.results

    @classmethod
    def emit_irdl(cls):
        op = irdl.operation_(cls.name)
        with ir.InsertionPoint(op.body):
            op_handle_type = irdl.is_(TypeAttr.get(AnyOpType.get()))
            param_handle_type = irdl.is_(TypeAttr.get(AnyParamType.get()))
            irdl.operands_(
                [op_handle_type, param_handle_type],
                ["op_args", "param_arg"],
                [irdl.Variadicity.variadic, irdl.Variadicity.single],
            )
            irdl.results_(
                [param_handle_type], ["param_results"], [irdl.Variadicity.variadic]
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
