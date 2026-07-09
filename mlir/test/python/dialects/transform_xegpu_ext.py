# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import transform
from mlir.dialects.transform import xegpu
from mlir.dialects.transform import structured, AnyValueType


def run(f):
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            print("\nTEST:", f.__name__)
            f()
        print(module)
    return f


@run
def getLoadOp():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("xegpu.dpas"),
    )
    with InsertionPoint(sequence.body):
        operand = transform.GetOperandOp(AnyValueType.get(), sequence.bodyTarget, [0])
        load_handle = xegpu.get_load_op(operand)
        transform.YieldOp()
    # CHECK-LABEL: TEST: getLoadOp
    # CHECK: transform.xegpu.get_load_op %


@run
def setAnchorLayout():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("xegpu.load_nd"),
    )
    with InsertionPoint(sequence.body):
        xegpu.set_anchor_layout(
            sequence.bodyTarget,
            sg_layout=[6, 4],
            sg_data=[32, 16],
            inst_data=[8, 16],
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: setAnchorLayout
    # CHECK: transform.xegpu.set_anchor_layout %
    # CHECK-NOT: index = 0
    # CHECK: sg_layout = [6, 4]
    # CHECK: sg_data = [32, 16]
    # CHECK: inst_data = [8, 16]


@run
def setAnchorLayoutDPAS():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("xegpu.dpas"),
    )
    with InsertionPoint(sequence.body):
        xegpu.set_anchor_layout(
            sequence.bodyTarget,
            index=1,
            sg_layout=[6, 4],
            sg_data=[32, 16],
            inst_data=[8, 16],
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: setAnchorLayoutDPAS
    # CHECK: transform.xegpu.set_anchor_layout %
    # CHECK: index = 1
    # CHECK: sg_layout = [6, 4]
    # CHECK: sg_data = [32, 16]
    # CHECK: inst_data = [8, 16]


@run
def setAnchorLayoutOrder():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("xegpu.load_nd"),
    )
    with InsertionPoint(sequence.body):
        xegpu.set_anchor_layout(
            sequence.bodyTarget,
            sg_layout=[6, 4],
            sg_data=[32, 16],
            inst_data=[8, 16],
            order=[1, 0],
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: setAnchorLayoutOrder
    # CHECK: transform.xegpu.set_anchor_layout %
    # CHECK-NOT: index = 0
    # CHECK: sg_layout = [6, 4]
    # CHECK: sg_data = [32, 16]
    # CHECK: inst_data = [8, 16]
    # CHECK: order = [1, 0]


@run
def setAnchorLayoutSlice():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("xegpu.load"),
    )
    with InsertionPoint(sequence.body):
        xegpu.set_anchor_layout(
            sequence.bodyTarget,
            sg_layout=[6, 4],
            sg_data=[32, 16],
            inst_data=[8, 16],
            slice_dims=[0],
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: setAnchorLayoutSlice
    # CHECK: transform.xegpu.set_anchor_layout %
    # CHECK-NOT: index = 0
    # CHECK: sg_layout = [6, 4]
    # CHECK: sg_data = [32, 16]
    # CHECK: inst_data = [8, 16]
    # CHECK: slice_dims = [0]


@run
def setGPULaunchThreadsOp():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("gpu.launch"),
    )
    with InsertionPoint(sequence.body):
        xegpu.set_gpu_launch_threads(sequence.bodyTarget, threads=[8, 4, 1])
        transform.YieldOp()
    # CHECK-LABEL: TEST: setGPULaunchThreadsOp
    # CHECK: transform.xegpu.set_gpu_launch_threads
    # CHECK: threads = [8, 4, 1]


@run
def insertPrefetch():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("xegpu.load_nd"),
    )
    with InsertionPoint(sequence.body):
        xegpu.insert_prefetch(sequence.bodyTarget)
        transform.YieldOp()
    # CHECK-LABEL: TEST: insertPrefetch
    # CHECK: transform.xegpu.insert_prefetch


@run
def insertPrefetchNbPrefetch():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("xegpu.load_nd"),
    )
    with InsertionPoint(sequence.body):
        xegpu.insert_prefetch(sequence.bodyTarget, nb_prefetch=2)
        transform.YieldOp()
    # CHECK-LABEL: TEST: insertPrefetchNbPrefetch
    # CHECK: transform.xegpu.insert_prefetch
    # CHECK-SAME: nb_prefetch = 2


@run
def insertPrefetchNbPrefetchParam():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("xegpu.load_nd"),
    )
    with InsertionPoint(sequence.body):
        int32_t = IntegerType.get_signless(32)
        param_int32_t = transform.ParamType.get(int32_t)
        nb_param = transform.ParamConstantOp(
            param_int32_t,
            IntegerAttr.get(int32_t, 2),
        )
        xegpu.insert_prefetch(sequence.bodyTarget, nb_prefetch=nb_param)
        transform.YieldOp()
    # CHECK-LABEL: TEST: insertPrefetchNbPrefetchParam
    # CHECK: %[[PARAM_OP:.*]] = transform.param.constant 2
    # CHECK: transform.xegpu.insert_prefetch
    # CHECK-SAME: nb_prefetch = %[[PARAM_OP]]


@run
def ConvertLayoutMinimal():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("xegpu.dpas"),
    )
    with InsertionPoint(sequence.body):
        operand = transform.GetOperandOp(AnyValueType.get(), sequence.bodyTarget, [0])
        xegpu.convert_layout(
            operand,
            input_sg_layout=[6, 4],
            input_sg_data=[32, 16],
            target_sg_layout=[6, 4],
            target_sg_data=[8, 16],
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: ConvertLayoutMinimal
    # CHECK: transform.xegpu.convert_layout %
    # CHECK: input_sg_layout = [6, 4]
    # CHECK: input_sg_data = [32, 16]
    # CHECK: target_sg_layout = [6, 4]
    # CHECK: target_sg_data = [8, 16]


@run
def ConvertLayout():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("xegpu.dpas"),
    )
    with InsertionPoint(sequence.body):
        operand = transform.GetOperandOp(AnyValueType.get(), sequence.bodyTarget, [1])
        xegpu.convert_layout(
            operand,
            input_sg_layout=[6, 4],
            input_sg_data=[32, 32],
            input_inst_data=[32, 16],
            input_order=[1, 0],
            target_sg_layout=[6, 4],
            target_sg_data=[32, 32],
            target_inst_data=[8, 16],
            target_order=[0, 1],
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: ConvertLayout
    # CHECK: transform.xegpu.convert_layout %
    # CHECK: input_sg_layout = [6, 4]
    # CHECK: input_sg_data = [32, 32]
    # CHECK: input_inst_data = [32, 16]
    # CHECK: input_order = [1, 0]
    # CHECK: target_sg_layout = [6, 4]
    # CHECK: target_sg_data = [32, 32]
    # CHECK: target_inst_data = [8, 16]
    # CHECK: target_order = [0, 1]
