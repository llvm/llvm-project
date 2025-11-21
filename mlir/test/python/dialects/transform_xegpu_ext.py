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
def getDescOpDefaultIndex():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("xegpu.dpas"),
    )
    with InsertionPoint(sequence.body):
        operand = transform.GetOperandOp(AnyValueType.get(), sequence.bodyTarget, [0])
        desc_handle = xegpu.get_desc_op(operand)
        transform.YieldOp()
    # CHECK-LABEL: TEST: getDescOpDefaultIndex
    # CHECK: transform.xegpu.get_desc_op %


@run
def setDescLayoutMinimal():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("xegpu.create_nd_tdesc"),
    )
    with InsertionPoint(sequence.body):
        xegpu.set_desc_layout(sequence.bodyTarget, sg_layout=[6, 4], sg_data=[32, 16])
        transform.YieldOp()
    # CHECK-LABEL: TEST: setDescLayoutMinimal
    # CHECK: %0 = transform.xegpu.set_desc_layout %
    # CHECK: sg_layout = [6, 4]
    # CHECK: sg_data = [32, 16]


@run
def setDescLayoutInstData():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("xegpu.create_nd_tdesc"),
    )
    with InsertionPoint(sequence.body):
        xegpu.set_desc_layout(
            sequence.bodyTarget, sg_layout=[6, 4], sg_data=[32, 16], inst_data=[8, 16]
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: setDescLayoutInstData
    # CHECK: %0 = transform.xegpu.set_desc_layout %
    # CHECK: sg_layout = [6, 4]
    # CHECK: sg_data = [32, 16]
    # CHECK: inst_data = [8, 16]


@run
def setOpLayoutAttrOperandMinimal():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("xegpu.dpas"),
    )
    with InsertionPoint(sequence.body):
        xegpu.set_op_layout_attr(
            sequence.bodyTarget,
            sg_layout=[6, 4],
            sg_data=[32, 16],
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: setOpLayoutAttr
    # CHECK: transform.xegpu.set_op_layout_attr %
    # NO-CHECK: index = 0
    # NO-CHECK: result
    # CHECK: sg_layout = [6, 4]
    # CHECK: sg_data = [32, 16]
    # NO-CHECK: inst_data


@run
def setOpLayoutAttrResult():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("xegpu.dpas"),
    )
    with InsertionPoint(sequence.body):
        xegpu.set_op_layout_attr(
            sequence.bodyTarget,
            index=0,
            sg_layout=[6, 4],
            sg_data=[32, 16],
            inst_data=[8, 16],
            result=True,
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: setOpLayoutAttr
    # CHECK: transform.xegpu.set_op_layout_attr %
    # NO-CHECK: index = 0
    # CHECK: result
    # CHECK: sg_layout = [6, 4]
    # CHECK: sg_data = [32, 16]
    # CHECK: inst_data = [8, 16]


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
def insertPrefetch0():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("xegpu.dpas"),
    )
    with InsertionPoint(sequence.body):
        operand = transform.GetOperandOp(AnyValueType.get(), sequence.bodyTarget, [0])
        xegpu.insert_prefetch(
            operand,
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: insertPrefetch0
    # CHECK: %[[OPR:.*]] = get_operand
    # CHECK: transform.xegpu.insert_prefetch %[[OPR]]


@run
def insertPrefetchNbPrefetch():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("xegpu.dpas"),
    )
    with InsertionPoint(sequence.body):
        operand = transform.GetOperandOp(AnyValueType.get(), sequence.bodyTarget, [0])
        xegpu.insert_prefetch(
            operand,
            nb_prefetch=2,
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: insertPrefetchNbPrefetch
    # CHECK: %[[OPR:.*]] = get_operand
    # CHECK: transform.xegpu.insert_prefetch %[[OPR]]
    # CHECK-SAME: nb_prefetch = 2


@run
def insertPrefetchNbPrefetchParam():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("xegpu.dpas"),
    )
    with InsertionPoint(sequence.body):
        operand = transform.GetOperandOp(AnyValueType.get(), sequence.bodyTarget, [0])
        int32_t = IntegerType.get_signless(32)
        param_int32_t = transform.ParamType.get(int32_t)
        nb_param = transform.ParamConstantOp(
            param_int32_t,
            IntegerAttr.get(int32_t, 2),
        )
        xegpu.insert_prefetch(
            operand,
            nb_prefetch=nb_param,
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: insertPrefetchNbPrefetchParam
    # CHECK: %[[OPR:.*]] = get_operand
    # CHECK: %[[PARAM_OP:.*]] = transform.param.constant 2
    # CHECK: transform.xegpu.insert_prefetch %[[OPR]]
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
            target_sg_layout=[6, 4],
            target_sg_data=[32, 32],
            target_inst_data=[8, 16],
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: ConvertLayout
    # CHECK: transform.xegpu.convert_layout %
    # CHECK: input_sg_layout = [6, 4]
    # CHECK: input_sg_data = [32, 32]
    # CHECK: input_inst_data = [32, 16]
    # CHECK: target_sg_layout = [6, 4]
    # CHECK: target_sg_data = [32, 32]
    # CHECK: target_inst_data = [8, 16]
