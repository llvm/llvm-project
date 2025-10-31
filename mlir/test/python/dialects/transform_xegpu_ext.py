# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import transform
from mlir.dialects.transform import xegpu
from mlir.dialects.transform import AnyValueType


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
        desc_handle = xegpu.GetDescOp(operand)
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
        xegpu.SetDescLayoutOp(sequence.bodyTarget, sg_layout=[6, 4], sg_data=[32, 16])
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
        xegpu.SetDescLayoutOp(
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
        xegpu.SetOpLayoutAttrOp(
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
        xegpu.SetOpLayoutAttrOp(
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
        transform.OperationType.get("gpu.lauch"),
    )
    with InsertionPoint(sequence.body):
        xegpu.SetGPULaunchThreadsOp(sequence.bodyTarget, threads=[8, 4, 1])
        transform.YieldOp()
    # CHECK-LABEL: TEST: setGPULaunchThreadsOp
    # CHECK: transform.xegpu.set_gpu_launch_threads
    # CHECK: threads = [8, 4, 1]
