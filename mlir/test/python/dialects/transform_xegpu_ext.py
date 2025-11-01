# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import transform
from mlir.dialects.transform import xegpu
from mlir.dialects.transform import structured


def run(f):
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            print("\nTEST:", f.__name__)
            f()
        print(module)
    return f


@run
def setDescLayoutMinimal():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("xegpu.create_nd_tdesc"),
    )
    with InsertionPoint(sequence.body):
        xegpu.SetDescLayoutOp(sequence.bodyTarget, sg_layout=[6, 4])
        transform.YieldOp()
    # CHECK-LABEL: TEST: setDescLayoutMinimal
    # CHECK: %0 = transform.xegpu.set_desc_layout %
    # CHECK: sg_layout = [6, 4]


@run
def setDescLayoutSgData():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("xegpu.create_nd_tdesc"),
    )
    with InsertionPoint(sequence.body):
        xegpu.SetDescLayoutOp(sequence.bodyTarget, sg_layout=[6, 4], sg_data=[32, 16])
        transform.YieldOp()
    # CHECK-LABEL: TEST: setDescLayoutSgData
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
        xegpu.SetDescLayoutOp(sequence.bodyTarget, sg_layout=[6, 4], inst_data=[8, 16])
        transform.YieldOp()
    # CHECK-LABEL: TEST: setDescLayoutInstData
    # CHECK: %0 = transform.xegpu.set_desc_layout %
    # CHECK: sg_layout = [6, 4]
    # CHECK: inst_data = [8, 16]
