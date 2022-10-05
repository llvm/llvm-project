# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import transform
from mlir.dialects import pdl
from mlir.dialects.transform import loop


def run(f):
  with Context(), Location.unknown():
    module = Module.create()
    with InsertionPoint(module.body):
      print("\nTEST:", f.__name__)
      f()
    print(module)
  return f


@run
def getParentLoop():
  sequence = transform.SequenceOp(transform.FailurePropagationMode.PROPAGATE, [], pdl.OperationType.get())
  with InsertionPoint(sequence.body):
    loop.GetParentForOp(sequence.bodyTarget, num_loops=2)
    transform.YieldOp()
  # CHECK-LABEL: TEST: getParentLoop
  # CHECK: = transform.loop.get_parent_for %
  # CHECK: num_loops = 2


@run
def loopOutline():
  sequence = transform.SequenceOp(transform.FailurePropagationMode.PROPAGATE, [], pdl.OperationType.get())
  with InsertionPoint(sequence.body):
    loop.LoopOutlineOp(sequence.bodyTarget, func_name="foo")
    transform.YieldOp()
  # CHECK-LABEL: TEST: loopOutline
  # CHECK: = transform.loop.outline %
  # CHECK: func_name = "foo"


@run
def loopPeel():
  sequence = transform.SequenceOp(transform.FailurePropagationMode.PROPAGATE, [], pdl.OperationType.get())
  with InsertionPoint(sequence.body):
    loop.LoopPeelOp(sequence.bodyTarget)
    transform.YieldOp()
  # CHECK-LABEL: TEST: loopPeel
  # CHECK: = transform.loop.peel %


@run
def loopPipeline():
  sequence = transform.SequenceOp(transform.FailurePropagationMode.PROPAGATE, [], pdl.OperationType.get())
  with InsertionPoint(sequence.body):
    loop.LoopPipelineOp(sequence.bodyTarget, iteration_interval=3)
    transform.YieldOp()
  # CHECK-LABEL: TEST: loopPipeline
  # CHECK: = transform.loop.pipeline %
  # CHECK-DAG: iteration_interval = 3
  # CHECK-DAG: read_latency = 10


@run
def loopUnroll():
  sequence = transform.SequenceOp(transform.FailurePropagationMode.PROPAGATE, [], pdl.OperationType.get())
  with InsertionPoint(sequence.body):
    loop.LoopUnrollOp(sequence.bodyTarget, factor=42)
    transform.YieldOp()
  # CHECK-LABEL: TEST: loopUnroll
  # CHECK: transform.loop.unroll %
  # CHECK: factor = 42
