# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import transform
from mlir.dialects import pdl


def run(f):
  with Context(), Location.unknown():
    module = Module.create()
    with InsertionPoint(module.body):
      print("\nTEST:", f.__name__)
      f()
    print(module)
  return f


@run
def testTypes():
  # CHECK-LABEL: TEST: testTypes
  # CHECK: !transform.any_op
  any_op = transform.AnyOpType.get()
  print(any_op)

  # CHECK: !transform.op<"foo.bar">
  # CHECK: foo.bar
  concrete_op = transform.OperationType.get("foo.bar")
  print(concrete_op)
  print(concrete_op.operation_name)


@run
def testSequenceOp():
  sequence = transform.SequenceOp(transform.FailurePropagationMode.PROPAGATE,
                                  [pdl.OperationType.get()],
                                  pdl.OperationType.get())
  with InsertionPoint(sequence.body):
    transform.YieldOp([sequence.bodyTarget])
  # CHECK-LABEL: TEST: testSequenceOp
  # CHECK: = transform.sequence -> !pdl.operation failures(propagate) {
  # CHECK: ^{{.*}}(%[[ARG0:.+]]: !pdl.operation):
  # CHECK:   yield %[[ARG0]] : !pdl.operation
  # CHECK: }


@run
def testNestedSequenceOp():
  sequence = transform.SequenceOp(transform.FailurePropagationMode.PROPAGATE, [], pdl.OperationType.get())
  with InsertionPoint(sequence.body):
    nested = transform.SequenceOp(transform.FailurePropagationMode.PROPAGATE, [], sequence.bodyTarget)
    with InsertionPoint(nested.body):
      doubly_nested = transform.SequenceOp(
          transform.FailurePropagationMode.PROPAGATE,
          [pdl.OperationType.get()], nested.bodyTarget)
      with InsertionPoint(doubly_nested.body):
        transform.YieldOp([doubly_nested.bodyTarget])
      transform.YieldOp()
    transform.YieldOp()
  # CHECK-LABEL: TEST: testNestedSequenceOp
  # CHECK: transform.sequence failures(propagate) {
  # CHECK: ^{{.*}}(%[[ARG0:.+]]: !pdl.operation):
  # CHECK:   sequence %[[ARG0]] : !pdl.operation failures(propagate) {
  # CHECK:   ^{{.*}}(%[[ARG1:.+]]: !pdl.operation):
  # CHECK:     = sequence %[[ARG1]] : !pdl.operation -> !pdl.operation failures(propagate) {
  # CHECK:     ^{{.*}}(%[[ARG2:.+]]: !pdl.operation):
  # CHECK:       yield %[[ARG2]] : !pdl.operation
  # CHECK:     }
  # CHECK:   }
  # CHECK: }


@run
def testTransformPDLOps():
  withPdl = transform.WithPDLPatternsOp(pdl.OperationType.get())
  with InsertionPoint(withPdl.body):
    sequence = transform.SequenceOp(transform.FailurePropagationMode.PROPAGATE,
                                    [pdl.OperationType.get()],
                                    withPdl.bodyTarget)
    with InsertionPoint(sequence.body):
      match = transform.PDLMatchOp(pdl.OperationType.get(), sequence.bodyTarget, "pdl_matcher")
      transform.YieldOp(match)
  # CHECK-LABEL: TEST: testTransformPDLOps
  # CHECK: transform.with_pdl_patterns {
  # CHECK: ^{{.*}}(%[[ARG0:.+]]: !pdl.operation):
  # CHECK:   = sequence %[[ARG0]] : !pdl.operation -> !pdl.operation failures(propagate) {
  # CHECK:   ^{{.*}}(%[[ARG1:.+]]: !pdl.operation):
  # CHECK:     %[[RES:.+]] = pdl_match @pdl_matcher in %[[ARG1]]
  # CHECK:     yield %[[RES]] : !pdl.operation
  # CHECK:   }
  # CHECK: }


@run
def testGetClosestIsolatedParentOp():
  sequence = transform.SequenceOp(transform.FailurePropagationMode.PROPAGATE, [], pdl.OperationType.get())
  with InsertionPoint(sequence.body):
    transform.GetClosestIsolatedParentOp(pdl.OperationType.get(), sequence.bodyTarget)
    transform.YieldOp()
  # CHECK-LABEL: TEST: testGetClosestIsolatedParentOp
  # CHECK: transform.sequence
  # CHECK: ^{{.*}}(%[[ARG1:.+]]: !pdl.operation):
  # CHECK:   = get_closest_isolated_parent %[[ARG1]]


@run
def testMergeHandlesOp():
  sequence = transform.SequenceOp(transform.FailurePropagationMode.PROPAGATE, [], pdl.OperationType.get())
  with InsertionPoint(sequence.body):
    transform.MergeHandlesOp([sequence.bodyTarget])
    transform.YieldOp()
  # CHECK-LABEL: TEST: testMergeHandlesOp
  # CHECK: transform.sequence
  # CHECK: ^{{.*}}(%[[ARG1:.+]]: !pdl.operation):
  # CHECK:   = merge_handles %[[ARG1]]


@run
def testReplicateOp():
  with_pdl = transform.WithPDLPatternsOp(pdl.OperationType.get())
  with InsertionPoint(with_pdl.body):
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.PROPAGATE, [], with_pdl.bodyTarget)
    with InsertionPoint(sequence.body):
      m1 = transform.PDLMatchOp(pdl.OperationType.get(), sequence.bodyTarget, "first")
      m2 = transform.PDLMatchOp(pdl.OperationType.get(), sequence.bodyTarget, "second")
      transform.ReplicateOp(m1, [m2])
      transform.YieldOp()
  # CHECK-LABEL: TEST: testReplicateOp
  # CHECK: %[[FIRST:.+]] = pdl_match
  # CHECK: %[[SECOND:.+]] = pdl_match
  # CHECK: %{{.*}} = replicate num(%[[FIRST]]) %[[SECOND]]
