# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import transform
from mlir.dialects.transform import pdl as transform_pdl


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
                                  [transform.AnyOpType.get()],
                                  transform.AnyOpType.get())
  with InsertionPoint(sequence.body):
    transform.YieldOp([sequence.bodyTarget])
  # CHECK-LABEL: TEST: testSequenceOp
  # CHECK: = transform.sequence -> !transform.any_op failures(propagate) {
  # CHECK: ^{{.*}}(%[[ARG0:.+]]: !transform.any_op):
  # CHECK:   yield %[[ARG0]] : !transform.any_op
  # CHECK: }


@run
def testNestedSequenceOp():
  sequence = transform.SequenceOp(transform.FailurePropagationMode.PROPAGATE, [], transform.AnyOpType.get())
  with InsertionPoint(sequence.body):
    nested = transform.SequenceOp(transform.FailurePropagationMode.PROPAGATE, [], sequence.bodyTarget)
    with InsertionPoint(nested.body):
      doubly_nested = transform.SequenceOp(
          transform.FailurePropagationMode.PROPAGATE,
          [transform.AnyOpType.get()], nested.bodyTarget)
      with InsertionPoint(doubly_nested.body):
        transform.YieldOp([doubly_nested.bodyTarget])
      transform.YieldOp()
    transform.YieldOp()
  # CHECK-LABEL: TEST: testNestedSequenceOp
  # CHECK: transform.sequence failures(propagate) {
  # CHECK: ^{{.*}}(%[[ARG0:.+]]: !transform.any_op):
  # CHECK:   sequence %[[ARG0]] : !transform.any_op failures(propagate) {
  # CHECK:   ^{{.*}}(%[[ARG1:.+]]: !transform.any_op):
  # CHECK:     = sequence %[[ARG1]] : !transform.any_op -> !transform.any_op failures(propagate) {
  # CHECK:     ^{{.*}}(%[[ARG2:.+]]: !transform.any_op):
  # CHECK:       yield %[[ARG2]] : !transform.any_op
  # CHECK:     }
  # CHECK:   }
  # CHECK: }


@run
def testSequenceOpWithExtras():
  sequence = transform.SequenceOp(
      transform.FailurePropagationMode.PROPAGATE, [], transform.AnyOpType.get(),
      [transform.AnyOpType.get(),
       transform.OperationType.get("foo.bar")])
  with InsertionPoint(sequence.body):
    transform.YieldOp()
  # CHECK-LABEL: TEST: testSequenceOpWithExtras
  # CHECK: transform.sequence failures(propagate)
  # CHECK: ^{{.*}}(%{{.*}}: !transform.any_op, %{{.*}}: !transform.any_op, %{{.*}}: !transform.op<"foo.bar">):


@run
def testNestedSequenceOpWithExtras():
  sequence = transform.SequenceOp(
      transform.FailurePropagationMode.PROPAGATE, [], transform.AnyOpType.get(),
      [transform.AnyOpType.get(),
       transform.OperationType.get("foo.bar")])
  with InsertionPoint(sequence.body):
    nested = transform.SequenceOp(transform.FailurePropagationMode.PROPAGATE,
                                  [], sequence.bodyTarget,
                                  sequence.bodyExtraArgs)
    with InsertionPoint(nested.body):
      transform.YieldOp()
    transform.YieldOp()
  # CHECK-LABEL: TEST: testNestedSequenceOpWithExtras
  # CHECK: transform.sequence failures(propagate)
  # CHECK: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op, %[[ARG1:.*]]: !transform.any_op, %[[ARG2:.*]]: !transform.op<"foo.bar">):
  # CHECK:   sequence %[[ARG0]], %[[ARG1]], %[[ARG2]] : (!transform.any_op, !transform.any_op, !transform.op<"foo.bar">)


@run
def testTransformPDLOps():
  withPdl = transform_pdl.WithPDLPatternsOp(transform.AnyOpType.get())
  with InsertionPoint(withPdl.body):
    sequence = transform.SequenceOp(transform.FailurePropagationMode.PROPAGATE,
                                    [transform.AnyOpType.get()],
                                    withPdl.bodyTarget)
    with InsertionPoint(sequence.body):
      match = transform_pdl.PDLMatchOp(transform.AnyOpType.get(), sequence.bodyTarget, "pdl_matcher")
      transform.YieldOp(match)
  # CHECK-LABEL: TEST: testTransformPDLOps
  # CHECK: transform.with_pdl_patterns {
  # CHECK: ^{{.*}}(%[[ARG0:.+]]: !transform.any_op):
  # CHECK:   = sequence %[[ARG0]] : !transform.any_op -> !transform.any_op failures(propagate) {
  # CHECK:   ^{{.*}}(%[[ARG1:.+]]: !transform.any_op):
  # CHECK:     %[[RES:.+]] = pdl_match @pdl_matcher in %[[ARG1]]
  # CHECK:     yield %[[RES]] : !transform.any_op
  # CHECK:   }
  # CHECK: }


@run
def testGetClosestIsolatedParentOp():
  sequence = transform.SequenceOp(transform.FailurePropagationMode.PROPAGATE, [], transform.AnyOpType.get())
  with InsertionPoint(sequence.body):
    transform.GetClosestIsolatedParentOp(transform.AnyOpType.get(), sequence.bodyTarget)
    transform.YieldOp()
  # CHECK-LABEL: TEST: testGetClosestIsolatedParentOp
  # CHECK: transform.sequence
  # CHECK: ^{{.*}}(%[[ARG1:.+]]: !transform.any_op):
  # CHECK:   = get_closest_isolated_parent %[[ARG1]]


@run
def testMergeHandlesOp():
  sequence = transform.SequenceOp(transform.FailurePropagationMode.PROPAGATE, [], transform.AnyOpType.get())
  with InsertionPoint(sequence.body):
    transform.MergeHandlesOp([sequence.bodyTarget])
    transform.YieldOp()
  # CHECK-LABEL: TEST: testMergeHandlesOp
  # CHECK: transform.sequence
  # CHECK: ^{{.*}}(%[[ARG1:.+]]: !transform.any_op):
  # CHECK:   = merge_handles %[[ARG1]]


@run
def testReplicateOp():
  with_pdl = transform_pdl.WithPDLPatternsOp(transform.AnyOpType.get())
  with InsertionPoint(with_pdl.body):
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.PROPAGATE, [], with_pdl.bodyTarget)
    with InsertionPoint(sequence.body):
      m1 = transform_pdl.PDLMatchOp(transform.AnyOpType.get(), sequence.bodyTarget, "first")
      m2 = transform_pdl.PDLMatchOp(transform.AnyOpType.get(), sequence.bodyTarget, "second")
      transform.ReplicateOp(m1, [m2])
      transform.YieldOp()
  # CHECK-LABEL: TEST: testReplicateOp
  # CHECK: %[[FIRST:.+]] = pdl_match
  # CHECK: %[[SECOND:.+]] = pdl_match
  # CHECK: %{{.*}} = replicate num(%[[FIRST]]) %[[SECOND]]
