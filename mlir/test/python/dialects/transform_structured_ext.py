# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import transform
from mlir.dialects import pdl
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
def testDecompose():
  sequence = transform.SequenceOp(transform.FailurePropagationMode.PROPAGATE)
  with InsertionPoint(sequence.body):
    structured.DecomposeOp(sequence.bodyTarget)
    transform.YieldOp()
  # CHECK-LABEL: TEST: testDecompose
  # CHECK: transform.sequence
  # CHECK: transform.structured.decompose


@run
def testGeneralize():
  sequence = transform.SequenceOp(transform.FailurePropagationMode.PROPAGATE)
  with InsertionPoint(sequence.body):
    structured.GeneralizeOp(sequence.bodyTarget)
    transform.YieldOp()
  # CHECK-LABEL: TEST: testGeneralize
  # CHECK: transform.sequence
  # CHECK: transform.structured.generalize


@run
def testInterchange():
  sequence = transform.SequenceOp(transform.FailurePropagationMode.PROPAGATE)
  with InsertionPoint(sequence.body):
    structured.InterchangeOp(
        sequence.bodyTarget,
        iterator_interchange=[
            IntegerAttr.get(IntegerType.get_signless(64), 1), 0
        ])
    transform.YieldOp()
  # CHECK-LABEL: TEST: testInterchange
  # CHECK: transform.sequence
  # CHECK: transform.structured.interchange
  # CHECK: iterator_interchange = [1, 0]


@run
def testMultitileSizes():
  sequence = transform.SequenceOp(transform.FailurePropagationMode.PROPAGATE)
  with InsertionPoint(sequence.body):
    structured.MultiTileSizesOp(
        sequence.bodyTarget, dimension=1, target_size=42)
    transform.YieldOp()
  # CHECK-LABEL: TEST: testMultitileSizes
  # CHECK: transform.sequence
  # CHECK: transform.structured.multitile_sizes
  # CHECK-DAG: dimension = 1
  # CHECK-DAG: target_size = 42


@run
def testPad():
  sequence = transform.SequenceOp(transform.FailurePropagationMode.PROPAGATE)
  with InsertionPoint(sequence.body):
    structured.PadOp(
        sequence.bodyTarget,
        padding_values=[FloatAttr.get_f32(42.0)],
        padding_dimensions=[1],
        transpose_paddings=[[1, 0]])
    transform.YieldOp()
  # CHECK-LABEL: TEST: testPad
  # CHECK: transform.sequence
  # CHECK: transform.structured.pad
  # CHECK-DAG: padding_values = [4.200000e+01 : f32]
  # CHECK-DAG: padding_dimensions = [1]
  # CHECK-DAG: transpose_paddings = {{\[}}[1, 0]]
  # CHECK-DAG: hoist_paddings = []
  # CHECK-DAG: pack_paddings = []


@run
def testScalarize():
  sequence = transform.SequenceOp(transform.FailurePropagationMode.PROPAGATE)
  with InsertionPoint(sequence.body):
    structured.ScalarizeOp(sequence.bodyTarget)
    transform.YieldOp()
  # CHECK-LABEL: TEST: testScalarize
  # CHECK: transform.structured.scalarize


@run
def testSplit():
  sequence = transform.SequenceOp(transform.FailurePropagationMode.PROPAGATE)
  with InsertionPoint(sequence.body):
    split = structured.SplitOp(sequence.bodyTarget, dimension=1, split_point=42)
    structured.SplitOp(
        split.results[0], dimension=3, split_point=split.results[1])
    transform.YieldOp()
  # CHECK-LABEL: TEST: testSplit
  # CHECK: %[[F:.+]], %[[S:.+]] = transform.structured.split %{{.*}} after 42 {dimension = 1
  # CHECK: transform.structured.split %[[F]] after %[[S]] {dimension = 3


@run
def testTileCompact():
  sequence = transform.SequenceOp(transform.FailurePropagationMode.PROPAGATE)
  with InsertionPoint(sequence.body):
    structured.TileOp(sequence.bodyTarget, sizes=[4, 8], interchange=[0, 1])
    transform.YieldOp()
  # CHECK-LABEL: TEST: testTileCompact
  # CHECK: transform.sequence
  # CHECK: %{{.+}}, %{{.+}}:2 = transform.structured.tile %{{.*}}[4, 8]
  # CHECK: interchange = [0, 1]


@run
def testTileAttributes():
  sequence = transform.SequenceOp(transform.FailurePropagationMode.PROPAGATE)
  attr = ArrayAttr.get(
      [IntegerAttr.get(IntegerType.get_signless(64), x) for x in [4, 8]])
  ichange = ArrayAttr.get(
      [IntegerAttr.get(IntegerType.get_signless(64), x) for x in [0, 1]])
  with InsertionPoint(sequence.body):
    structured.TileOp(sequence.bodyTarget, sizes=attr, interchange=ichange)
    transform.YieldOp()
  # CHECK-LABEL: TEST: testTileAttributes
  # CHECK: transform.sequence
  # CHECK: %{{.+}}, %{{.+}}:2 = transform.structured.tile %{{.*}}[4, 8]
  # CHECK: interchange = [0, 1]


@run
def testTileZero():
  sequence = transform.SequenceOp(transform.FailurePropagationMode.PROPAGATE)
  with InsertionPoint(sequence.body):
    structured.TileOp(
        sequence.bodyTarget, sizes=[4, 0, 2, 0], interchange=[0, 1, 2, 3])
    transform.YieldOp()
  # CHECK-LABEL: TEST: testTileZero
  # CHECK: transform.sequence
  # CHECK: %{{.+}}, %{{.+}}:2 = transform.structured.tile %{{.*}}[4, 0, 2, 0]
  # CHECK: interchange = [0, 1, 2, 3]


@run
def testTileDynamic():
  with_pdl = transform.WithPDLPatternsOp()
  with InsertionPoint(with_pdl.body):
    sequence = transform.SequenceOp(transform.FailurePropagationMode.PROPAGATE,
                                    with_pdl.bodyTarget)
    with InsertionPoint(sequence.body):
      m1 = transform.PDLMatchOp(sequence.bodyTarget, "first")
      m2 = transform.PDLMatchOp(sequence.bodyTarget, "second")
      structured.TileOp(sequence.bodyTarget, sizes=[m1, 3, m2, 0])
      transform.YieldOp()
  # CHECK-LABEL: TEST: testTileDynamic
  # CHECK: %[[FIRST:.+]] = pdl_match
  # CHECK: %[[SECOND:.+]] = pdl_match
  # CHECK: %{{.+}}, %{{.+}}:3 = transform.structured.tile %{{.*}}[%[[FIRST]], 3, %[[SECOND]], 0]


@run
def testVectorize():
  sequence = transform.SequenceOp(transform.FailurePropagationMode.PROPAGATE)
  with InsertionPoint(sequence.body):
    structured.VectorizeOp(sequence.bodyTarget, vectorize_padding=True)
    transform.YieldOp()
  # CHECK-LABEL: TEST: testVectorize
  # CHECK: transform.sequence
  # CHECK: = transform.structured.vectorize
  # CHECK: {vectorize_padding}
