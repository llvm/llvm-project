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
def testInterchange():
  sequence = transform.SequenceOp()
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
def testPad():
  sequence = transform.SequenceOp()
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
  sequence = transform.SequenceOp()
  with InsertionPoint(sequence.body):
    structured.ScalarizeOp(sequence.bodyTarget)
    transform.YieldOp()
  # CHECK-LABEL: TEST: testScalarize
  # CHECK: transform.structured.scalarize


@run
def testTileCompact():
  sequence = transform.SequenceOp()
  with InsertionPoint(sequence.body):
    structured.TileOp(sequence.bodyTarget, sizes=[4, 8], interchange=[0, 1])
    transform.YieldOp()
  # CHECK-LABEL: TEST: testTileCompact
  # CHECK: transform.sequence
  # CHECK: %{{.+}}, %{{.+}}:2 = transform.structured.tile
  # CHECK-DAG: interchange = [0, 1]
  # CHECK-DAG: sizes = [4, 8]


@run
def testTileAttributes():
  sequence = transform.SequenceOp()
  attr = ArrayAttr.get(
      [IntegerAttr.get(IntegerType.get_signless(64), x) for x in [4, 8]])
  ichange = ArrayAttr.get(
      [IntegerAttr.get(IntegerType.get_signless(64), x) for x in [0, 1]])
  with InsertionPoint(sequence.body):
    structured.TileOp(sequence.bodyTarget, sizes=attr, interchange=ichange)
    transform.YieldOp()
  # CHECK-LABEL: TEST: testTileAttributes
  # CHECK: transform.sequence
  # CHECK: structured.tile
  # CHECK-DAG: interchange = [0, 1]
  # CHECK-DAG: sizes = [4, 8]


@run
def testTileZero():
  sequence = transform.SequenceOp()
  with InsertionPoint(sequence.body):
    structured.TileOp(
        sequence.bodyTarget, sizes=[4, 0, 2, 0], interchange=[0, 1, 2, 3])
    transform.YieldOp()
  # CHECK-LABEL: TEST: testTileZero
  # CHECK: transform.sequence
  # CHECK: %{{.+}}, %{{.+}}:2 = transform.structured.tile
  # CHECK-DAG: interchange = [0, 1, 2, 3]
  # CHECK-DAG: sizes = [4, 0, 2, 0]


@run
def testVectorize():
  sequence = transform.SequenceOp()
  with InsertionPoint(sequence.body):
    structured.VectorizeOp(sequence.bodyTarget, vectorize_padding=True)
    transform.YieldOp()
  # CHECK-LABEL: TEST: testVectorize
  # CHECK: transform.sequence
  # CHECK: = transform.structured.vectorize
  # CHECK: vectorize_padding = true
