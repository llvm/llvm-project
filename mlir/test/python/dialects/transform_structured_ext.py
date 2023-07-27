# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import transform
from mlir.dialects import pdl
from mlir.dialects.transform import structured
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
def testDecompose():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.PROPAGATE, [], pdl.OperationType.get()
    )
    with InsertionPoint(sequence.body):
        structured.DecomposeOp(sequence.bodyTarget)
        transform.YieldOp()
    # CHECK-LABEL: TEST: testDecompose
    # CHECK: transform.sequence
    # CHECK: transform.structured.decompose


@run
def testFuseIntoContainingOpTypes():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.PROPAGATE, [], transform.AnyOpType.get()
    )
    with InsertionPoint(sequence.body):
        fused = structured.MatchOp.match_op_names(sequence.bodyTarget, ["test.dummy"])
        containing = structured.MatchOp.match_op_names(
            sequence.bodyTarget, ["test.dummy"]
        )
        structured.FuseIntoContainingOp(
            transform.OperationType.get("test.dummy"),
            transform.OperationType.get("test.dummy"),
            fused,
            containing,
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: testFuseIntoContainingOpTypes
    # CHECK: = transform.structured.fuse_into_containing_op
    # CHECK-SAME: (!transform.any_op, !transform.any_op) -> (!transform.op<"test.dummy">, !transform.op<"test.dummy">)


@run
def testFuseIntoContainingOpCompact():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.PROPAGATE, [], transform.AnyOpType.get()
    )
    with InsertionPoint(sequence.body):
        fused = structured.MatchOp.match_op_names(sequence.bodyTarget, ["test.dummy"])
        containing = structured.MatchOp.match_op_names(
            sequence.bodyTarget, ["test.dummy"]
        )
        structured.FuseIntoContainingOp(fused, containing)
        transform.YieldOp()
    # CHECK-LABEL: TEST: testFuseIntoContainingOpCompact
    # CHECK: = transform.structured.fuse_into_containing_op
    # CHECK-SAME: (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)


@run
def testGeneralize():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.PROPAGATE, [], pdl.OperationType.get()
    )
    with InsertionPoint(sequence.body):
        structured.GeneralizeOp(sequence.bodyTarget)
        transform.YieldOp()
    # CHECK-LABEL: TEST: testGeneralize
    # CHECK: transform.sequence
    # CHECK: transform.structured.generalize


@run
def testInterchange():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.PROPAGATE, [], pdl.OperationType.get()
    )
    with InsertionPoint(sequence.body):
        structured.InterchangeOp(sequence.bodyTarget, iterator_interchange=[1, 0])
        transform.YieldOp()
    # CHECK-LABEL: TEST: testInterchange
    # CHECK: transform.sequence
    # CHECK: transform.structured.interchange
    # CHECK: iterator_interchange = [1, 0]


@run
def testMatchOpNamesString():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.PROPAGATE, [], transform.AnyOpType.get()
    )
    with InsertionPoint(sequence.body):
        structured.MatchOp.match_op_names(sequence.bodyTarget, "test.dummy")
        transform.YieldOp()
    # CHECK-LABEL: TEST: testMatchOpNamesString
    # CHECK: transform.structured.match ops
    # CHECK-SAME: ["test.dummy"]
    # CHECK-SAME: (!transform.any_op) -> !transform.any_op


@run
def testMatchOpNamesList():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.PROPAGATE, [], transform.AnyOpType.get()
    )
    with InsertionPoint(sequence.body):
        structured.MatchOp.match_op_names(sequence.bodyTarget, ["test.dummy"])
        transform.YieldOp()
    # CHECK-LABEL: TEST: testMatchOpNamesList
    # CHECK: transform.structured.match ops
    # CHECK-SAME: ["test.dummy"]
    # CHECK-SAME: (!transform.any_op) -> !transform.any_op


@run
def testMatchOpNamesTyped():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.PROPAGATE, [], transform.AnyOpType.get()
    )
    with InsertionPoint(sequence.body):
        structured.MatchOp.match_op_names(
            transform.OperationType.get("test.dummy"),
            sequence.bodyTarget,
            ["test.dummy"],
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: testMatchOpNamesTyped
    # CHECK: transform.structured.match ops
    # CHECK-SAME: ["test.dummy"]
    # CHECK-SAME: (!transform.any_op) -> !transform.op<"test.dummy">


@run
def testMultitileSizes():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.PROPAGATE, [], pdl.OperationType.get()
    )
    with InsertionPoint(sequence.body):
        structured.MultiTileSizesOp(
            pdl.OperationType.get(), sequence.bodyTarget, dimension=1, target_size=42
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: testMultitileSizes
    # CHECK: transform.sequence
    # CHECK: transform.structured.multitile_sizes
    # CHECK-DAG: dimension = 1
    # CHECK-DAG: target_size = 42


@run
def testPad():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.PROPAGATE, [], pdl.OperationType.get()
    )
    with InsertionPoint(sequence.body):
        structured.PadOp(
            sequence.bodyTarget,
            padding_values=[FloatAttr.get_f32(42.0)],
            padding_dimensions=[1],
            transpose_paddings=[[1, 0]],
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: testPad
    # CHECK: transform.sequence
    # CHECK: transform.structured.pad
    # CHECK-DAG: padding_values = [4.200000e+01 : f32]
    # CHECK-DAG: padding_dimensions = [1]
    # CHECK-DAG: transpose_paddings = {{\[}}[1, 0]]
    # (pack_paddings has default values)


@run
def testScalarize():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.PROPAGATE, [], pdl.OperationType.get()
    )
    with InsertionPoint(sequence.body):
        structured.ScalarizeOp(sequence.bodyTarget)
        transform.YieldOp()
    # CHECK-LABEL: TEST: testScalarize
    # CHECK: transform.structured.scalarize


@run
def testSplit():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.PROPAGATE, [], pdl.OperationType.get()
    )
    with InsertionPoint(sequence.body):
        split = structured.SplitOp(sequence.bodyTarget, dimension=1, split_point=42)
        structured.SplitOp(split.results[0], dimension=3, split_point=split.results[1])
        transform.YieldOp()
    # CHECK-LABEL: TEST: testSplit
    # CHECK: %[[F:.+]], %[[S:.+]] = transform.structured.split %{{.*}} after 42 {dimension = 1
    # CHECK: transform.structured.split %[[F]] after %[[S]] {dimension = 3


@run
def testTileCompact():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.PROPAGATE, [], pdl.OperationType.get()
    )
    with InsertionPoint(sequence.body):
        structured.TileOp(sequence.bodyTarget, sizes=[4, 8], interchange=[0, 1])
        transform.YieldOp()
    # CHECK-LABEL: TEST: testTileCompact
    # CHECK: transform.sequence
    # CHECK: %{{.+}}, %{{.+}}:2 = transform.structured.tile %{{.*}}[4, 8]
    # CHECK: interchange = [0, 1]


@run
def testTileAttributes():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.PROPAGATE, [], pdl.OperationType.get()
    )
    attr = DenseI64ArrayAttr.get([4, 8])
    ichange = DenseI64ArrayAttr.get([0, 1])
    with InsertionPoint(sequence.body):
        structured.TileOp(sequence.bodyTarget, sizes=attr, interchange=ichange)
        transform.YieldOp()
    # CHECK-LABEL: TEST: testTileAttributes
    # CHECK: transform.sequence
    # CHECK: %{{.+}}, %{{.+}}:2 = transform.structured.tile %{{.*}}[4, 8]
    # CHECK: interchange = [0, 1]


@run
def testTileZero():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.PROPAGATE, [], pdl.OperationType.get()
    )
    with InsertionPoint(sequence.body):
        structured.TileOp(
            sequence.bodyTarget, sizes=[4, 0, 2, 0], interchange=[0, 1, 2, 3]
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: testTileZero
    # CHECK: transform.sequence
    # CHECK: %{{.+}}, %{{.+}}:2 = transform.structured.tile %{{.*}}[4, 0, 2, 0]
    # CHECK: interchange = [0, 1, 2, 3]


@run
def testTileDynamic():
    with_pdl = transform_pdl.WithPDLPatternsOp(pdl.OperationType.get())
    with InsertionPoint(with_pdl.body):
        sequence = transform.SequenceOp(
            transform.FailurePropagationMode.PROPAGATE, [], with_pdl.bodyTarget
        )
        with InsertionPoint(sequence.body):
            m1 = transform_pdl.PDLMatchOp(
                pdl.OperationType.get(), sequence.bodyTarget, "first"
            )
            m2 = transform_pdl.PDLMatchOp(
                pdl.OperationType.get(), sequence.bodyTarget, "second"
            )
            structured.TileOp(sequence.bodyTarget, sizes=[m1, 3, m2, 0])
            transform.YieldOp()
    # CHECK-LABEL: TEST: testTileDynamic
    # CHECK: %[[FIRST:.+]] = pdl_match
    # CHECK: %[[SECOND:.+]] = pdl_match
    # CHECK: %{{.+}}, %{{.+}}:3 = transform.structured.tile %{{.*}}[%[[FIRST]], 3, %[[SECOND]], 0]


@run
def testTileExplicitLoopTypeSingle():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.PROPAGATE, [], transform.AnyOpType.get()
    )
    with InsertionPoint(sequence.body):
        structured.TileOp(
            transform.OperationType.get("scf.for"), sequence.bodyTarget, sizes=[2, 3, 4]
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: testTileExplicitLoopTypeSingle
    # CHECK: = transform.structured.tile %{{.*}} : (!{{.*}}) ->
    # CHECK-COUNT-3: !transform.op<"scf.for">


@run
def testTileExplicitLoopTypeAll():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.PROPAGATE, [], transform.AnyOpType.get()
    )
    types = [
        transform.OperationType.get(x)
        for x in ["scf.for", "scf.parallel", "scf.forall"]
    ]
    with InsertionPoint(sequence.body):
        structured.TileOp(types, sequence.bodyTarget, sizes=[2, 3, 4])
        transform.YieldOp()
    # CHECK-LABEL: TEST: testTileExplicitLoopTypeAll
    # CHECK: = transform.structured.tile
    # CHECK-SAME : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">,
    # CHECK-SAME: !transform.op<"scf.parallel">, !transform.op<"scf.forall">


@run
def testTileToForallCompact():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.PROPAGATE,
        [],
        transform.OperationType.get("linalg.matmul"),
    )
    with InsertionPoint(sequence.body):
        structured.TileToForallOp(sequence.bodyTarget, num_threads=[2, 3, 4])
        transform.YieldOp()
    # CHECK-LABEL: TEST: testTileToForallCompact
    # CHECK: = transform.structured.tile_to_forall_op
    # CHECK-SAME: num_threads [2, 3, 4] tile_sizes []
    # CHECK-SAME: (!transform.op<"linalg.matmul">) -> (!transform.any_op, !transform.any_op)


@run
def testTileToForallLoopsAndTileOpTypes():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.PROPAGATE, [], transform.AnyOpType.get()
    )
    with InsertionPoint(sequence.body):
        structured.TileToForallOp(
            transform.OperationType.get("scf.forall"),  # loops_type
            transform.OperationType.get("linalg.matmul"),  # tiled_op_type
            sequence.bodyTarget,
            num_threads=[2, 3, 4],
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: testTileToForallLoopsAndTileOpTypes
    # CHECK: = transform.structured.tile_to_forall_op
    # CHECK-SAME: num_threads [2, 3, 4] tile_sizes []
    # CHECK-SAME: (!transform.any_op) -> (!transform.op<"scf.forall">, !transform.op<"linalg.matmul">)


@run
def testTileToForallTileSizes():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.PROPAGATE, [], transform.AnyOpType.get()
    )
    with InsertionPoint(sequence.body):
        structured.TileToForallOp(sequence.bodyTarget, tile_sizes=[2, 3, 4])
        transform.YieldOp()
    # CHECK-LABEL: TEST: testTileToForallTileSizes
    # CHECK: = transform.structured.tile_to_forall_op
    # CHECK-SAME: num_threads [] tile_sizes [2, 3, 4]


@run
def testTileToForallMixedDynamic():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.PROPAGATE, [], transform.AnyOpType.get()
    )
    with InsertionPoint(sequence.body):
        n = structured.MatchOp.match_op_names(sequence.bodyTarget, ["test.dummy"])
        structured.TileToForallOp(sequence.bodyTarget, num_threads=[n, 3, 4])
        transform.YieldOp()
    # CHECK-LABEL: TEST: testTileToForallMixedDynamic
    # CHECK: = transform.structured.tile_to_forall_op
    # CHECK-SAME: num_threads [%{{.*}} : !transform.any_op, 3, 4]


@run
def testTileToForallPackedDynamic():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.PROPAGATE, [], transform.AnyOpType.get()
    )
    with InsertionPoint(sequence.body):
        n = structured.MatchOp.match_op_names(sequence.bodyTarget, ["test.dummy"])
        structured.TileToForallOp(sequence.bodyTarget, num_threads=n)
        transform.YieldOp()
    # CHECK-LABEL: TEST: testTileToForallPackedDynamic
    # CHECK: = transform.structured.tile_to_forall_op
    # CHECK-SAME: num_threads *(%0 : !transform.any_op)


@run
def testTileToForallMapping():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.PROPAGATE, [], transform.AnyOpType.get()
    )
    with InsertionPoint(sequence.body):
        mapping = Attribute.parse("[ #gpu.thread<y>, #gpu.thread<x> ]")
        structured.TileToForallOp(
            sequence.bodyTarget, num_threads=[2, 3], mapping=mapping
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: testTileToForallMapping
    # CHECK: = transform.structured.tile_to_forall_op
    # CHECK-SAME: mapping = [#gpu.thread<y>, #gpu.thread<x>]


@run
def testVectorize():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.PROPAGATE, [], pdl.OperationType.get()
    )
    with InsertionPoint(sequence.body):
        structured.VectorizeOp(sequence.bodyTarget, vectorize_padding=True)
        transform.YieldOp()
    # CHECK-LABEL: TEST: testVectorize
    # CHECK: transform.sequence
    # CHECK: = transform.structured.vectorize
    # CHECK: {vectorize_padding}
