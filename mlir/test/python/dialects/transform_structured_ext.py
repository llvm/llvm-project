# RUN: %PYTHON %s | FileCheck %s

import functools
from typing import Callable

from mlir.ir import *
from mlir.dialects import transform
from mlir.dialects import pdl
from mlir.dialects.transform import structured
from mlir.dialects.transform import pdl as transform_pdl
from mlir.dialects.transform.extras import constant_param


def run(f):
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            print("\nTEST:", f.__name__)
            f()
        module.operation.verify()
        print(module)
    return f


def create_sequence(func: Callable) -> Callable:
    @functools.wraps(func)
    def decorated() -> None:
        sequence = transform.SequenceOp(
            transform.FailurePropagationMode.Propagate,
            [],
            transform.AnyOpType.get(),
        )
        with InsertionPoint(sequence.body):
            func(sequence.bodyTarget)
            transform.YieldOp()

    return decorated


@run
@create_sequence
def testBufferizeToAllocationOpCompact(target):
    structured.BufferizeToAllocationOp(target)
    # CHECK-LABEL: TEST: testBufferizeToAllocationOpCompact
    # CHECK: transform.sequence
    # CHECK: transform.structured.bufferize_to_allocation


@run
@create_sequence
def testBufferizeToAllocationOpArgs(target):
    structured.BufferizeToAllocationOp(
        target,
        memory_space=3,
        memcpy_op="memref.copy",
        alloc_op="memref.alloca",
        bufferize_destination_only=True,
    )
    # CHECK-LABEL: TEST: testBufferizeToAllocationOpArgs
    # CHECK: transform.sequence
    # CHECK: transform.structured.bufferize_to_allocation
    # CHECK-SAME: alloc_op = "memref.alloca"
    # CHECK-SAME: bufferize_destination_only
    # CHECK-SAME: memcpy_op = "memref.copy"
    # CHECK-SAME: memory_space = 3


@run
@create_sequence
def testDecompose(target):
    structured.DecomposeOp(target)
    # CHECK-LABEL: TEST: testDecompose
    # CHECK: transform.sequence
    # CHECK: transform.structured.decompose


@run
@create_sequence
def testFuseIntoContainingOpTypes(target):
    fused = structured.MatchOp.match_op_names(target, ["test.dummy"])
    containing = structured.MatchOp.match_op_names(target, ["test.dummy"])
    structured.FuseIntoContainingOp(
        transform.OperationType.get("test.dummy"),
        transform.OperationType.get("test.dummy"),
        fused,
        containing,
    )
    # CHECK-LABEL: TEST: testFuseIntoContainingOpTypes
    # CHECK: = transform.structured.fuse_into_containing_op
    # CHECK-SAME: (!transform.any_op, !transform.any_op) -> (!transform.op<"test.dummy">, !transform.op<"test.dummy">)


@run
@create_sequence
def testFuseIntoContainingOpCompact(target):
    fused = structured.MatchOp.match_op_names(target, ["test.dummy"])
    containing = structured.MatchOp.match_op_names(target, ["test.dummy"])
    structured.FuseIntoContainingOp(fused, containing)
    # CHECK-LABEL: TEST: testFuseIntoContainingOpCompact
    # CHECK: = transform.structured.fuse_into_containing_op
    # CHECK-SAME: (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)


@run
@create_sequence
def testFuseOpCompact(target):
    structured.FuseOp(
        target, tile_sizes=[4, 8], tile_interchange=[0, 1], apply_cleanup=True
    )
    # CHECK-LABEL: TEST: testFuseOpCompact
    # CHECK: transform.sequence
    # CHECK: %{{.+}}, %{{.+}}:2 = transform.structured.fuse %{{.*}} tile_sizes [4, 8]
    # CHECK-SAME: interchange [0, 1] {apply_cleanup}
    # CHECK-SAME: (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)


@run
@create_sequence
def testFuseOpCompactForall(target):
    structured.FuseOp(
        target,
        tile_sizes=[4, 8],
        apply_cleanup=True,
        use_forall=True,
    )
    # CHECK-LABEL: TEST: testFuseOpCompact
    # CHECK: transform.sequence
    # CHECK: %{{.+}}, %{{.+}} = transform.structured.fuse %{{.*}} tile_sizes [4, 8]
    # CHECK-SAME: {apply_cleanup, use_forall}
    # CHECK-SAME: (!transform.any_op) -> (!transform.any_op, !transform.any_op)


@run
@create_sequence
def testFuseOpNoArg(target):
    structured.FuseOp(target)
    # CHECK-LABEL: TEST: testFuseOpNoArg
    # CHECK: transform.sequence
    # CHECK: %{{.+}} = transform.structured.fuse %{{.*}} :
    # CHECK-SAME: (!transform.any_op) -> !transform.any_op


@run
@create_sequence
def testFuseOpParams(target):
    structured.FuseOp(
        target,
        tile_sizes=[constant_param(4), Attribute.parse("8")],
        tile_interchange=[constant_param(0), Attribute.parse("1")],
    )
    # CHECK-LABEL: TEST: testFuseOpParams
    # CHECK: transform.sequence
    # CHECK-DAG: %[[P:.*]] = transform.param.constant 4
    # CHECK-DAG: %[[I:.*]] = transform.param.constant 0
    # CHECK: %{{.+}}, %{{.+}}:2 = transform.structured.fuse
    # CHECK-SAME: tile_sizes [%[[P]], 8]
    # CHECK-SAME: interchange [%[[I]], 1]
    # CHECK-SAME: (!transform.any_op, !transform.param<i64>, !transform.param<i64>) -> (!transform.any_op, !transform.any_op, !transform.any_op)


@run
@create_sequence
def testFuseOpHandles(target):
    size1 = structured.MatchOp.match_op_names(target, ["arith.constant"])
    ichange1 = structured.MatchOp.match_op_names(target, ["arith.constant"])
    structured.FuseOp(
        target,
        tile_sizes=[size1, 8],
        tile_interchange=[ichange1, 1],
    )
    # CHECK-LABEL: TEST: testFuseOpHandles
    # CHECK: transform.sequence
    # CHECK: %[[H:.*]] = transform.structured.match
    # CHECK: %[[I:.*]] = transform.structured.match
    # CHECK: %{{.+}}, %{{.+}}:2 = transform.structured.fuse
    # CHECK-SAME: tile_sizes [%[[H]], 8]
    # CHECK-SAME: interchange [%[[I]], 1]
    # CHECK-SAME: (!transform.any_op, !transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)


@run
@create_sequence
def testFuseOpAttributes(target):
    attr = DenseI64ArrayAttr.get([4, 8])
    ichange = DenseI64ArrayAttr.get([0, 1])
    structured.FuseOp(target, tile_sizes=attr, tile_interchange=ichange)
    # CHECK-LABEL: TEST: testFuseOpAttributes
    # CHECK: transform.sequence
    # CHECK: %{{.+}}, %{{.+}}:2 = transform.structured.fuse %{{.*}} tile_sizes [4, 8]
    # CHECK-SAME: interchange [0, 1]
    # CHECK-SAME: (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)


@run
@create_sequence
def testGeneralize(target):
    structured.GeneralizeOp(target)
    # CHECK-LABEL: TEST: testGeneralize
    # CHECK: transform.sequence
    # CHECK: transform.structured.generalize


@run
@create_sequence
def testInterchange(target):
    structured.InterchangeOp(target, iterator_interchange=[1, 0])
    # CHECK-LABEL: TEST: testInterchange
    # CHECK: transform.sequence
    # CHECK: transform.structured.interchange
    # CHECK: iterator_interchange = [1, 0]


@run
@create_sequence
def testMapCopyToThreadsOpCompact(target):
    structured.MapCopyToThreadsOp(
        target, total_num_threads=32, desired_bit_alignment=128
    )
    # CHECK-LABEL: TEST: testMapCopyToThreadsOpCompact
    # CHECK: = transform.structured.gpu.map_copy_to_threads
    # CHECK-SAME: total_num_threads = 32
    # CHECK-SAME: desired_bit_alignment = 128
    # CHECK-SAME:  (!transform.any_op) -> (!transform.any_op, !transform.any_op)


@run
@create_sequence
def testMapCopyToThreadsOpTypes(target):
    structured.MapCopyToThreadsOp(
        transform.OperationType.get("test.opA"),
        transform.OperationType.get("test.opB"),
        target,
        total_num_threads=32,
        desired_bit_alignment=128,
    )
    # CHECK-LABEL: TEST: testMapCopyToThreadsOpTypes
    # CHECK: = transform.structured.gpu.map_copy_to_threads
    # CHECK-SAME: total_num_threads = 32
    # CHECK-SAME: desired_bit_alignment = 128
    # CHECK-SAME:  (!transform.any_op) -> (!transform.op<"test.opA">, !transform.op<"test.opB">)


@run
@create_sequence
def testMatchOpNamesString(target):
    structured.MatchOp.match_op_names(target, "test.dummy")
    # CHECK-LABEL: TEST: testMatchOpNamesString
    # CHECK: transform.structured.match ops
    # CHECK-SAME: ["test.dummy"]
    # CHECK-SAME: (!transform.any_op) -> !transform.any_op


@run
@create_sequence
def testMatchOpNamesList(target):
    structured.MatchOp.match_op_names(target, ["test.dummy"])
    # CHECK-LABEL: TEST: testMatchOpNamesList
    # CHECK: transform.structured.match ops
    # CHECK-SAME: ["test.dummy"]
    # CHECK-SAME: (!transform.any_op) -> !transform.any_op


@run
@create_sequence
def testVectorizeNoArgs(target):
    structured.VectorizeOp(target)
    # CHECK-LABEL: TEST: testVectorizeNoArgs
    # CHECK: transform.sequence
    # CHECK: transform.structured.vectorize
    # CHECK-NOT:     vector_sizes


@run
@create_sequence
def testVectorizeStatic(target):
    structured.VectorizeOp(target, [16, 4])
    # CHECK-LABEL: TEST: testVectorizeStatic
    # CHECK: transform.sequence
    # CHECK: transform.structured.vectorize
    # CHECK-SAME:     vector_sizes [16, 4]


@run
@create_sequence
def testVectorizeArray(target):
    sizes = Attribute.parse("[16, 4]")
    structured.VectorizeOp(target, sizes)
    # CHECK-LABEL: TEST: testVectorizeArray
    # CHECK: transform.sequence
    # CHECK: transform.structured.vectorize
    # CHECK-SAME:     vector_sizes [16, 4]


@run
@create_sequence
def testVectorizeMixed(target):
    sz1 = structured.MatchOp.match_op_names(target, ["arith.constant"])
    sz2 = Attribute.parse("4")
    structured.VectorizeOp(target, [sz1, sz2])
    # CHECK-LABEL: TEST: testVectorizeMixed
    # CHECK: transform.sequence
    # CHECK: %[[V0:.*]] = transform.structured.match
    # CHECK: transform.structured.vectorize
    # CHECK-SAME:     vector_sizes [%[[V0]], 4]


@run
@create_sequence
def testVectorizeEmpty(target):
    structured.VectorizeOp(target, [])
    # CHECK-LABEL: TEST: testVectorizeEmpty
    # CHECK: transform.sequence
    # CHECK: transform.structured.vectorize
    # CHECK-NOT:     vector_sizes


@run
@create_sequence
def testVectorizeScalable(target):
    sz1 = structured.MatchOp.match_op_names(target, ["arith.constant"])
    sz2 = Attribute.parse("4")
    structured.VectorizeOp(target, [16, [sz1], [sz2], [8]])
    # CHECK-LABEL: TEST: testVectorizeScalable
    # CHECK: transform.sequence
    # CHECK-DAG: %[[V0:.*]] = transform.structured.match
    # CHECK-DAG: transform.structured.vectorize
    # CHECK-SAME:     vector_sizes [16, [%[[V0]]], [4], [8]]


@run
@create_sequence
def testVectorizeArgs(target):
    structured.VectorizeOp(target, [16, 4], vectorize_nd_extract=True)
    # CHECK-LABEL: TEST: testVectorizeArgs
    # CHECK: transform.sequence
    # CHECK: transform.structured.vectorize
    # CHECK-SAME: vectorize_nd_extract


@run
@create_sequence
def testMatchOpNamesTyped(target):
    structured.MatchOp.match_op_names(
        transform.OperationType.get("test.dummy"),
        target,
        ["test.dummy"],
    )
    # CHECK-LABEL: TEST: testMatchOpNamesTyped
    # CHECK: transform.structured.match ops
    # CHECK-SAME: ["test.dummy"]
    # CHECK-SAME: (!transform.any_op) -> !transform.op<"test.dummy">


@run
@create_sequence
def testMultitileSizesCompact(target):
    structured.MultiTileSizesOp(
        transform.AnyOpType.get(), target, dimension=1, target_size=42
    )
    # CHECK-LABEL: TEST: testMultitileSizes
    # CHECK: transform.sequence
    # CHECK-NOT: divisor
    # CHECK: transform.structured.multitile_sizes
    # CHECK-NOT: divisor
    # CHECK-DAG: dimension = 1
    # CHECK-NOT: divisor
    # CHECK-DAG: target_size = 42
    # CHECK-NOT: divisor


@run
@create_sequence
def testMultitileSizesAllArgs(target):
    structured.MultiTileSizesOp(
        transform.AnyOpType.get(),
        target,
        dimension=1,
        target_size=42,
        divisor=2,
    )
    # CHECK-LABEL: TEST: testMultitileSizes
    # CHECK: transform.sequence
    # CHECK: transform.structured.multitile_sizes
    # CHECK-DAG: dimension = 1
    # CHECK-DAG: divisor = 2
    # CHECK-DAG: target_size = 42


@run
@create_sequence
def testPadOpNoArgs(target):
    structured.PadOp(target)
    # CHECK-LABEL: TEST: testPadOpNoArgs
    # CHECK: transform.sequence
    # CHECK: transform.structured.pad
    # CHECK-NOT: copy_back_op
    # CHECK-NOT: nofold_flags
    # CHECK-NOT: pad_to_multiple_of
    # CHECK-NOT: padding_dimensions
    # CHECK-NOT: padding_values
    # CHECK-NOT: transpose_paddings


@run
@create_sequence
def testPadOpArgs(target):
    structured.PadOp(
        target,
        pad_to_multiple_of=[128],
        padding_values=[FloatAttr.get_f32(42.0), StringAttr.get("0")],
        padding_dimensions=Attribute.parse("[1]"),
        nofold_flags=[0],
        transpose_paddings=[[1, Attribute.parse("0")], Attribute.parse("[0, 1]")],
        copy_back_op="linalg.copy",
    )
    # CHECK-LABEL: TEST: testPadOpArgs
    # CHECK: transform.sequence
    # CHECK: transform.structured.pad
    # CHECK-DAG: pad_to_multiple_of [128]
    # CHECK-DAG: copy_back_op = "linalg.copy"
    # CHECK-DAG: nofold_flags = [0]
    # CHECK-DAG: padding_dimensions = [1]
    # CHECK-DAG: padding_values = [4.200000e+01 : f32, "0"]
    # CHECK-DAG: transpose_paddings = {{\[}}[1, 0], [0, 1]]


@run
@create_sequence
def testPadOpArgsParam(target):
    structured.PadOp(
        target,
        pad_to_multiple_of=[constant_param(128), Attribute.parse("2"), 10],
        padding_dimensions=Attribute.parse("[0, 1, 2]"),
    )
    # CHECK-LABEL: TEST: testPadOpArgsParam
    # CHECK: transform.sequence
    # CHECK-DAG: %[[P:.*]] = transform.param.constant 128
    # CHECK: transform.structured.pad
    # CHECK-DAG: pad_to_multiple_of [%[[P]], 2, 10]
    # CHECK-DAG: padding_dimensions = [0, 1, 2]


@run
@create_sequence
def testScalarize(target):
    structured.ScalarizeOp(target)
    # CHECK-LABEL: TEST: testScalarize
    # CHECK: transform.structured.scalarize


@run
@create_sequence
def testSplit(target):
    handle = structured.SplitOp(target, dimension=1, chunk_sizes=42)
    split = transform.SplitHandleOp(
        [transform.AnyOpType.get(), transform.AnyOpType.get()], handle
    )
    structured.SplitOp(split.results[0], dimension=3, chunk_sizes=split.results[1])
    # CHECK-LABEL: TEST: testSplit
    # CHECK: %[[G:.+]] = transform.structured.split %{{.*}} after 42 {dimension = 1
    # CHECK: %[[F:.+]]:2 = split_handle %[[G]]
    # CHECK: transform.structured.split %[[F]]#0 after %[[F]]#1 {dimension = 3


@run
@create_sequence
def testTileCompact(target):
    structured.TileUsingForOp(target, sizes=[4, 8], interchange=[0, 1])
    # CHECK-LABEL: TEST: testTileCompact
    # CHECK: transform.sequence
    # CHECK: %{{.+}}, %{{.+}}:2 = transform.structured.tile_using_for %{{.*}}[4, 8]
    # CHECK: interchange = [0, 1]


@run
@create_sequence
def testTileAttributes(target):
    attr = DenseI64ArrayAttr.get([4, 8])
    ichange = DenseI64ArrayAttr.get([0, 1])
    structured.TileUsingForOp(target, sizes=attr, interchange=ichange)
    # CHECK-LABEL: TEST: testTileAttributes
    # CHECK: transform.sequence
    # CHECK: %{{.+}}, %{{.+}}:2 = transform.structured.tile_using_for %{{.*}}[4, 8]
    # CHECK: interchange = [0, 1]


@run
@create_sequence
def testTileZero(target):
    structured.TileUsingForOp(target, sizes=[4, 0, 2, 0], interchange=[0, 1, 2, 3])
    # CHECK-LABEL: TEST: testTileZero
    # CHECK: transform.sequence
    # CHECK: %{{.+}}, %{{.+}}:2 = transform.structured.tile_using_for %{{.*}}[4, 0, 2, 0]
    # CHECK: interchange = [0, 1, 2, 3]


@run
def testTileDynamic():
    with_pdl = transform_pdl.WithPDLPatternsOp(pdl.OperationType.get())
    with InsertionPoint(with_pdl.body):
        sequence = transform.SequenceOp(
            transform.FailurePropagationMode.Propagate, [], with_pdl.bodyTarget
        )
        with InsertionPoint(sequence.body):
            m1 = transform_pdl.PDLMatchOp(
                pdl.OperationType.get(), sequence.bodyTarget, "first"
            )
            m2 = transform_pdl.PDLMatchOp(
                pdl.OperationType.get(), sequence.bodyTarget, "second"
            )
            structured.TileUsingForOp(sequence.bodyTarget, sizes=[m1, 3, m2, 0])
            transform.YieldOp()
    # CHECK-LABEL: TEST: testTileDynamic
    # CHECK: %[[FIRST:.+]] = pdl_match
    # CHECK: %[[SECOND:.+]] = pdl_match
    # CHECK: %{{.+}}, %{{.+}}:3 = transform.structured.tile_using_for %{{.*}}[%[[FIRST]], 3, %[[SECOND]], 0]


@run
@create_sequence
def testTileExplicitLoopTypeSingle(target):
    structured.TileUsingForOp(
        transform.OperationType.get("scf.for"), target, sizes=[2, 3, 4]
    )
    # CHECK-LABEL: TEST: testTileExplicitLoopTypeSingle
    # CHECK: = transform.structured.tile_using_for %{{.*}} : (!{{.*}}) ->
    # CHECK-COUNT-3: !transform.op<"scf.for">


@run
@create_sequence
def testTileExplicitLoopTypeAll(target):
    types = [
        transform.OperationType.get(x)
        for x in ["scf.for", "scf.parallel", "scf.forall"]
    ]
    structured.TileUsingForOp(types, target, sizes=[2, 3, 4])
    # CHECK-LABEL: TEST: testTileExplicitLoopTypeAll
    # CHECK: = transform.structured.tile
    # CHECK-SAME: (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">,
    # CHECK-SAME: !transform.op<"scf.parallel">, !transform.op<"scf.forall">


@run
@create_sequence
def testTileScalable(target):
    structured.TileUsingForOp(
        target,
        sizes=[4, [2]],
    )
    # CHECK-LABEL: TEST: testTileScalable
    # CHECK: transform.sequence
    # CHECK: %{{.+}}, %{{.+}}:2 = transform.structured.tile_using_for %{{.*}}[4, [2]]


@run
@create_sequence
def testTileToForallCompact(target):
    matmul = transform.CastOp(transform.OperationType.get("linalg.matmul"), target)
    structured.TileUsingForallOp(matmul, num_threads=[2, 3, 4])
    # CHECK-LABEL: TEST: testTileToForallCompact
    # CHECK: = transform.structured.tile_using_forall
    # CHECK-SAME: num_threads [2, 3, 4]
    # CHECK-SAME: (!transform.op<"linalg.matmul">) -> (!transform.any_op, !transform.any_op)


@run
@create_sequence
def testTileToForallLoopsAndTileOpTypes(target):
    structured.TileUsingForallOp(
        transform.OperationType.get("scf.forall"),  # loops_type
        transform.OperationType.get("linalg.matmul"),  # tiled_op_type
        target,
        num_threads=[2, 3, 4],
    )
    # CHECK-LABEL: TEST: testTileToForallLoopsAndTileOpTypes
    # CHECK: = transform.structured.tile_using_forall
    # CHECK-SAME: num_threads [2, 3, 4]
    # CHECK-SAME: (!transform.any_op) -> (!transform.op<"scf.forall">, !transform.op<"linalg.matmul">)


@run
@create_sequence
def testTileToForallTileSizes(target):
    structured.TileUsingForallOp(target, tile_sizes=[2, 3, 4])
    # CHECK-LABEL: TEST: testTileToForallTileSizes
    # CHECK: = transform.structured.tile_using_forall
    # CHECK-SAME: tile_sizes [2, 3, 4]


@run
@create_sequence
def testTileToForallMixedDynamic(target):
    n = structured.MatchOp.match_op_names(target, ["test.dummy"])
    structured.TileUsingForallOp(target, num_threads=[n, 3, 4])
    # CHECK-LABEL: TEST: testTileToForallMixedDynamic
    # CHECK: = transform.structured.tile_using_forall
    # CHECK-SAME: num_threads [%{{.*}}, 3, 4] : (!transform.any_op, !transform.any_op)


@run
@create_sequence
def testTileToForallPackedDynamic(target):
    n = structured.MatchOp.match_op_names(target, ["test.dummy"])
    structured.TileUsingForallOp(target, num_threads=n)
    # CHECK-LABEL: TEST: testTileToForallPackedDynamic
    # CHECK: = transform.structured.tile_using_forall
    # CHECK-SAME: num_threads *(%0) : (!transform.any_op, !transform.any_op)


@run
@create_sequence
def testTileToForallMapping(target):
    mapping = Attribute.parse("[ #gpu.thread<y>, #gpu.thread<x> ]")
    structured.TileUsingForallOp(target, num_threads=[2, 3], mapping=mapping)
    # CHECK-LABEL: TEST: testTileToForallMapping
    # CHECK: = transform.structured.tile_using_forall
    # CHECK-SAME: mapping = [#gpu.thread<y>, #gpu.thread<x>]


@run
@create_sequence
def testVectorizeChildrenAndApplyPatternsAllAttrs(target):
    structured.VectorizeChildrenAndApplyPatternsOp(
        target,
        disable_multi_reduction_to_contract_patterns=True,
        disable_transfer_permutation_map_lowering_patterns=True,
        vectorize_nd_extract=True,
        vectorize_padding=True,
    )
    # CHECK-LABEL: TEST: testVectorizeChildrenAndApplyPatternsAllAttrs
    # CHECK: transform.sequence
    # CHECK: = transform.structured.vectorize
    # CHECK-SAME: disable_multi_reduction_to_contract_patterns
    # CHECK-SAME: disable_transfer_permutation_map_lowering_patterns
    # CHECK-SAME: vectorize_nd_extract
    # CHECK-SAME: vectorize_padding


@run
@create_sequence
def testVectorizeChildrenAndApplyPatternsNoAttrs(target):
    structured.VectorizeChildrenAndApplyPatternsOp(
        target,
        disable_multi_reduction_to_contract_patterns=False,
        disable_transfer_permutation_map_lowering_patterns=False,
        vectorize_nd_extract=False,
        vectorize_padding=False,
    )
    # CHECK-LABEL: TEST: testVectorizeChildrenAndApplyPatternsNoAttrs
    # CHECK: transform.sequence
    # CHECK: = transform.structured.vectorize
    # CHECK-NOT: disable_multi_reduction_to_contract_patterns
    # CHECK-NOT: disable_transfer_permutation_map_lowering_patterns
    # CHECK-NOT: vectorize_nd_extract
    # CHECK-NOT: vectorize_padding


@run
@create_sequence
def testMatchInterfaceEnum(target):
    names = ArrayAttr.get([StringAttr.get("test.dummy")])
    result_type = transform.AnyOpType.get()
    fused = structured.MatchOp.__base__(
        result_type,
        target,
        ops=names,
        interface=structured.MatchInterfaceEnum.LinalgOp,
    )
    # CHECK-LABEL: TEST: testMatchInterfaceEnum
    # CHECK: transform.sequence
    # CHECK: = transform.structured.match
    # CHECK: interface{LinalgOp}


@run
@create_sequence
def testMatchInterfaceEnumReplaceAttributeBuilder(target):
    @register_attribute_builder("MatchInterfaceEnum", replace=True)
    def match_interface_enum(x, context):
        if x == "LinalgOp":
            y = 0
        elif x == "TilingInterface":
            y = 1
        return IntegerAttr.get(IntegerType.get_signless(32, context=context), y)

    names = ArrayAttr.get([StringAttr.get("test.dummy")])
    result_type = transform.AnyOpType.get()
    fused = structured.MatchOp.__base__(
        result_type,
        target,
        ops=names,
        interface="TilingInterface",
    )
    # CHECK-LABEL: TEST: testMatchInterfaceEnumReplaceAttributeBuilder
    # CHECK: transform.sequence
    # CHECK: = transform.structured.match
    # CHECK: interface{TilingInterface}
