# RUN: %PYTHON %s | FileCheck %s

import mlir.dialects.arith as arith
import mlir.dialects.memref as memref
import mlir.extras.types as T
from mlir.dialects.memref import _infer_memref_subview_result_type
from mlir.ir import *


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: testSubViewAccessors
@run
def testSubViewAccessors():
    ctx = Context()
    module = Module.parse(
        r"""
    func.func @f1(%arg0: memref<?x?xf32>) {
      %0 = arith.constant 0 : index
      %1 = arith.constant 1 : index
      %2 = arith.constant 2 : index
      %3 = arith.constant 3 : index
      %4 = arith.constant 4 : index
      %5 = arith.constant 5 : index
      memref.subview %arg0[%0, %1][%2, %3][%4, %5] : memref<?x?xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      return
    }
  """,
        ctx,
    )
    func_body = module.body.operations[0].regions[0].blocks[0]
    subview = func_body.operations[6]

    assert subview.source == subview.operands[0]
    assert len(subview.offsets) == 2
    assert len(subview.sizes) == 2
    assert len(subview.strides) == 2
    assert subview.result == subview.results[0]

    # CHECK: SubViewOp
    print(type(subview).__name__)

    # CHECK: constant 0
    print(subview.offsets[0])
    # CHECK: constant 1
    print(subview.offsets[1])
    # CHECK: constant 2
    print(subview.sizes[0])
    # CHECK: constant 3
    print(subview.sizes[1])
    # CHECK: constant 4
    print(subview.strides[0])
    # CHECK: constant 5
    print(subview.strides[1])


# CHECK-LABEL: TEST: testCustomBuidlers
@run
def testCustomBuidlers():
    with Context() as ctx, Location.unknown(ctx):
        module = Module.parse(
            r"""
      func.func @f1(%arg0: memref<?x?xf32>, %arg1: index, %arg2: index) {
        return
      }
    """
        )
        f = module.body.operations[0]
        func_body = f.regions[0].blocks[0]
        with InsertionPoint.at_block_terminator(func_body):
            memref.LoadOp(f.arguments[0], f.arguments[1:])

        # CHECK: func @f1(%[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
        # CHECK: memref.load %[[ARG0]][%[[ARG1]], %[[ARG2]]]
        print(module)
        assert module.operation.verify()


# CHECK-LABEL: TEST: testMemRefAttr
@run
def testMemRefAttr():
    with Context() as ctx, Location.unknown(ctx):
        module = Module.create()
        with InsertionPoint(module.body):
            memref.global_("objFifo_in0", T.memref(16, T.i32()))
        # CHECK: memref.global @objFifo_in0 : memref<16xi32>
        print(module)


# CHECK-LABEL: TEST: testSubViewOpInferReturnTypeSemantics
@run
def testSubViewOpInferReturnTypeSemantics():
    with Context() as ctx, Location.unknown(ctx):
        module = Module.create()
        with InsertionPoint(module.body):
            x = memref.alloc(T.memref(10, 10, T.i32()), [], [])
            # CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<10x10xi32>
            print(x.owner)

            y = memref.subview(x, [1, 1], [3, 3], [1, 1])
            assert y.owner.verify()
            # CHECK: %{{.*}} = memref.subview %[[ALLOC]][1, 1] [3, 3] [1, 1] : memref<10x10xi32> to memref<3x3xi32, strided<[10, 1], offset: 11>>
            print(y.owner)

            z = memref.subview(
                x,
                [arith.constant(T.index(), 1), 1],
                [3, 3],
                [1, 1],
            )
            # CHECK: %{{.*}} =  memref.subview %[[ALLOC]][1, 1] [3, 3] [1, 1] : memref<10x10xi32> to memref<3x3xi32, strided<[10, 1], offset: 11>>
            print(z.owner)

            z = memref.subview(
                x,
                [arith.constant(T.index(), 3), arith.constant(T.index(), 4)],
                [3, 3],
                [1, 1],
            )
            # CHECK: %{{.*}} =  memref.subview %[[ALLOC]][3, 4] [3, 3] [1, 1] : memref<10x10xi32> to memref<3x3xi32, strided<[10, 1], offset: 34>>
            print(z.owner)

            s = arith.addi(arith.constant(T.index(), 3), arith.constant(T.index(), 4))
            z = memref.subview(
                x,
                [s, 0],
                [3, 3],
                [1, 1],
            )
            # CHECK: {{.*}} = memref.subview %[[ALLOC]][%0, 0] [3, 3] [1, 1] : memref<10x10xi32> to memref<3x3xi32, strided<[10, 1], offset: ?>>
            print(z)

            try:
                _infer_memref_subview_result_type(
                    x.type,
                    [arith.constant(T.index(), 3), arith.constant(T.index(), 4)],
                    [ShapedType.get_dynamic_size(), 3],
                    [1, 1],
                )
            except ValueError as e:
                # CHECK: Only inferring from python or mlir integer constant is supported
                print(e)

            try:
                memref.subview(
                    x,
                    [arith.constant(T.index(), 3), arith.constant(T.index(), 4)],
                    [ShapedType.get_dynamic_size(), 3],
                    [1, 1],
                )
            except ValueError as e:
                # CHECK: mixed static/dynamic offset/sizes/strides requires explicit result type
                print(e)

            layout = StridedLayoutAttr.get(ShapedType.get_dynamic_size(), [10, 1])
            x = memref.alloc(
                T.memref(
                    10,
                    10,
                    T.i32(),
                    layout=layout,
                ),
                [],
                [arith.constant(T.index(), 42)],
            )
            # CHECK: %[[DYNAMICALLOC:.*]] = memref.alloc()[%c42] : memref<10x10xi32, strided<[10, 1], offset: ?>>
            print(x.owner)
            y = memref.subview(
                x,
                [1, 1],
                [3, 3],
                [1, 1],
                result_type=T.memref(3, 3, T.i32(), layout=layout),
            )
            # CHECK: %{{.*}} = memref.subview %[[DYNAMICALLOC]][1, 1] [3, 3] [1, 1] : memref<10x10xi32, strided<[10, 1], offset: ?>> to memref<3x3xi32, strided<[10, 1], offset: ?>>
            print(y.owner)


# CHECK-LABEL: TEST: testSubViewOpInferReturnTypeExtensiveSlicing
@run
def testSubViewOpInferReturnTypeExtensiveSlicing():
    def check_strides_offset(memref, np_view):
        layout = memref.type.layout
        dtype_size_in_bytes = np_view.dtype.itemsize
        golden_strides = (np.array(np_view.strides) // dtype_size_in_bytes).tolist()
        golden_offset = (
            np_view.ctypes.data - np_view.base.ctypes.data
        ) // dtype_size_in_bytes

        assert (layout.strides, layout.offset) == (golden_strides, golden_offset)

    with Context() as ctx, Location.unknown(ctx):
        module = Module.create()
        with InsertionPoint(module.body):
            shape = (10, 22, 3, 44)
            golden_mem = np.zeros(shape, dtype=np.int32)
            mem1 = memref.alloc(T.memref(*shape, T.i32()), [], [])

            # fmt: off
            check_strides_offset(memref.subview(mem1, (1, 0, 0, 0), (1, 22, 3, 44), (1, 1, 1, 1)), golden_mem[1:2, ...])
            check_strides_offset(memref.subview(mem1, (0, 1, 0, 0), (10, 1, 3, 44), (1, 1, 1, 1)), golden_mem[:, 1:2])
            check_strides_offset(memref.subview(mem1, (0, 0, 1, 0), (10, 22, 1, 44), (1, 1, 1, 1)), golden_mem[:, :, 1:2])
            check_strides_offset(memref.subview(mem1, (0, 0, 0, 1), (10, 22, 3, 1), (1, 1, 1, 1)), golden_mem[:, :, :, 1:2])
            check_strides_offset(memref.subview(mem1, (0, 1, 0, 1), (10, 1, 3, 1), (1, 1, 1, 1)), golden_mem[:, 1:2, :, 1:2])
            check_strides_offset(memref.subview(mem1, (1, 0, 0, 1), (1, 22, 3, 1), (1, 1, 1, 1)), golden_mem[1:2, :, :, 1:2])
            check_strides_offset(memref.subview(mem1, (1, 1, 0, 0), (1, 1, 3, 44), (1, 1, 1, 1)), golden_mem[1:2, 1:2, :, :])
            check_strides_offset(memref.subview(mem1, (0, 0, 1, 1), (10, 22, 1, 1), (1, 1, 1, 1)), golden_mem[:, :, 1:2, 1:2])
            check_strides_offset(memref.subview(mem1, (0, 1, 1, 0), (10, 1, 1, 44), (1, 1, 1, 1)), golden_mem[:, 1:2, 1:2, :])
            check_strides_offset(memref.subview(mem1, (1, 0, 1, 0), (1, 22, 1, 44), (1, 1, 1, 1)), golden_mem[1:2, :, 1:2, :])
            check_strides_offset(memref.subview(mem1, (1, 1, 0, 1), (1, 1, 3, 1), (1, 1, 1, 1)), golden_mem[1:2, 1:2, :, 1:2])
            check_strides_offset(memref.subview(mem1, (1, 0, 1, 1), (1, 22, 1, 1), (1, 1, 1, 1)), golden_mem[1:2, :, 1:2, 1:2])
            check_strides_offset(memref.subview(mem1, (0, 1, 1, 1), (10, 1, 1, 1), (1, 1, 1, 1)), golden_mem[:, 1:2, 1:2, 1:2])
            check_strides_offset(memref.subview(mem1, (1, 1, 1, 0), (1, 1, 1, 44), (1, 1, 1, 1)), golden_mem[1:2, 1:2, 1:2, :])
            # fmt: on

            # default strides and offset means no stridedlayout attribute means affinemap layout
            assert memref.subview(
                mem1, (0, 0, 0, 0), (10, 22, 3, 44), (1, 1, 1, 1)
            ).type.layout == AffineMapAttr.get(
                AffineMap.get(
                    4,
                    0,
                    [
                        AffineDimExpr.get(0),
                        AffineDimExpr.get(1),
                        AffineDimExpr.get(2),
                        AffineDimExpr.get(3),
                    ],
                )
            )

            shape = (7, 22, 30, 44)
            golden_mem = np.zeros(shape, dtype=np.int32)
            mem2 = memref.alloc(T.memref(*shape, T.i32()), [], [])
            # fmt: off
            check_strides_offset(memref.subview(mem2, (0, 0, 0, 0), (7, 11, 3, 44), (1, 2, 1, 1)), golden_mem[:, 0:22:2])
            check_strides_offset(memref.subview(mem2, (0, 0, 0, 0), (7, 11, 11, 44), (1, 2, 30, 1)), golden_mem[:, 0:22:2, 0:330:30])
            check_strides_offset(memref.subview(mem2, (0, 0, 0, 0), (7, 11, 11, 11), (1, 2, 30, 400)), golden_mem[:, 0:22:2, 0:330:30, 0:4400:400])
            # fmt: on

            shape = (8, 8)
            golden_mem = np.zeros(shape, dtype=np.int32)
            # fmt: off
            mem3 = memref.alloc(T.memref(*shape, T.i32()), [], [])
            check_strides_offset(memref.subview(mem3, (0, 0), (4, 4), (1, 1)), golden_mem[0:4, 0:4])
            check_strides_offset(memref.subview(mem3, (4, 4), (4, 4), (1, 1)), golden_mem[4:8, 4:8])
            # fmt: on
