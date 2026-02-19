# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
import mlir.dialects.builtin as builtin
import mlir.dialects.func as func
import mlir.dialects.x86vector as x86vector


def run(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        f()
    return f


# CHECK-LABEL: TEST: testAvxOp
@run
def testAvxOp():
    module = Module.create()
    with InsertionPoint(module.body):

        @func.FuncOp.from_py_func(MemRefType.get((1,), BF16Type.get()))
        def avx_op(arg):
            return x86vector.BcstToPackedF32Op(
                a=arg, dst=VectorType.get((8,), F32Type.get())
            )

    # CHECK-LABEL: func @avx_op(
    # CHECK-SAME:      %[[ARG:.+]]: memref<1xbf16>) -> vector<8xf32> {
    #       CHECK:   %[[VAL:.+]] = x86vector.avx.bcst_to_f32.packed %[[ARG]]
    #       CHECK:   return %[[VAL]] : vector<8xf32>
    #       CHECK: }
    print(module)


# CHECK-LABEL: TEST: testAvx512Op
@run
def testAvx512Op():
    module = Module.create()
    with InsertionPoint(module.body):

        @func.FuncOp.from_py_func(VectorType.get((8,), F32Type.get()))
        def avx512_op(arg):
            return x86vector.CvtPackedF32ToBF16Op(
                a=arg, dst=VectorType.get((8,), BF16Type.get())
            )

    # CHECK-LABEL: func @avx512_op(
    # CHECK-SAME:      %[[ARG:.+]]: vector<8xf32>) -> vector<8xbf16> {
    #       CHECK:   %[[VAL:.+]] = x86vector.avx512.cvt.packed.f32_to_bf16 %[[ARG]]
    #       CHECK:   return %[[VAL]] : vector<8xbf16>
    #       CHECK: }
    print(module)


# CHECK-LABEL: TEST: testAvx10Op
@run
def testAvx10Op():
    module = Module.create()
    with InsertionPoint(module.body):

        @func.FuncOp.from_py_func(
            VectorType.get((16,), IntegerType.get(32)),
            VectorType.get((64,), IntegerType.get(8)),
            VectorType.get((64,), IntegerType.get(8)),
        )
        def avx10_op(*args):
            return x86vector.AVX10DotInt8Op(w=args[0], a=args[1], b=args[2])

    # CHECK-LABEL: func @avx10_op(
    # CHECK-SAME:      %[[W:.+]]: vector<16xi32>, %[[A:.+]]: vector<64xi8>,
    # CHECK-SAME:      %[[B:.+]]: vector<64xi8>) -> vector<16xi32> {
    #       CHECK:   %[[VAL:.+]] = x86vector.avx10.dot.i8 %[[W]], %[[A]], %[[B]]
    #       CHECK:   return %[[VAL]] : vector<16xi32>
    #       CHECK: }
    print(module)
