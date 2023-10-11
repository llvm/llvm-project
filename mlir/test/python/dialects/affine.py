# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
import mlir.dialects.func as func
import mlir.dialects.arith as arith
import mlir.dialects.affine as affine
import mlir.dialects.memref as memref


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: testAffineStoreOp
@run
def testAffineStoreOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f32 = F32Type.get()
            index_type = IndexType.get()
            memref_type_out = MemRefType.get([12, 12], f32)

            # CHECK: func.func @affine_store_test(%[[ARG0:.*]]: index) -> memref<12x12xf32> {
            @func.FuncOp.from_py_func(index_type)
            def affine_store_test(arg0):
                # CHECK: %[[O_VAR:.*]] = memref.alloc() : memref<12x12xf32>
                mem = memref.AllocOp(memref_type_out, [], []).result

                d0 = AffineDimExpr.get(0)
                s0 = AffineSymbolExpr.get(0)
                map = AffineMap.get(1, 1, [s0 * 3, d0 + s0 + 1])

                # CHECK: %[[A1:.*]] = arith.constant 2.100000e+00 : f32
                a1 = arith.ConstantOp(f32, 2.1)

                # CHECK: affine.store %[[A1]], %alloc[symbol(%[[ARG0]]) * 3, %[[ARG0]] + symbol(%[[ARG0]]) + 1] : memref<12x12xf32>
                affine.AffineStoreOp(a1, mem, map, map_operands=[arg0, arg0])

                return mem

        print(module)
