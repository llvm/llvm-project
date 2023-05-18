# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
import mlir.dialects.builtin as builtin
import mlir.dialects.func as func
import mlir.dialects.vector as vector


def run(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        f()
    return f


# CHECK-LABEL: TEST: testPrintOp
@run
def testPrintOp():
    module = Module.create()
    with InsertionPoint(module.body):

        @func.FuncOp.from_py_func(VectorType.get((12, 5), F32Type.get()))
        def print_vector(arg):
            return vector.PrintOp(source=arg)

    # CHECK-LABEL: func @print_vector(
    # CHECK-SAME:                     %[[ARG:.*]]: vector<12x5xf32>) {
    #       CHECK:   vector.print %[[ARG]] : vector<12x5xf32>
    #       CHECK:   return
    #       CHECK: }
    print(module)


# CHECK-LABEL: TEST: testTransferReadOp
@run
def testTransferReadOp():
    module = Module.create()
    with InsertionPoint(module.body):
        vector_type = VectorType.get([2, 3], F32Type.get())
        memref_type = MemRefType.get(
            [ShapedType.get_dynamic_size(), ShapedType.get_dynamic_size()],
            F32Type.get(),
        )
        index_type = IndexType.get()
        mask_type = VectorType.get(vector_type.shape, IntegerType.get_signless(1))
        identity_map = AffineMap.get_identity(vector_type.rank)
        identity_map_attr = AffineMapAttr.get(identity_map)
        f = func.FuncOp(
            "transfer_read", ([memref_type, index_type, F32Type.get(), mask_type], [])
        )
        with InsertionPoint(f.add_entry_block()):
            A, zero, padding, mask = f.arguments
            vector.TransferReadOp(
                vector_type, A, [zero, zero], identity_map_attr, padding, mask=mask
            )
            vector.TransferReadOp(
                vector_type, A, [zero, zero], identity_map_attr, padding
            )
            func.ReturnOp([])

    # CHECK: @transfer_read(%[[MEM:.*]]: memref<?x?xf32>, %[[IDX:.*]]: index,
    # CHECK: %[[PAD:.*]]: f32, %[[MASK:.*]]: vector<2x3xi1>)
    # CHECK: vector.transfer_read %[[MEM]][%[[IDX]], %[[IDX]]], %[[PAD]], %[[MASK]]
    # CHECK: vector.transfer_read %[[MEM]][%[[IDX]], %[[IDX]]], %[[PAD]]
    # CHECK-NOT: %[[MASK]]
    print(module)


# CHECK-LABEL: TEST: testBitEnumCombiningKind
@run
def testBitEnumCombiningKind():
    module = Module.create()
    with InsertionPoint(module.body):
        f32 = F32Type.get()
        vector_type = VectorType.get([16], f32)

        @func.FuncOp.from_py_func(vector_type)
        def reduction(arg):
            v = vector.ReductionOp(f32, vector.CombiningKind.ADD, arg)
            return v

    # CHECK: func.func @reduction(%[[VEC:.*]]: vector<16xf32>) -> f32 {
    # CHECK: %0 = vector.reduction <add>, %[[VEC]] : vector<16xf32> into f32
    print(module)
