# RUN: %PYTHON %s | FileCheck %s
# This is just a smoke test that the dialect is functional.

from mlir.ir import *
from mlir.dialects import amdgpu, func


def constructAndPrintInModule(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f()
        print(module)
    return f


# CHECK-LABEL: testSmoke
@constructAndPrintInModule
def testSmoke():
    # CHECK: amdgpu.lds_barrier
    amdgpu.LDSBarrierOp()


# CHECK-LABEL: testFatRawBufferCastOpParams
@constructAndPrintInModule
def testFatRawBufferCastOpParams():
    memref_type = MemRefType.get(
        [ShapedType.get_dynamic_size(), ShapedType.get_dynamic_size()],
        F32Type.get(),
    )
    f = func.FuncOp("test_raw_buffer_cast_params", ([memref_type], []))
    with InsertionPoint(f.add_entry_block()):
        block_args = f.arguments
        amdgpu.FatRawBufferCastOp(block_args[0])
        amdgpu.FatRawBufferCastOp(block_args[0], resetOffset=True)
        amdgpu.FatRawBufferCastOp(block_args[0], boundsCheck=False)
        amdgpu.FatRawBufferCastOp(block_args[0], boundsCheck=False, resetOffset=True)
        func.ReturnOp([])

    # CHECK:     func.func @test_raw_buffer_cast_params(%[[ARG0:.+]]: memref<?x?xf32>) {
    # CHECK:        amdgpu.fat_raw_buffer_cast %[[ARG0]] : memref<?x?xf32> to memref<?x?xf32, #amdgpu.address_space<fat_raw_buffer>>
    # CHECK-NEXT:   amdgpu.fat_raw_buffer_cast %[[ARG0]] resetOffset : memref<?x?xf32> to memref<?x?xf32, #amdgpu.address_space<fat_raw_buffer>>
    # CHECK-NEXT:   amdgpu.fat_raw_buffer_cast %[[ARG0]] boundsCheck(false) : memref<?x?xf32> to memref<?x?xf32, #amdgpu.address_space<fat_raw_buffer>>
    # CHECK-NEXT:   amdgpu.fat_raw_buffer_cast %[[ARG0]] boundsCheck(false) resetOffset : memref<?x?xf32> to memref<?x?xf32, #amdgpu.address_space<fat_raw_buffer>>
