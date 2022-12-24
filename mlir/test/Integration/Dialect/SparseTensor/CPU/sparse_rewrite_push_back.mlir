// DEFINE: %{option} = enable-runtime-library=false
// DEFINE: %{command} = mlir-opt %s --sparse-compiler=%{option} | \
// DEFINE: mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// DEFINE: FileCheck %s
//
// RUN: %{command}
//
// Do the same run, but now with vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true"
// RUN: %{command}

module {
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %d0 = arith.constant 0.0 : f32
    %d1 = arith.constant 1.0 : f32
    %d2 = arith.constant 2.0 : f32

    %bufferSizes = memref.alloc(%c1) : memref<?xindex>
    %buffer = memref.alloc(%c1) : memref<?xf32>

    memref.store %c0, %bufferSizes[%c0] : memref<?xindex>
    %buffer2, %s0 = sparse_tensor.push_back %c0, %buffer, %d2 : index, memref<?xf32>, f32
    %buffer3, %s1 = sparse_tensor.push_back %s0, %buffer2, %d1, %c10 : index, memref<?xf32>, f32, index

    // CHECK: 16
    %capacity = memref.dim %buffer3, %c0 : memref<?xf32>
    vector.print %capacity : index

    // CHECK: 11
    vector.print %s1 : index

    // CHECK (  2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 )
    %values = vector.transfer_read %buffer3[%c0], %d0: memref<?xf32>, vector<11xf32>
    vector.print %values : vector<11xf32>

    // Release the buffers.
    memref.dealloc %bufferSizes : memref<?xindex>
    memref.dealloc %buffer3 : memref<?xf32>
    return
  }
}

