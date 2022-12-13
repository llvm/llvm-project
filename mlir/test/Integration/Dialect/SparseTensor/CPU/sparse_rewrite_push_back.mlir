// RUN: mlir-opt %s --sparse-compiler=enable-runtime-library=false | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

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
    %buffer2 = sparse_tensor.push_back %bufferSizes, %buffer, %d2 {idx=0 : index} : memref<?xindex>, memref<?xf32>, f32
    %buffer3 = sparse_tensor.push_back %bufferSizes, %buffer2, %d1, %c10 {idx=0 : index} : memref<?xindex>, memref<?xf32>, f32, index

    // CHECK: 16
    %capacity = memref.dim %buffer3, %c0 : memref<?xf32>
    vector.print %capacity : index

    // CHECK: ( 11 )
    %size = vector.transfer_read %bufferSizes[%c0], %c0: memref<?xindex>, vector<1xindex>
    vector.print %size : vector<1xindex>

    // CHECK (  2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 )
    %values = vector.transfer_read %buffer3[%c0], %d0: memref<?xf32>, vector<11xf32>
    vector.print %values : vector<11xf32>

    // Release the buffers.
    memref.dealloc %bufferSizes : memref<?xindex>
    memref.dealloc %buffer3 : memref<?xf32>
    return
  }
}

