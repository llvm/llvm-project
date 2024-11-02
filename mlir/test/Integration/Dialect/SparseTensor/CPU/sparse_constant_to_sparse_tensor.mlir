// RUN: mlir-opt %s --sparse-compiler | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#Tensor1  = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed"]
}>

//
// Integration tests for conversions from sparse constants to sparse tensors.
//
module {
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %d0 = arith.constant 0.0 : f64

    // A tensor in COO format.
    %ti = arith.constant sparse<[[0, 0], [0, 7], [1, 2], [4, 2], [5, 3], [6, 4], [6, 6], [9, 7]],
                          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]> : tensor<10x8xf64>

    // Convert the tensor in COO format to a sparse tensor with annotation #Tensor1.
    %ts = sparse_tensor.convert %ti : tensor<10x8xf64> to tensor<10x8xf64, #Tensor1>

    // CHECK: ( 0, 1, 4, 5, 6, 9 )
    %i0 = sparse_tensor.indices %ts { dimension = 0 : index } : tensor<10x8xf64, #Tensor1> to memref<?xindex>
    %i0r = vector.transfer_read %i0[%c0], %c0: memref<?xindex>, vector<6xindex>
    vector.print %i0r : vector<6xindex>

    // CHECK: ( 0, 7, 2, 2, 3, 4, 6, 7 )
    %i1 = sparse_tensor.indices %ts { dimension = 1 : index } : tensor<10x8xf64, #Tensor1> to memref<?xindex>
    %i1r = vector.transfer_read %i1[%c0], %c0: memref<?xindex>, vector<8xindex>
    vector.print %i1r : vector<8xindex>

    // CHECK: ( 1, 2, 3, 4, 5, 6, 7, 8 )
    %v = sparse_tensor.values %ts : tensor<10x8xf64, #Tensor1> to memref<?xf64>
    %vr = vector.transfer_read %v[%c0], %d0: memref<?xf64>, vector<8xf64>
    vector.print %vr : vector<8xf64>

    // Release the resources.
    bufferization.dealloc_tensor %ts : tensor<10x8xf64, #Tensor1>

    return
  }
}

