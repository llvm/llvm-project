// Force this file to use the kDirect method for sparse2sparse.
// RUN: mlir-opt %s --sparse-compiler="s2s-strategy=2" | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#Tensor1 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "dense", "compressed" ]
}>

// NOTE: dense after compressed is not currently supported for the target
// of direct-sparse2sparse conversion.  (It's fine for the source though.)
#Tensor2 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed", "dense" ]
}>

#Tensor3 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "dense", "compressed" ],
  dimOrdering = affine_map<(i,j,k) -> (i,k,j)>
}>

module {
  //
  // Utilities for output and releasing memory.
  //
  func.func @dump(%arg0: tensor<2x3x4xf64>) {
    %c0 = arith.constant 0 : index
    %d0 = arith.constant -1.0 : f64
    %0 = vector.transfer_read %arg0[%c0, %c0, %c0], %d0: tensor<2x3x4xf64>, vector<2x3x4xf64>
    vector.print %0 : vector<2x3x4xf64>
    return
  }
  func.func @dumpAndRelease_234(%arg0: tensor<2x3x4xf64>) {
    call @dump(%arg0) : (tensor<2x3x4xf64>) -> ()
    return
  }

  //
  // Main driver.
  //
  func.func @entry() {
    //
    // Initialize a 3-dim dense tensor.
    //
    %src = arith.constant dense<[
       [  [  1.0,  2.0,  3.0,  4.0 ],
          [  5.0,  6.0,  7.0,  8.0 ],
          [  9.0, 10.0, 11.0, 12.0 ] ],
       [  [ 13.0, 14.0, 15.0, 16.0 ],
          [ 17.0, 18.0, 19.0, 20.0 ],
          [ 21.0, 22.0, 23.0, 24.0 ] ]
    ]> : tensor<2x3x4xf64>

    //
    // Convert dense tensor directly to various sparse tensors.
    //
    %s1 = sparse_tensor.convert %src : tensor<2x3x4xf64> to tensor<2x3x4xf64, #Tensor1>
    %s2 = sparse_tensor.convert %src : tensor<2x3x4xf64> to tensor<2x3x4xf64, #Tensor2>
    %s3 = sparse_tensor.convert %src : tensor<2x3x4xf64> to tensor<2x3x4xf64, #Tensor3>

    //
    // Convert sparse tensor directly to another sparse format.
    //
    %t13 = sparse_tensor.convert %s1 : tensor<2x3x4xf64, #Tensor1> to tensor<2x3x4xf64, #Tensor3>
    %t21 = sparse_tensor.convert %s2 : tensor<2x3x4xf64, #Tensor2> to tensor<2x3x4xf64, #Tensor1>
    %t23 = sparse_tensor.convert %s2 : tensor<2x3x4xf64, #Tensor2> to tensor<2x3x4xf64, #Tensor3>
    %t31 = sparse_tensor.convert %s3 : tensor<2x3x4xf64, #Tensor3> to tensor<2x3x4xf64, #Tensor1>

    //
    // Convert sparse tensor back to dense.
    //
    %d13 = sparse_tensor.convert %t13 : tensor<2x3x4xf64, #Tensor3> to tensor<2x3x4xf64>
    %d21 = sparse_tensor.convert %t21 : tensor<2x3x4xf64, #Tensor1> to tensor<2x3x4xf64>
    %d23 = sparse_tensor.convert %t23 : tensor<2x3x4xf64, #Tensor3> to tensor<2x3x4xf64>
    %d31 = sparse_tensor.convert %t31 : tensor<2x3x4xf64, #Tensor1> to tensor<2x3x4xf64>

    //
    // Check round-trip equality.  And release dense tensors.
    //
    // CHECK-COUNT-5: ( ( ( 1, 2, 3, 4 ), ( 5, 6, 7, 8 ), ( 9, 10, 11, 12 ) ), ( ( 13, 14, 15, 16 ), ( 17, 18, 19, 20 ), ( 21, 22, 23, 24 ) ) )
    call @dump(%src) : (tensor<2x3x4xf64>) -> ()
    call @dumpAndRelease_234(%d13) : (tensor<2x3x4xf64>) -> ()
    call @dumpAndRelease_234(%d21) : (tensor<2x3x4xf64>) -> ()
    call @dumpAndRelease_234(%d23) : (tensor<2x3x4xf64>) -> ()
    call @dumpAndRelease_234(%d31) : (tensor<2x3x4xf64>) -> ()

    //
    // Release sparse tensors.
    //
    bufferization.dealloc_tensor %t13 : tensor<2x3x4xf64, #Tensor3>
    bufferization.dealloc_tensor %t21 : tensor<2x3x4xf64, #Tensor1>
    bufferization.dealloc_tensor %t23 : tensor<2x3x4xf64, #Tensor3>
    bufferization.dealloc_tensor %t31 : tensor<2x3x4xf64, #Tensor1>
    bufferization.dealloc_tensor %s1 : tensor<2x3x4xf64, #Tensor1>
    bufferization.dealloc_tensor %s2 : tensor<2x3x4xf64, #Tensor2>
    bufferization.dealloc_tensor %s3 : tensor<2x3x4xf64, #Tensor3>

    return
  }
}
