// RUN: mlir-opt %s \
// RUN:  -sparsification -sparse-tensor-conversion \
// RUN:  -linalg-bufferize -convert-linalg-to-loops \
// RUN:  -convert-vector-to-scf -convert-scf-to-std \
// RUN:  -func-bufferize -tensor-constant-bufferize -tensor-bufferize \
// RUN:  -std-bufferize -finalizing-bufferize \
// RUN:  -convert-vector-to-llvm -convert-memref-to-llvm -convert-std-to-llvm \
// RUN:  -reconcile-unrealized-casts \
// RUN:  | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext \
// RUN:  | \
// RUN: FileCheck %s

#Tensor1  = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j,k) -> (i,j,k)>
}>

#Tensor2  = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j,k) -> (j,k,i)>
}>

#Tensor3  = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j,k) -> (k,i,j)>
}>

#Tensor4  = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j,k) -> (i,j,k)>
}>

#Tensor5  = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j,k) -> (j,k,i)>
}>

#Tensor6  = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j,k) -> (k,i,j)>
}>

//
// Integration test that tests conversions from sparse to dense tensors.
//
module {
  //
  // Output utilities.
  //
  func @dumpf64(%arg0: tensor<2x3x4xf64>) {
    %c0 = arith.constant 0 : index
    %d0 = arith.constant -1.0 : f64
    %0 = vector.transfer_read %arg0[%c0, %c0, %c0], %d0: tensor<2x3x4xf64>, vector<2x3x4xf64>
    vector.print %0 : vector<2x3x4xf64>
    return
  }

  //
  // Main driver.
  //
  func @entry() {
    //
    // Initialize a 3-dim dense tensor.
    //
    %t = arith.constant dense<[
       [  [  1.0,  2.0,  3.0,  4.0 ],
          [  5.0,  6.0,  7.0,  8.0 ],
          [  9.0, 10.0, 11.0, 12.0 ] ],
       [  [ 13.0, 14.0, 15.0, 16.0 ],
          [ 17.0, 18.0, 19.0, 20.0 ],
          [ 21.0, 22.0, 23.0, 24.0 ] ]
    ]> : tensor<2x3x4xf64>

    //
    // Convert dense tensor directly to various sparse tensors.
    //    tensor1: stored as 2x3x4
    //    tensor2: stored as 3x4x2
    //    tensor3: stored as 4x2x3
    //
    %1 = sparse_tensor.convert %t : tensor<2x3x4xf64> to tensor<2x3x4xf64, #Tensor1>
    %2 = sparse_tensor.convert %t : tensor<2x3x4xf64> to tensor<2x3x4xf64, #Tensor2>
    %3 = sparse_tensor.convert %t : tensor<2x3x4xf64> to tensor<2x3x4xf64, #Tensor3>
    %4 = sparse_tensor.convert %t : tensor<2x3x4xf64> to tensor<2x3x4xf64, #Tensor4>
    %5 = sparse_tensor.convert %t : tensor<2x3x4xf64> to tensor<2x3x4xf64, #Tensor5>
    %6 = sparse_tensor.convert %t : tensor<2x3x4xf64> to tensor<2x3x4xf64, #Tensor6>

    //
    // Convert sparse tensor back to dense.
    //
    %a = sparse_tensor.convert %1 : tensor<2x3x4xf64, #Tensor1> to tensor<2x3x4xf64>
    %b = sparse_tensor.convert %2 : tensor<2x3x4xf64, #Tensor2> to tensor<2x3x4xf64>
    %c = sparse_tensor.convert %3 : tensor<2x3x4xf64, #Tensor3> to tensor<2x3x4xf64>
    %d = sparse_tensor.convert %4 : tensor<2x3x4xf64, #Tensor4> to tensor<2x3x4xf64>
    %e = sparse_tensor.convert %5 : tensor<2x3x4xf64, #Tensor5> to tensor<2x3x4xf64>
    %f = sparse_tensor.convert %6 : tensor<2x3x4xf64, #Tensor6> to tensor<2x3x4xf64>

    //
    // Check round-trip equality.
    //
    // CHECK:      ( ( ( 1, 2, 3, 4 ), ( 5, 6, 7, 8 ), ( 9, 10, 11, 12 ) ), ( ( 13, 14, 15, 16 ), ( 17, 18, 19, 20 ), ( 21, 22, 23, 24 ) ) )
    // CHECK-NEXT: ( ( ( 1, 2, 3, 4 ), ( 5, 6, 7, 8 ), ( 9, 10, 11, 12 ) ), ( ( 13, 14, 15, 16 ), ( 17, 18, 19, 20 ), ( 21, 22, 23, 24 ) ) )
    // CHECK-NEXT: ( ( ( 1, 2, 3, 4 ), ( 5, 6, 7, 8 ), ( 9, 10, 11, 12 ) ), ( ( 13, 14, 15, 16 ), ( 17, 18, 19, 20 ), ( 21, 22, 23, 24 ) ) )
    // CHECK-NEXT: ( ( ( 1, 2, 3, 4 ), ( 5, 6, 7, 8 ), ( 9, 10, 11, 12 ) ), ( ( 13, 14, 15, 16 ), ( 17, 18, 19, 20 ), ( 21, 22, 23, 24 ) ) )
    // CHECK-NEXT: ( ( ( 1, 2, 3, 4 ), ( 5, 6, 7, 8 ), ( 9, 10, 11, 12 ) ), ( ( 13, 14, 15, 16 ), ( 17, 18, 19, 20 ), ( 21, 22, 23, 24 ) ) )
    // CHECK-NEXT: ( ( ( 1, 2, 3, 4 ), ( 5, 6, 7, 8 ), ( 9, 10, 11, 12 ) ), ( ( 13, 14, 15, 16 ), ( 17, 18, 19, 20 ), ( 21, 22, 23, 24 ) ) )
    // CHECK-NEXT: ( ( ( 1, 2, 3, 4 ), ( 5, 6, 7, 8 ), ( 9, 10, 11, 12 ) ), ( ( 13, 14, 15, 16 ), ( 17, 18, 19, 20 ), ( 21, 22, 23, 24 ) ) )
    call @dumpf64(%t) : (tensor<2x3x4xf64>) -> ()
    call @dumpf64(%a) : (tensor<2x3x4xf64>) -> ()
    call @dumpf64(%b) : (tensor<2x3x4xf64>) -> ()
    call @dumpf64(%c) : (tensor<2x3x4xf64>) -> ()
    call @dumpf64(%d) : (tensor<2x3x4xf64>) -> ()
    call @dumpf64(%e) : (tensor<2x3x4xf64>) -> ()
    call @dumpf64(%f) : (tensor<2x3x4xf64>) -> ()

    //
    // Release the resources.
    //
    sparse_tensor.release %1 : tensor<2x3x4xf64, #Tensor1>
    sparse_tensor.release %2 : tensor<2x3x4xf64, #Tensor2>
    sparse_tensor.release %3 : tensor<2x3x4xf64, #Tensor3>
    sparse_tensor.release %4 : tensor<2x3x4xf64, #Tensor4>
    sparse_tensor.release %5 : tensor<2x3x4xf64, #Tensor5>
    sparse_tensor.release %6 : tensor<2x3x4xf64, #Tensor6>

    %ma = memref.buffer_cast %a : memref<2x3x4xf64>
    %mb = memref.buffer_cast %b : memref<2x3x4xf64>
    %mc = memref.buffer_cast %c : memref<2x3x4xf64>
    %md = memref.buffer_cast %d : memref<2x3x4xf64>
    %me = memref.buffer_cast %e : memref<2x3x4xf64>
    %mf = memref.buffer_cast %f : memref<2x3x4xf64>

    memref.dealloc %ma : memref<2x3x4xf64>
    memref.dealloc %mb : memref<2x3x4xf64>
    memref.dealloc %mc : memref<2x3x4xf64>
    memref.dealloc %md : memref<2x3x4xf64>
    memref.dealloc %me : memref<2x3x4xf64>
    memref.dealloc %mf : memref<2x3x4xf64>

    return
  }
}
