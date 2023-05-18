//--------------------------------------------------------------------------------------------------
// WHEN CREATING A NEW TEST, PLEASE JUST COPY & PASTE WITHOUT EDITS.
//
// Set-up that's shared across all tests in this directory. In principle, this
// config could be moved to lit.local.cfg. However, there are downstream users that
//  do not use these LIT config files. Hence why this is kept inline.
//
// DEFINE: %{sparsifier_opts} = enable-runtime-library=true
// DEFINE: %{sparsifier_opts_sve} = enable-arm-sve=true %{sparsifier_opts}
// DEFINE: %{compile} = mlir-opt %s --sparsifier="%{sparsifier_opts}"
// DEFINE: %{compile_sve} = mlir-opt %s --sparsifier="%{sparsifier_opts_sve}"
// DEFINE: %{run_libs} = -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils
// DEFINE: %{run_opts} = -e main -entry-point-result=void
// DEFINE: %{run} = mlir-cpu-runner %{run_opts} %{run_libs}
// DEFINE: %{run_sve} = %mcr_aarch64_cmd --march=aarch64 --mattr="+sve" %{run_opts} %{run_libs}
//
// DEFINE: %{env} =
//--------------------------------------------------------------------------------------------------

// REDEFINE: %{sparsifier_opts} = enable-runtime-library=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#Tensor1 = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : dense, d1 : dense, d2 : compressed)

}>

// NOTE: dense after compressed is not currently supported for the target
// of direct-sparse2sparse conversion.  (It's fine for the source though.)
#Tensor2 = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : dense, d1 : compressed, d2 : dense)

}>

#Tensor3 = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : dense, d2 : dense, d1 : compressed)

}>

#SingletonTensor1 = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : dense, d1 : compressed(nonunique), d2 : singleton)

}>

// This also checks the singleton->compressed conversion.
#SingletonTensor3 = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : dense, d1 : dense, d2 : compressed)

}>

module {
  //
  // Utility for output.
  //
  func.func @dump(%arg0: tensor<2x3x4xf64>) {
    %c0 = arith.constant 0 : index
    %d0 = arith.constant -1.0 : f64
    %0 = vector.transfer_read %arg0[%c0, %c0, %c0], %d0: tensor<2x3x4xf64>, vector<2x3x4xf64>
    vector.print %0 : vector<2x3x4xf64>
    return
  }

  //
  // The first test suite (for non-singleton LevelTypes).
  //
  func.func @testNonSingleton() {
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
    %s3 = sparse_tensor.convert %src : tensor<2x3x4xf64> to tensor<2x3x4xf64, #Tensor3>

    //
    // Convert sparse tensor directly to another sparse format.
    //
    %t13 = sparse_tensor.convert %s1 : tensor<2x3x4xf64, #Tensor1> to tensor<2x3x4xf64, #Tensor3>
    %t31 = sparse_tensor.convert %s3 : tensor<2x3x4xf64, #Tensor3> to tensor<2x3x4xf64, #Tensor1>

    //
    // Convert sparse tensor back to dense.
    //
    %d13 = sparse_tensor.convert %t13 : tensor<2x3x4xf64, #Tensor3> to tensor<2x3x4xf64>
    %d31 = sparse_tensor.convert %t31 : tensor<2x3x4xf64, #Tensor1> to tensor<2x3x4xf64>

    //
    // Check round-trip equality.  And release dense tensors.
    //
    // CHECK-COUNT-3: ( ( ( 1, 2, 3, 4 ), ( 5, 6, 7, 8 ), ( 9, 10, 11, 12 ) ), ( ( 13, 14, 15, 16 ), ( 17, 18, 19, 20 ), ( 21, 22, 23, 24 ) ) )
    call @dump(%src) : (tensor<2x3x4xf64>) -> ()
    call @dump(%d13) : (tensor<2x3x4xf64>) -> ()
    call @dump(%d31) : (tensor<2x3x4xf64>) -> ()

    //
    // Release the resources.
    //
    bufferization.dealloc_tensor %t13 : tensor<2x3x4xf64, #Tensor3>
    bufferization.dealloc_tensor %t31 : tensor<2x3x4xf64, #Tensor1>
    bufferization.dealloc_tensor %s1 : tensor<2x3x4xf64, #Tensor1>
    bufferization.dealloc_tensor %s3 : tensor<2x3x4xf64, #Tensor3>
    bufferization.dealloc_tensor %d13 : tensor<2x3x4xf64>
    bufferization.dealloc_tensor %d31 : tensor<2x3x4xf64>

    return
  }

  //
  // The second test suite (for singleton LevelTypes).
  //
  func.func @testSingleton() {
    //
    // Initialize a 3-dim dense tensor with the 3rd dim being singleton.
    //
    %src = arith.constant dense<[
       [  [  1.0,  0.0,  0.0,  0.0 ],
          [  0.0,  6.0,  0.0,  0.0 ],
          [  0.0,  0.0, 11.0,  0.0 ] ],
       [  [  0.0, 14.0,  0.0,  0.0 ],
          [  0.0,  0.0,  0.0, 20.0 ],
          [ 21.0,  0.0,  0.0,  0.0 ] ]
    ]> : tensor<2x3x4xf64>

    //
    // Convert dense tensor directly to various sparse tensors.
    //
    %s1 = sparse_tensor.convert %src : tensor<2x3x4xf64> to tensor<2x3x4xf64, #SingletonTensor1>
    %s3 = sparse_tensor.convert %src : tensor<2x3x4xf64> to tensor<2x3x4xf64, #SingletonTensor3>

    //
    // Convert sparse tensor directly to another sparse format.
    //
    %t13 = sparse_tensor.convert %s1 : tensor<2x3x4xf64, #SingletonTensor1> to tensor<2x3x4xf64, #SingletonTensor3>
    %t31 = sparse_tensor.convert %s3 : tensor<2x3x4xf64, #SingletonTensor3> to tensor<2x3x4xf64, #SingletonTensor1>

    //
    // Convert sparse tensor back to dense.
    //
    %d13 = sparse_tensor.convert %t13 : tensor<2x3x4xf64, #SingletonTensor3> to tensor<2x3x4xf64>
    %d31 = sparse_tensor.convert %t31 : tensor<2x3x4xf64, #SingletonTensor1> to tensor<2x3x4xf64>

    //
    // Check round-trip equality.  And release dense tensors.
    //
    // CHECK-COUNT-3: ( ( ( 1, 0, 0, 0 ), ( 0, 6, 0, 0 ), ( 0, 0, 11, 0 ) ), ( ( 0, 14, 0, 0 ), ( 0, 0, 0, 20 ), ( 21, 0, 0, 0 ) ) )
    call @dump(%src) : (tensor<2x3x4xf64>) -> ()
    call @dump(%d13) : (tensor<2x3x4xf64>) -> ()
    call @dump(%d31) : (tensor<2x3x4xf64>) -> ()

    //
    // Release the resources.
    //
    bufferization.dealloc_tensor %t13 : tensor<2x3x4xf64, #SingletonTensor3>
    bufferization.dealloc_tensor %t31 : tensor<2x3x4xf64, #SingletonTensor1>
    bufferization.dealloc_tensor %s1 : tensor<2x3x4xf64, #SingletonTensor1>
    bufferization.dealloc_tensor %s3 : tensor<2x3x4xf64, #SingletonTensor3>
    bufferization.dealloc_tensor %d13 : tensor<2x3x4xf64>
    bufferization.dealloc_tensor %d31 : tensor<2x3x4xf64>

    return
  }

  //
  // Main driver.
  //
  func.func @main() {
    call @testNonSingleton() : () -> ()
    call @testSingleton() : () -> ()
    return
  }
}
