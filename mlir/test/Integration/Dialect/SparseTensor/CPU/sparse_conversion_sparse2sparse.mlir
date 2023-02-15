// Force this file to use the kDirect method for sparse2sparse.
// DEFINE: %{option} = "enable-runtime-library=true s2s-strategy=2"
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler=%{option}
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// DEFINE: FileCheck %s
//
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{option} = "enable-runtime-library=false s2s-strategy=2"
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false s2s-strategy=2 vl=2 reassociate-fp-reductions=true enable-index-optimizations=true"
// RUN: %{compile} | %{run}

// Do the same run, but now with direct IR generation and, if available, VLA
// vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false vl=4 enable-arm-sve=%ENABLE_VLA"
// REDEFINE: %{run} = %lli \
// REDEFINE:   --entry-function=entry_lli \
// REDEFINE:   --extra-module=%S/Inputs/main_for_lli.ll \
// REDEFINE:   %VLA_ARCH_ATTR_OPTIONS \
// REDEFINE:   --dlopen=%mlir_native_utils_lib_dir/libmlir_c_runner_utils%shlibext | \
// REDEFINE: FileCheck %s
// RUN: %{compile} | mlir-translate -mlir-to-llvmir | %{run}

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

#SingletonTensor1 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed", "singleton" ]
}>

// This also checks the compressed->dense conversion (when there are zeros).
#SingletonTensor2 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "dense", "singleton" ]
}>

// This also checks the singleton->compressed conversion.
#SingletonTensor3 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "dense", "compressed" ]
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
  // The first test suite (for non-singleton DimLevelTypes).
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
    call @dump(%d13) : (tensor<2x3x4xf64>) -> ()
    call @dump(%d21) : (tensor<2x3x4xf64>) -> ()
    call @dump(%d23) : (tensor<2x3x4xf64>) -> ()
    call @dump(%d31) : (tensor<2x3x4xf64>) -> ()

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

  //
  // The second test suite (for singleton DimLevelTypes).
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
    %s2 = sparse_tensor.convert %src : tensor<2x3x4xf64> to tensor<2x3x4xf64, #SingletonTensor2>
    %s3 = sparse_tensor.convert %src : tensor<2x3x4xf64> to tensor<2x3x4xf64, #SingletonTensor3>

    //
    // Convert sparse tensor directly to another sparse format.
    //
    %t12 = sparse_tensor.convert %s1 : tensor<2x3x4xf64, #SingletonTensor1> to tensor<2x3x4xf64, #SingletonTensor2>
    %t13 = sparse_tensor.convert %s1 : tensor<2x3x4xf64, #SingletonTensor1> to tensor<2x3x4xf64, #SingletonTensor3>
    %t21 = sparse_tensor.convert %s2 : tensor<2x3x4xf64, #SingletonTensor2> to tensor<2x3x4xf64, #SingletonTensor1>
    %t23 = sparse_tensor.convert %s2 : tensor<2x3x4xf64, #SingletonTensor2> to tensor<2x3x4xf64, #SingletonTensor3>
    %t31 = sparse_tensor.convert %s3 : tensor<2x3x4xf64, #SingletonTensor3> to tensor<2x3x4xf64, #SingletonTensor1>
    %t32 = sparse_tensor.convert %s3 : tensor<2x3x4xf64, #SingletonTensor3> to tensor<2x3x4xf64, #SingletonTensor2>

    //
    // Convert sparse tensor back to dense.
    //
    %d12 = sparse_tensor.convert %t12 : tensor<2x3x4xf64, #SingletonTensor2> to tensor<2x3x4xf64>
    %d13 = sparse_tensor.convert %t13 : tensor<2x3x4xf64, #SingletonTensor3> to tensor<2x3x4xf64>
    %d21 = sparse_tensor.convert %t21 : tensor<2x3x4xf64, #SingletonTensor1> to tensor<2x3x4xf64>
    %d23 = sparse_tensor.convert %t23 : tensor<2x3x4xf64, #SingletonTensor3> to tensor<2x3x4xf64>
    %d31 = sparse_tensor.convert %t31 : tensor<2x3x4xf64, #SingletonTensor1> to tensor<2x3x4xf64>
    %d32 = sparse_tensor.convert %t32 : tensor<2x3x4xf64, #SingletonTensor2> to tensor<2x3x4xf64>

    //
    // Check round-trip equality.  And release dense tensors.
    //
    // CHECK-COUNT-7: ( ( ( 1, 0, 0, 0 ), ( 0, 6, 0, 0 ), ( 0, 0, 11, 0 ) ), ( ( 0, 14, 0, 0 ), ( 0, 0, 0, 20 ), ( 21, 0, 0, 0 ) ) )
    call @dump(%src) : (tensor<2x3x4xf64>) -> ()
    call @dump(%d12) : (tensor<2x3x4xf64>) -> ()
    call @dump(%d13) : (tensor<2x3x4xf64>) -> ()
    call @dump(%d21) : (tensor<2x3x4xf64>) -> ()
    call @dump(%d23) : (tensor<2x3x4xf64>) -> ()
    call @dump(%d31) : (tensor<2x3x4xf64>) -> ()
    call @dump(%d32) : (tensor<2x3x4xf64>) -> ()

    //
    // Release sparse tensors.
    //
    bufferization.dealloc_tensor %t12 : tensor<2x3x4xf64, #SingletonTensor2>
    bufferization.dealloc_tensor %t13 : tensor<2x3x4xf64, #SingletonTensor3>
    bufferization.dealloc_tensor %t21 : tensor<2x3x4xf64, #SingletonTensor1>
    bufferization.dealloc_tensor %t23 : tensor<2x3x4xf64, #SingletonTensor3>
    bufferization.dealloc_tensor %t31 : tensor<2x3x4xf64, #SingletonTensor1>
    bufferization.dealloc_tensor %t32 : tensor<2x3x4xf64, #SingletonTensor2>
    bufferization.dealloc_tensor %s1 : tensor<2x3x4xf64, #SingletonTensor1>
    bufferization.dealloc_tensor %s2 : tensor<2x3x4xf64, #SingletonTensor2>
    bufferization.dealloc_tensor %s3 : tensor<2x3x4xf64, #SingletonTensor3>

    return
  }

  //
  // Main driver.
  //
  func.func @entry() {
    call @testNonSingleton() : () -> ()
    call @testSingleton() : () -> ()
    return
  }
}
