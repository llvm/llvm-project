// DEFINE: %{option} = "enable-runtime-library=false s2s-strategy=2"
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler=%{option}
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_c_runner_utils | \
// DEFINE: FileCheck %s
//
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false s2s-strategy=2 vl=2 reassociate-fp-reductions=true enable-index-optimizations=true"
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation and, if available, VLA
// vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false vl=4 enable-arm-sve=%ENABLE_VLA"
// REDEFINE: %{run} = %lli_host_or_aarch64_cmd \
// REDEFINE:   --entry-function=entry_lli \
// REDEFINE:   --extra-module=%S/Inputs/main_for_lli.ll \
// REDEFINE:   %VLA_ARCH_ATTR_OPTIONS \
// REDEFINE:   --dlopen=%mlir_native_utils_lib_dir/libmlir_c_runner_utils%shlibext | \
// REDEFINE: FileCheck %s
// RUN: %{compile} | mlir-translate -mlir-to-llvmir | %{run}

#Tensor1 = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed-nu", "singleton-nu", "singleton" ]
}>

#Tensor2 = #sparse_tensor.encoding<{
  lvlTypes = [ "dense", "compressed", "dense" ]
}>

#Tensor3 = #sparse_tensor.encoding<{
  lvlTypes = [ "dense", "dense", "compressed" ],
  dimToLvl = affine_map<(i,j,k) -> (i,k,j)>
}>

module {
  //
  // Utility for output.
  //
  func.func @dump(%arg0: tensor<2x3x4xf32>) {
    %c0 = arith.constant 0 : index
    %d0 = arith.constant -1.0 : f32
    %0 = vector.transfer_read %arg0[%c0, %c0, %c0], %d0: tensor<2x3x4xf32>, vector<2x3x4xf32>
    vector.print %0 : vector<2x3x4xf32>
    return
  }

  //
  // The first test suite (for non-singleton DimLevelTypes).
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
    %t1 = sparse_tensor.convert %s1 : tensor<2x3x4xf64, #Tensor1> to tensor<2x3x4xf32, #Tensor1>
    %t2 = sparse_tensor.convert %s2 : tensor<2x3x4xf64, #Tensor2> to tensor<2x3x4xf32, #Tensor2>
    %t3 = sparse_tensor.convert %s3 : tensor<2x3x4xf64, #Tensor3> to tensor<2x3x4xf32, #Tensor3>

    //
    // Convert sparse tensor back to dense.
    //
    %d1 = sparse_tensor.convert %t1 : tensor<2x3x4xf32, #Tensor1> to tensor<2x3x4xf32>
    %d2 = sparse_tensor.convert %t2 : tensor<2x3x4xf32, #Tensor2> to tensor<2x3x4xf32>
    %d3 = sparse_tensor.convert %t3 : tensor<2x3x4xf32, #Tensor3> to tensor<2x3x4xf32>

    //
    // Check round-trip equality.  And release dense tensors.
    //
    // CHECK-COUNT-3: ( ( ( 1, 2, 3, 4 ), ( 5, 6, 7, 8 ), ( 9, 10, 11, 12 ) ), ( ( 13, 14, 15, 16 ), ( 17, 18, 19, 20 ), ( 21, 22, 23, 24 ) ) )
    call @dump(%d1) : (tensor<2x3x4xf32>) -> ()
    call @dump(%d2) : (tensor<2x3x4xf32>) -> ()
    call @dump(%d3) : (tensor<2x3x4xf32>) -> ()

    //
    // Release sparse tensors.
    //
    bufferization.dealloc_tensor %t1 : tensor<2x3x4xf32, #Tensor1>
    bufferization.dealloc_tensor %t2 : tensor<2x3x4xf32, #Tensor2>
    bufferization.dealloc_tensor %t3 : tensor<2x3x4xf32, #Tensor3>
    bufferization.dealloc_tensor %s1 : tensor<2x3x4xf64, #Tensor1>
    bufferization.dealloc_tensor %s2 : tensor<2x3x4xf64, #Tensor2>
    bufferization.dealloc_tensor %s3 : tensor<2x3x4xf64, #Tensor3>

    return
  }
}
