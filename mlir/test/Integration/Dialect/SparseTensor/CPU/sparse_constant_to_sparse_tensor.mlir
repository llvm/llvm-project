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

// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false enable-buffer-initialization=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false enable-buffer-initialization=true vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#Tensor1  = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed, d1 : compressed)
}>

//
// Integration tests for conversions from sparse constants to sparse tensors.
//
module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %d0 = arith.constant 0.0 : f64

    // A tensor in COO format.
    %ti = arith.constant sparse<[[0, 0], [0, 7], [1, 2], [4, 2], [5, 3], [6, 4], [6, 6], [9, 7]],
                          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]> : tensor<10x8xf64>

    // Convert the tensor in COO format to a sparse tensor with annotation #Tensor1.
    %ts = sparse_tensor.convert %ti : tensor<10x8xf64> to tensor<10x8xf64, #Tensor1>

    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 8
    // CHECK-NEXT: dim = ( 10, 8 )
    // CHECK-NEXT: lvl = ( 10, 8 )
    // CHECK-NEXT: pos[0] : ( 0, 6 )
    // CHECK-NEXT: crd[0] : ( 0, 1, 4, 5, 6, 9 )
    // CHECK-NEXT: pos[1] : ( 0, 2, 3, 4, 5, 7, 8 )
    // CHECK-NEXT: crd[1] : ( 0, 7, 2, 2, 3, 4, 6, 7 )
    // CHECK-NEXT: values : ( 1, 2, 3, 4, 5, 6, 7, 8 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %ts : tensor<10x8xf64, #Tensor1>

    // Release the resources.
    bufferization.dealloc_tensor %ts : tensor<10x8xf64, #Tensor1>

    return
  }
}

