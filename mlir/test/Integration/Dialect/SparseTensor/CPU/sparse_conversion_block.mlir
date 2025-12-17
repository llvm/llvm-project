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
// DEFINE: %{run_libs_sve} = -shared-libs=%native_mlir_runner_utils,%native_mlir_c_runner_utils
// DEFINE: %{run_opts} = -e main -entry-point-result=void
// DEFINE: %{run} = mlir-runner %{run_opts} %{run_libs}
// DEFINE: %{run_sve} = %mcr_aarch64_cmd --march=aarch64 --mattr="+sve" %{run_opts} %{run_libs_sve}
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

#CSR  = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

#CSC  = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : dense, d0 : compressed)
}>

#BSR = #sparse_tensor.encoding<{
   map = ( i, j ) ->
      ( i floordiv 2 : dense,
        j floordiv 2 : compressed,
        i mod 2      : dense,
        j mod 2      : dense
      )
}>


//
// Integration test that tests conversions between sparse tensors.
//
module {
  //
  // Main driver.
  //
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    //
    // Initialize a 2-dim dense tensor.
    //
    %t = arith.constant sparse<[[0, 0], [0, 1], [0, 2], [0, 3],
                                [1, 0], [1, 1], [1, 2], [1, 3]],
                                [ 1.0, 2.0, 3.0, 4.0,
                                  5.0, 6.0, 7.0, 8.0 ]> : tensor<2x4xf64>

    %td = arith.constant dense<[[ 1.0, 2.0, 3.0, 4.0 ],
                                [ 5.0, 6.0, 7.0, 8.0 ]]> : tensor<2x4xf64>

    // constant -> BSR (either from SparseElementAttibutes or DenseElementAttribute)
    %1 = sparse_tensor.convert %t : tensor<2x4xf64> to tensor<2x4xf64, #BSR>
    %2 = sparse_tensor.convert %td : tensor<2x4xf64> to tensor<2x4xf64, #BSR>
    %3 = sparse_tensor.convert %1 : tensor<2x4xf64, #BSR> to tensor<2x4xf64, #CSR>
    %4 = sparse_tensor.convert %1 : tensor<2x4xf64, #BSR> to tensor<2x4xf64, #CSC>

    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 8
    // CHECK-NEXT: dim = ( 2, 4 )
    // CHECK-NEXT: lvl = ( 1, 2, 2, 2 )
    // CHECK-NEXT: pos[1] : ( 0, 2 )
    // CHECK-NEXT: crd[1] : ( 0, 1 )
    // CHECK-NEXT: values : ( 1, 2, 5, 6, 3, 4, 7, 8 )
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 8
    // CHECK-NEXT: dim = ( 2, 4 )
    // CHECK-NEXT: lvl = ( 1, 2, 2, 2 )
    // CHECK-NEXT: pos[1] : ( 0, 2 )
    // CHECK-NEXT: crd[1] : ( 0, 1 )
    // CHECK-NEXT: values : ( 1, 2, 5, 6, 3, 4, 7, 8 )
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 8
    // CHECK-NEXT: dim = ( 2, 4 )
    // CHECK-NEXT: lvl = ( 2, 4 )
    // CHECK-NEXT: pos[1] : ( 0, 4, 8 )
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 3, 0, 1, 2, 3 )
    // CHECK-NEXT: values : ( 1, 2, 3, 4, 5, 6, 7, 8 )
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 8
    // CHECK-NEXT: dim = ( 2, 4 )
    // CHECK-NEXT: lvl = ( 4, 2 )
    // CHECK-NEXT: pos[1] : ( 0, 2, 4, 6, 8 )
    // CHECK-NEXT: crd[1] : ( 0, 1, 0, 1, 0, 1, 0, 1 )
    // CHECK-NEXT: values : ( 1, 5, 2, 6, 3, 7, 4, 8 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %1 : tensor<2x4xf64, #BSR>
    sparse_tensor.print %2 : tensor<2x4xf64, #BSR>
    sparse_tensor.print %3 : tensor<2x4xf64, #CSR>
    sparse_tensor.print %4 : tensor<2x4xf64, #CSC>

    // TODO: Fix memory leaks.
    bufferization.dealloc_tensor %1 : tensor<2x4xf64, #BSR>
    bufferization.dealloc_tensor %2 : tensor<2x4xf64, #BSR>
    bufferization.dealloc_tensor %3 : tensor<2x4xf64, #CSR>
    bufferization.dealloc_tensor %4 : tensor<2x4xf64, #CSC>

    return
  }
}
