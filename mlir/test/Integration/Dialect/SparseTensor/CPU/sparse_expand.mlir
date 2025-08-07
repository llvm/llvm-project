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
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#CSC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : dense, d0 : compressed)
}>

module {
  //
  // Column-wise storage forces the ijk loop to permute into jki
  // so that access pattern expansion (workspace) needs to be
  // done along dimension with size 8.
  //
  func.func @matmul(%A: tensor<8x2xf64, #CSC>,
                    %B: tensor<2x4xf64, #CSC>) -> tensor<8x4xf64, #CSC> {
    %C = tensor.empty() : tensor<8x4xf64, #CSC>
    %D = linalg.matmul
      ins(%A, %B: tensor<8x2xf64, #CSC>, tensor<2x4xf64, #CSC>)
         outs(%C: tensor<8x4xf64, #CSC>) -> tensor<8x4xf64, #CSC>
    return %D: tensor<8x4xf64, #CSC>
  }

  //
  // Main driver.
  //
  func.func @main() {
    %c0 = arith.constant 0 : index
    %d1 = arith.constant -1.0 : f64

    // Initialize various dense matrices for stress testing.
    %da = arith.constant dense<[
        [ 1.1, 2.1 ],
        [ 1.2, 2.2 ],
        [ 1.3, 2.3 ],
        [ 1.4, 2.4 ],
        [ 1.5, 2.5 ],
        [ 1.6, 2.6 ],
        [ 1.7, 2.7 ],
        [ 1.8, 2.8 ]
    ]> : tensor<8x2xf64>
    %db = arith.constant dense<[
        [ 10.1, 11.1, 12.1, 13.1 ],
        [ 10.2, 11.2, 12.2, 13.2 ]
    ]> : tensor<2x4xf64>

    // Convert all these matrices to sparse format.
    %x1 = sparse_tensor.convert %da : tensor<8x2xf64> to tensor<8x2xf64, #CSC>
    %x2 = sparse_tensor.convert %db : tensor<2x4xf64> to tensor<2x4xf64, #CSC>

    // Call kernels with dense.
    %x3 = call @matmul(%x1, %x2)
       : (tensor<8x2xf64, #CSC>,
          tensor<2x4xf64, #CSC>) -> tensor<8x4xf64, #CSC>

    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 32
    // CHECK-NEXT: dim = ( 8, 4 )
    // CHECK-NEXT: lvl = ( 4, 8 )
    // CHECK-NEXT: pos[1] : ( 0, 8, 16, 24, 32 )
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0,
    // CHECK-SAME:            1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7 )
    // CHECK-NEXT: values : ( 32.53, 34.56, 36.59, 38.62, 40.65, 42.68, 44.71, 46.74,
    // CHECK-SAME:            35.73, 37.96, 40.19, 42.42, 44.65, 46.88, 49.11, 51.34,
    // CHECK-SAME:            38.93, 41.36, 43.79, 46.22, 48.65, 51.08, 53.51, 55.94,
    // CHECK-SAME:            42.13, 44.76, 47.39, 50.02, 52.65, 55.28, 57.91, 60.54 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %x3 : tensor<8x4xf64, #CSC>

    // Release the resources.
    bufferization.dealloc_tensor %x1 : tensor<8x2xf64, #CSC>
    bufferization.dealloc_tensor %x2 : tensor<2x4xf64, #CSC>
    bufferization.dealloc_tensor %x3 : tensor<8x4xf64, #CSC>

    return
  }
}
