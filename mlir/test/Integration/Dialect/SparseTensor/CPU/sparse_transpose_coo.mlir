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
// DEFINE: %{run} = mlir-cpu-runner %{run_opts} %{run_libs}
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
// Do the same run, but now with vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with  VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#SortedCOO = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton(soa))
}>

module {

  // A linalg representation of some higher "transpose" op.
  func.func @transpose_coo(%arga: tensor<10x5xf32, #SortedCOO>)
                               -> tensor<5x10xf32, #SortedCOO> {
    %0 = tensor.empty() : tensor<5x10xf32, #SortedCOO>
    %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arga : tensor<10x5xf32, #SortedCOO>)
      outs(%0 : tensor<5x10xf32, #SortedCOO>) {
        ^bb0(%in: f32, %out: f32):
           linalg.yield %in : f32
      } -> tensor<5x10xf32, #SortedCOO>
    return %1 : tensor<5x10xf32, #SortedCOO>
  }

  func.func @main() {
    %f0  = arith.constant 0.0 : f32
    %c0  = arith.constant 0   : index
    %c1  = arith.constant 1   : index

    %A = arith.constant dense<
        [ [ 10.0, 20.0, 30.0, 40.0, 50.0 ],
          [ 11.0, 21.0, 31.0, 41.0, 51.0 ],
          [ 12.0, 22.0, 32.0, 42.0, 52.0 ],
          [ 13.0, 23.0, 33.0, 43.0, 53.0 ],
          [ 14.0, 24.0, 34.0, 44.0, 54.0 ],
          [ 15.0, 25.0, 35.0, 45.0, 55.0 ],
          [ 16.0, 26.0, 36.0, 46.0, 56.0 ],
          [ 17.0, 27.0, 37.0, 47.0, 57.0 ],
          [ 18.0, 28.0, 38.0, 48.0, 58.0 ],
          [ 19.0, 29.0, 39.0, 49.0, 59.0 ] ]
    > : tensor<10x5xf32>

    // Stress test with a "sparse" version of A.
    %SA = sparse_tensor.convert %A
      : tensor<10x5xf32> to tensor<10x5xf32, #SortedCOO>
    %SAT = call @transpose_coo(%SA) : (tensor<10x5xf32, #SortedCOO>)
                                    -> tensor<5x10xf32, #SortedCOO>

    //
    // Verify original and transposed sorted COO.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 50
    // CHECK-NEXT: dim = ( 10, 5 )
    // CHECK-NEXT: lvl = ( 10, 5 )
    // CHECK-NEXT: pos[0] : ( 0, 50 )
    // CHECK-NEXT: crd[0] : ( 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9 )
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4 )
    // CHECK-NEXT: values : ( 10, 20, 30, 40, 50, 11, 21, 31, 41, 51, 12, 22, 32, 42, 52, 13, 23, 33, 43, 53, 14, 24, 34, 44, 54, 15, 25, 35, 45, 55, 16, 26, 36, 46, 56, 17, 27, 37, 47, 57, 18, 28, 38, 48, 58, 19, 29, 39, 49, 59 )
    // CHECK-NEXT: ----
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 50
    // CHECK-NEXT: dim = ( 5, 10 )
    // CHECK-NEXT: lvl = ( 5, 10 )
    // CHECK-NEXT: pos[0] : ( 0, 50 )
    // CHECK-NEXT: crd[0] : ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4 )
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 )
    // CHECK-NEXT: values : ( 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %SA : tensor<10x5xf32, #SortedCOO>
    sparse_tensor.print %SAT : tensor<5x10xf32, #SortedCOO>

    // Release resources.
    bufferization.dealloc_tensor %SA  : tensor<10x5xf32, #SortedCOO>
    bufferization.dealloc_tensor %SAT : tensor<5x10xf32, #SortedCOO>

    return
  }
}
