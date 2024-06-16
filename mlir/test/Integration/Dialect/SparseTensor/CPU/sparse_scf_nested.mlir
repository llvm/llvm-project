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
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false vl=4 
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with  VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#SparseMatrix = #sparse_tensor.encoding<{ map = (d0, d1, d2) -> (d0 : compressed, d1 : compressed, d2 : compressed) }>

module @func_sparse.2 {
  // Do elementwise x+1 when true, x-1 when false
  func.func public @condition(%cond: i1, %arg0: tensor<2x3x4xf64, #SparseMatrix>) -> tensor<2x3x4xf64, #SparseMatrix> {
    %1 = scf.if %cond -> (tensor<2x3x4xf64, #SparseMatrix>) {
      %cst_2 = arith.constant dense<1.000000e+00> : tensor<f64>
      %cst_3 = arith.constant dense<1.000000e+00> : tensor<2x3x4xf64>
      %2 = tensor.empty() : tensor<2x3x4xf64, #SparseMatrix>
      %3 = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0, %cst_3 : tensor<2x3x4xf64, #SparseMatrix>, tensor<2x3x4xf64>)
        outs(%2 : tensor<2x3x4xf64, #SparseMatrix>) {
          ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
            %4 = arith.subf %arg1, %arg2 : f64
            linalg.yield %4 : f64
          } -> tensor<2x3x4xf64, #SparseMatrix>
        scf.yield %3 : tensor<2x3x4xf64, #SparseMatrix>
    } else {
      %cst_2 = arith.constant dense<1.000000e+00> : tensor<f64>
      %cst_3 = arith.constant dense<1.000000e+00> : tensor<2x3x4xf64>
      %2 = tensor.empty() : tensor<2x3x4xf64, #SparseMatrix>
      %3 = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0, %cst_3 : tensor<2x3x4xf64, #SparseMatrix>, tensor<2x3x4xf64>)
        outs(%2 : tensor<2x3x4xf64, #SparseMatrix>) {
          ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
            %4 = arith.addf %arg1, %arg2 : f64
            linalg.yield %4 : f64
          } -> tensor<2x3x4xf64, #SparseMatrix>
        scf.yield %3 : tensor<2x3x4xf64, #SparseMatrix>
    }
    return %1 : tensor<2x3x4xf64, #SparseMatrix>
  }

  func.func public @main() {
    %src = arith.constant dense<[
     [  [  1.0,  2.0,  3.0,  4.0 ],
        [  5.0,  6.0,  7.0,  8.0 ],
        [  9.0, 10.0, 11.0, 12.0 ] ],
     [  [ 13.0, 14.0, 15.0, 16.0 ],
        [ 17.0, 18.0, 19.0, 20.0 ],
        [ 21.0, 22.0, 23.0, 24.0 ] ]
    ]> : tensor<2x3x4xf64>

    %t = arith.constant 1 : i1
    %f = arith.constant 0 : i1

    %sm = sparse_tensor.convert %src : tensor<2x3x4xf64> to tensor<2x3x4xf64, #SparseMatrix>

    %sm_t = call @condition(%t, %sm) : (i1, tensor<2x3x4xf64, #SparseMatrix>) -> tensor<2x3x4xf64, #SparseMatrix>
    %sm_f = call @condition(%f, %sm) : (i1, tensor<2x3x4xf64, #SparseMatrix>) -> tensor<2x3x4xf64, #SparseMatrix>

    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 24
    // CHECK-NEXT: dim = ( 2, 3, 4 )
    // CHECK-NEXT: lvl = ( 2, 3, 4 )
    // CHECK-NEXT: pos[0] : ( 0, 2 )
    // CHECK-NEXT: crd[0] : ( 0, 1 )
    // CHECK-NEXT: pos[1] : ( 0, 3, 6 )
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 0, 1, 2 )
    // CHECK-NEXT: pos[2] : ( 0, 4, 8, 12, 16, 20, 24 )
    // CHECK-NEXT: crd[2] : ( 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3 )
    // CHECK-NEXT: values : ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 )
    // CHECK-NEXT: ----
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 24
    // CHECK-NEXT: dim = ( 2, 3, 4 )
    // CHECK-NEXT: lvl = ( 2, 3, 4 )
    // CHECK-NEXT: pos[0] : ( 0, 2 )
    // CHECK-NEXT: crd[0] : ( 0, 1 )
    // CHECK-NEXT: pos[1] : ( 0, 3, 6 )
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 0, 1, 2 )
    // CHECK-NEXT: pos[2] : ( 0, 4, 8, 12, 16, 20, 24 )
    // CHECK-NEXT: crd[2] : ( 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3 )
    // CHECK-NEXT: values : ( 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %sm_t : tensor<2x3x4xf64, #SparseMatrix>
    sparse_tensor.print %sm_f : tensor<2x3x4xf64, #SparseMatrix>

    bufferization.dealloc_tensor %sm : tensor<2x3x4xf64, #SparseMatrix>
    bufferization.dealloc_tensor %sm_t : tensor<2x3x4xf64, #SparseMatrix>
    bufferization.dealloc_tensor %sm_f : tensor<2x3x4xf64, #SparseMatrix>
    return
  }
}
