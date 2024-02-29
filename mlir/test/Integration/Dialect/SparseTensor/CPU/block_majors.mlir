//--------------------------------------------------------------------------------------------------
// WHEN CREATING A NEW TEST, PLEASE JUST COPY & PASTE WITHOUT EDITS.
//
// Set-up that's shared across all tests in this directory. In principle, this
// config could be moved to lit.local.cfg. However, there are downstream users that
// do not use these LIT config files. Hence why this is kept inline.
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
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false enable-buffer-initialization=true vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s

#BSR_row_rowmajor = #sparse_tensor.encoding<{
  map = (i, j) ->
    ( i floordiv 3 : dense
    , j floordiv 4 : compressed
    , i mod 3 : dense
    , j mod 4 : dense
    )
}>

#BSR_row_colmajor = #sparse_tensor.encoding<{
  map = (i, j) ->
    ( i floordiv 3 : dense
    , j floordiv 4 : compressed
    , j mod 4 : dense
    , i mod 3 : dense
    )
}>

#BSR_col_rowmajor = #sparse_tensor.encoding<{
  map = (i, j) ->
    ( j floordiv 4 : dense
    , i floordiv 3 : compressed
    , i mod 3 : dense
    , j mod 4 : dense
    )
}>

#BSR_col_colmajor = #sparse_tensor.encoding<{
  map = (i, j) ->
    ( j floordiv 4 : dense
    , i floordiv 3 : compressed
    , j mod 4 : dense
    , i mod 3 : dense
    )
}>

//
// Example 3x4 block storage of a 6x16 matrix:
//
//  +---------+---------+---------+---------+
//  | 1 2 . . | . . . . | . . . . | . . . . |
//  | . . . . | . . . . | . . . . | . . . . |
//  | . . . 3 | . . . . | . . . . | . . . . |
//  +---------+---------+---------+---------+
//  | . . . . | . . . . | 4 5 . . | . . . . |
//  | . . . . | . . . . | . . . . | . . . . |
//  | . . . . | . . . . | . . 6 7 | . . . . |
//  +---------+---------+---------+---------+
//
// Storage for CSR block storage. Note that this essentially
// provides CSR storage of 2x4 blocks with either row-major
// or column-major storage within each 3x4 block of elements.
//
//    positions[1]   : 0 1 2
//    coordinates[1] : 0 2
//    values         : 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3,
//                     4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7 [row-major]
//
//                     1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3,
//                     4, 0, 0, 5, 0, 0, 0, 0, 6, 0, 0, 7 [col-major]
//
// Storage for CSC block storage. Note that this essentially
// provides CSC storage of 4x2 blocks with either row-major
// or column-major storage within each 3x4 block of elements.
//
//    positions[1]   : 0 1 1 2 2
//    coordinates[1] : 0 1
//    values         : 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3,
//                     4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7 [row-major]
//
//                     1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3,
//                     4, 0, 0, 5, 0, 0, 0, 0, 6, 0, 0, 7 [col-major]
//
module {


  //
  // CHECK: ---- Sparse Tensor ----
  // CHECK-NEXT: nse = 24
  // CHECK-NEXT: pos[1] : ( 0, 1, 2,
  // CHECK-NEXT: crd[1] : ( 0, 2,
  // CHECK-NEXT: values : ( 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7,
  // CHECK-NEXT: ----
  //
  func.func @foo1() {
    // Build.
    %c0 = arith.constant 0   : index
    %f0 = arith.constant 0.0 : f64
    %m = arith.constant sparse<
        [ [0, 0], [0, 1], [2, 3], [3, 8], [3, 9], [5, 10], [5, 11] ],
        [ 1., 2., 3., 4., 5., 6., 7.]
    > : tensor<6x16xf64>
    %s1 = sparse_tensor.convert %m : tensor<6x16xf64> to tensor<?x?xf64, #BSR_row_rowmajor>
    // Test.
    sparse_tensor.print %s1 : tensor<?x?xf64, #BSR_row_rowmajor>
    // Release.
    bufferization.dealloc_tensor %s1: tensor<?x?xf64, #BSR_row_rowmajor>
    return
  }

  //
  // CHECK-NEXT: ---- Sparse Tensor ----
  // CHECK-NEXT: nse = 24
  // CHECK-NEXT: pos[1] : ( 0, 1, 2,
  // CHECK-NEXT: crd[1] : ( 0, 2,
  // CHECK-NEXT: values : ( 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 5, 0, 0, 0, 0, 6, 0, 0, 7,
  // CHECK-NEXT: ----
  //
  func.func @foo2() {
    // Build.
    %c0 = arith.constant 0   : index
    %f0 = arith.constant 0.0 : f64
    %m = arith.constant sparse<
        [ [0, 0], [0, 1], [2, 3], [3, 8], [3, 9], [5, 10], [5, 11] ],
        [ 1., 2., 3., 4., 5., 6., 7.]
    > : tensor<6x16xf64>
    %s2 = sparse_tensor.convert %m : tensor<6x16xf64> to tensor<?x?xf64, #BSR_row_colmajor>
    // Test.
    sparse_tensor.print %s2 : tensor<?x?xf64, #BSR_row_colmajor>
    // Release.
    bufferization.dealloc_tensor %s2: tensor<?x?xf64, #BSR_row_colmajor>
    return
  }

  //
  // CHECK-NEXT: ---- Sparse Tensor ----
  // CHECK-NEXT: nse = 24
  // CHECK-NEXT: pos[1] : ( 0, 1, 1, 2, 2,
  // CHECK-NEXT: crd[1] : ( 0, 1,
  // CHECK-NEXT: values : ( 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7,
  // CHECK-NEXT: ----
  //
  func.func @foo3() {
    // Build.
    %c0 = arith.constant 0   : index
    %f0 = arith.constant 0.0 : f64
    %m = arith.constant sparse<
        [ [0, 0], [0, 1], [2, 3], [3, 8], [3, 9], [5, 10], [5, 11] ],
        [ 1., 2., 3., 4., 5., 6., 7.]
    > : tensor<6x16xf64>
    %s3 = sparse_tensor.convert %m : tensor<6x16xf64> to tensor<?x?xf64, #BSR_col_rowmajor>
    // Test.
    sparse_tensor.print %s3 : tensor<?x?xf64, #BSR_col_rowmajor>
    // Release.
    bufferization.dealloc_tensor %s3: tensor<?x?xf64, #BSR_col_rowmajor>
    return
  }

  //
  // CHECK-NEXT: ---- Sparse Tensor ----
  // CHECK-NEXT: nse = 24
  // CHECK-NEXT: pos[1] : ( 0, 1, 1, 2, 2,
  // CHECK-NEXT: crd[1] : ( 0, 1,
  // CHECK-NEXT: values : ( 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 5, 0, 0, 0, 0, 6, 0, 0, 7,
  // CHECK-NEXT: ----
  //
  func.func @foo4() {
    // Build.
    %c0 = arith.constant 0   : index
    %f0 = arith.constant 0.0 : f64
    %m = arith.constant sparse<
        [ [0, 0], [0, 1], [2, 3], [3, 8], [3, 9], [5, 10], [5, 11] ],
        [ 1., 2., 3., 4., 5., 6., 7.]
    > : tensor<6x16xf64>
    %s4 = sparse_tensor.convert %m : tensor<6x16xf64> to tensor<?x?xf64, #BSR_col_colmajor>
    // Test.
    sparse_tensor.print %s4 : tensor<?x?xf64, #BSR_col_colmajor>
    // Release.
    bufferization.dealloc_tensor %s4: tensor<?x?xf64, #BSR_col_colmajor>
    return
  }

  func.func @main() {
    call @foo1() : () -> ()
    call @foo2() : () -> ()
    call @foo3() : () -> ()
    call @foo4() : () -> ()
    return
  }
}
