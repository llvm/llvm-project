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
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}


#SparseVector = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed)
}>

#SparseMatrix = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed, d1 : compressed)
}>

#trait_1d = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "X(i) = a(i) op i"
}

#trait_2d = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(i,j) op i op j"
}

//
// Test with indices. Note that a lot of results are actually
// dense, but this is done to stress test all the operations.
//
module {

  //
  // Kernel that uses index in the index notation (conjunction).
  //
  func.func @sparse_index_1d_conj(%arga: tensor<8xi64, #SparseVector>)
                                 -> tensor<8xi64, #SparseVector> {
    %init = tensor.empty() : tensor<8xi64, #SparseVector>
    %r = linalg.generic #trait_1d
        ins(%arga: tensor<8xi64, #SparseVector>)
       outs(%init: tensor<8xi64, #SparseVector>) {
        ^bb(%a: i64, %x: i64):
          %i = linalg.index 0 : index
          %ii = arith.index_cast %i : index to i64
          %m1 = arith.muli %a, %ii : i64
          linalg.yield %m1 : i64
    } -> tensor<8xi64, #SparseVector>
    return %r : tensor<8xi64, #SparseVector>
  }

  //
  // Kernel that uses index in the index notation (disjunction).
  //
  func.func @sparse_index_1d_disj(%arga: tensor<8xi64, #SparseVector>)
                                 -> tensor<8xi64, #SparseVector> {
    %init = tensor.empty() : tensor<8xi64, #SparseVector>
    %r = linalg.generic #trait_1d
        ins(%arga: tensor<8xi64, #SparseVector>)
       outs(%init: tensor<8xi64, #SparseVector>) {
        ^bb(%a: i64, %x: i64):
          %i = linalg.index 0 : index
          %ii = arith.index_cast %i : index to i64
          %m1 = arith.addi %a, %ii : i64
          linalg.yield %m1 : i64
    } -> tensor<8xi64, #SparseVector>
    return %r : tensor<8xi64, #SparseVector>
  }

  //
  // Kernel that uses indices in the index notation (conjunction).
  //
  func.func @sparse_index_2d_conj(%arga: tensor<3x4xi64, #SparseMatrix>)
                                 -> tensor<3x4xi64, #SparseMatrix> {
    %init = tensor.empty() : tensor<3x4xi64, #SparseMatrix>
    %r = linalg.generic #trait_2d
        ins(%arga: tensor<3x4xi64, #SparseMatrix>)
       outs(%init: tensor<3x4xi64, #SparseMatrix>) {
        ^bb(%a: i64, %x: i64):
          %i = linalg.index 0 : index
          %j = linalg.index 1 : index
          %ii = arith.index_cast %i : index to i64
          %jj = arith.index_cast %j : index to i64
          %m1 = arith.muli %ii, %a : i64
          %m2 = arith.muli %jj, %m1 : i64
          linalg.yield %m2 : i64
    } -> tensor<3x4xi64, #SparseMatrix>
    return %r : tensor<3x4xi64, #SparseMatrix>
  }

  //
  // Kernel that uses indices in the index notation (disjunction).
  //
  func.func @sparse_index_2d_disj(%arga: tensor<3x4xi64, #SparseMatrix>)
                                 -> tensor<3x4xi64, #SparseMatrix> {
    %init = tensor.empty() : tensor<3x4xi64, #SparseMatrix>
    %r = linalg.generic #trait_2d
        ins(%arga: tensor<3x4xi64, #SparseMatrix>)
       outs(%init: tensor<3x4xi64, #SparseMatrix>) {
        ^bb(%a: i64, %x: i64):
          %i = linalg.index 0 : index
          %j = linalg.index 1 : index
          %ii = arith.index_cast %i : index to i64
          %jj = arith.index_cast %j : index to i64
          %m1 = arith.addi %ii, %a : i64
          %m2 = arith.addi %jj, %m1 : i64
          linalg.yield %m2 : i64
    } -> tensor<3x4xi64, #SparseMatrix>
    return %r : tensor<3x4xi64, #SparseMatrix>
  }

  func.func @add_outer_2d(%arg0: tensor<2x3xf32, #SparseMatrix>)
                         -> tensor<2x3xf32, #SparseMatrix> {
    %0 = tensor.empty() : tensor<2x3xf32, #SparseMatrix>
    %1 = linalg.generic #trait_2d
      ins(%arg0 : tensor<2x3xf32, #SparseMatrix>)
      outs(%0 : tensor<2x3xf32, #SparseMatrix>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %2 = linalg.index 0 : index
      %3 = arith.index_cast %2 : index to i64
      %4 = arith.uitofp %3 : i64 to f32
      %5 = arith.addf %arg1, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<2x3xf32, #SparseMatrix>
    return %1 : tensor<2x3xf32, #SparseMatrix>
  }

  //
  // Main driver.
  //
  func.func @main() {
    %c0 = arith.constant 0 : index
    %du = arith.constant -1 : i64
    %df = arith.constant -1.0 : f32

    // Setup input sparse vector.
    %v1 = arith.constant sparse<[[2], [4]], [ 10, 20]> : tensor<8xi64>
    %sv = sparse_tensor.convert %v1 : tensor<8xi64> to tensor<8xi64, #SparseVector>

    // Setup input "sparse" vector.
    %v2 = arith.constant dense<[ 1,  2,  4,  8,  16,  32,  64,  128 ]> : tensor<8xi64>
    %dv = sparse_tensor.convert %v2 : tensor<8xi64> to tensor<8xi64, #SparseVector>

    // Setup input sparse matrix.
    %m1 = arith.constant sparse<[[1,1], [2,3]], [10, 20]> : tensor<3x4xi64>
    %sm = sparse_tensor.convert %m1 : tensor<3x4xi64> to tensor<3x4xi64, #SparseMatrix>

    // Setup input "sparse" matrix.
    %m2 = arith.constant dense <[ [ 1,  1,  1,  1 ],
                                  [ 1,  2,  1,  1 ],
                                  [ 1,  1,  3,  4 ] ]> : tensor<3x4xi64>
    %dm = sparse_tensor.convert %m2 : tensor<3x4xi64> to tensor<3x4xi64, #SparseMatrix>

    // Setup input sparse f32 matrix.
    %mf32 = arith.constant sparse<[[0,1], [1,2]], [10.0, 41.0]> : tensor<2x3xf32>
    %sf32 = sparse_tensor.convert %mf32 : tensor<2x3xf32> to tensor<2x3xf32, #SparseMatrix>

    // Call the kernels.
    %0 = call @sparse_index_1d_conj(%sv) : (tensor<8xi64, #SparseVector>)
      -> tensor<8xi64, #SparseVector>
    %1 = call @sparse_index_1d_disj(%sv) : (tensor<8xi64, #SparseVector>)
      -> tensor<8xi64, #SparseVector>
    %2 = call @sparse_index_1d_conj(%dv) : (tensor<8xi64, #SparseVector>)
      -> tensor<8xi64, #SparseVector>
    %3 = call @sparse_index_1d_disj(%dv) : (tensor<8xi64, #SparseVector>)
      -> tensor<8xi64, #SparseVector>
    %4 = call @sparse_index_2d_conj(%sm) : (tensor<3x4xi64, #SparseMatrix>)
      -> tensor<3x4xi64, #SparseMatrix>
    %5 = call @sparse_index_2d_disj(%sm) : (tensor<3x4xi64, #SparseMatrix>)
      -> tensor<3x4xi64, #SparseMatrix>
    %6 = call @sparse_index_2d_conj(%dm) : (tensor<3x4xi64, #SparseMatrix>)
      -> tensor<3x4xi64, #SparseMatrix>
    %7 = call @sparse_index_2d_disj(%dm) : (tensor<3x4xi64, #SparseMatrix>)
      -> tensor<3x4xi64, #SparseMatrix>

    //
    // Verify result.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 2
    // CHECK-NEXT: dim = ( 8 )
    // CHECK-NEXT: lvl = ( 8 )
    // CHECK-NEXT: pos[0] : ( 0, 2 )
    // CHECK-NEXT: crd[0] : ( 2, 4 )
    // CHECK-NEXT: values : ( 20, 80 )
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 8
    // CHECK-NEXT: dim = ( 8 )
    // CHECK-NEXT: lvl = ( 8 )
    // CHECK-NEXT: pos[0] : ( 0, 8 )
    // CHECK-NEXT: crd[0] : ( 0, 1, 2, 3, 4, 5, 6, 7 )
    // CHECK-NEXT: values : ( 0, 1, 12, 3, 24, 5, 6, 7 )
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 8
    // CHECK-NEXT: dim = ( 8 )
    // CHECK-NEXT: lvl = ( 8 )
    // CHECK-NEXT: pos[0] : ( 0, 8 )
    // CHECK-NEXT: crd[0] : ( 0, 1, 2, 3, 4, 5, 6, 7 )
    // CHECK-NEXT: values : ( 0, 2, 8, 24, 64, 160, 384, 896 )
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 8
    // CHECK-NEXT: dim = ( 8 )
    // CHECK-NEXT: lvl = ( 8 )
    // CHECK-NEXT: pos[0] : ( 0, 8 )
    // CHECK-NEXT: crd[0] : ( 0, 1, 2, 3, 4, 5, 6, 7 )
    // CHECK-NEXT: values : ( 1, 3, 6, 11, 20, 37, 70, 135 )
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 2
    // CHECK-NEXT: dim = ( 3, 4 )
    // CHECK-NEXT: lvl = ( 3, 4 )
    // CHECK-NEXT: pos[0] : ( 0, 2 )
    // CHECK-NEXT: crd[0] : ( 1, 2 )
    // CHECK-NEXT: pos[1] : ( 0, 1, 2 )
    // CHECK-NEXT: crd[1] : ( 1, 3 )
    // CHECK-NEXT: values : ( 10, 120 )
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 12
    // CHECK-NEXT: dim = ( 3, 4 )
    // CHECK-NEXT: lvl = ( 3, 4 )
    // CHECK-NEXT: pos[0] : ( 0, 3 )
    // CHECK-NEXT: crd[0] : ( 0, 1, 2 )
    // CHECK-NEXT: pos[1] : ( 0, 4, 8, 12 )
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3 )
    // CHECK-NEXT: values : ( 0, 1, 2, 3, 1, 12, 3, 4, 2, 3, 4, 25 )
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 12
    // CHECK-NEXT: dim = ( 3, 4 )
    // CHECK-NEXT: lvl = ( 3, 4 )
    // CHECK-NEXT: pos[0] : ( 0, 3 )
    // CHECK-NEXT: crd[0] : ( 0, 1, 2 )
    // CHECK-NEXT: pos[1] : ( 0, 4, 8, 12 )
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3 )
    // CHECK-NEXT: values : ( 0, 0, 0, 0, 0, 2, 2, 3, 0, 2, 12, 24 )
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 12
    // CHECK-NEXT: dim = ( 3, 4 )
    // CHECK-NEXT: lvl = ( 3, 4 )
    // CHECK-NEXT: pos[0] : ( 0, 3 )
    // CHECK-NEXT: crd[0] : ( 0, 1, 2 )
    // CHECK-NEXT: pos[1] : ( 0, 4, 8, 12 )
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3 )
    // CHECK-NEXT: values : ( 1, 2, 3, 4, 2, 4, 4, 5, 3, 4, 7, 9 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %0 : tensor<8xi64, #SparseVector>
    sparse_tensor.print %1 : tensor<8xi64, #SparseVector>
    sparse_tensor.print %2 : tensor<8xi64, #SparseVector>
    sparse_tensor.print %3 : tensor<8xi64, #SparseVector>
    sparse_tensor.print %4 : tensor<3x4xi64, #SparseMatrix>
    sparse_tensor.print %5 : tensor<3x4xi64, #SparseMatrix>
    sparse_tensor.print %6 : tensor<3x4xi64, #SparseMatrix>
    sparse_tensor.print %7 : tensor<3x4xi64, #SparseMatrix>

    //
    // Call the f32 kernel, verify the result.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 6
    // CHECK-NEXT: dim = ( 2, 3 )
    // CHECK-NEXT: lvl = ( 2, 3 )
    // CHECK-NEXT: pos[0] : ( 0, 2 )
    // CHECK-NEXT: crd[0] : ( 0, 1 )
    // CHECK-NEXT: pos[1] : ( 0, 3, 6 )
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 0, 1, 2 )
    // CHECK-NEXT: values : ( 0, 10, 0, 1, 1, 42 )
    // CHECK-NEXT: ----
    //
    %100 = call @add_outer_2d(%sf32) : (tensor<2x3xf32, #SparseMatrix>)
      -> tensor<2x3xf32, #SparseMatrix>
    sparse_tensor.print %100 : tensor<2x3xf32, #SparseMatrix>

    // Release resources.
    bufferization.dealloc_tensor %sv : tensor<8xi64, #SparseVector>
    bufferization.dealloc_tensor %dv : tensor<8xi64, #SparseVector>
    bufferization.dealloc_tensor %0 : tensor<8xi64, #SparseVector>
    bufferization.dealloc_tensor %1 : tensor<8xi64, #SparseVector>
    bufferization.dealloc_tensor %2 : tensor<8xi64, #SparseVector>
    bufferization.dealloc_tensor %3 : tensor<8xi64, #SparseVector>
    bufferization.dealloc_tensor %sm : tensor<3x4xi64, #SparseMatrix>
    bufferization.dealloc_tensor %dm : tensor<3x4xi64, #SparseMatrix>
    bufferization.dealloc_tensor %4 : tensor<3x4xi64, #SparseMatrix>
    bufferization.dealloc_tensor %5 : tensor<3x4xi64, #SparseMatrix>
    bufferization.dealloc_tensor %6 : tensor<3x4xi64, #SparseMatrix>
    bufferization.dealloc_tensor %7 : tensor<3x4xi64, #SparseMatrix>
    bufferization.dealloc_tensor %sf32 : tensor<2x3xf32, #SparseMatrix>
    bufferization.dealloc_tensor %100 : tensor<2x3xf32, #SparseMatrix>

    return
  }
}
