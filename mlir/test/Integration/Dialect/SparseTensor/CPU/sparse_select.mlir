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
// Do the same run, but now with vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false vl=4 enable-buffer-initialization=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with  VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#SparseVector = #sparse_tensor.encoding<{map = (d0) -> (d0 : compressed)}>
#CSR = #sparse_tensor.encoding<{map = (d0, d1) -> (d0 : dense, d1 : compressed)}>
#CSC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : dense, d0 : compressed)
}>

//
// Traits for tensor operations.
//
#trait_vec_select = {
  indexing_maps = [
    affine_map<(i) -> (i)>, // A
    affine_map<(i) -> (i)>  // C (out)
  ],
  iterator_types = ["parallel"]
}

#trait_mat_select = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A (in)
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"]
}

module {
  func.func @vecSelect(%arga: tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector> {
    %c0 = arith.constant 0 : index
    %cf1 = arith.constant 1.0 : f64
    %d0 = tensor.dim %arga, %c0 : tensor<?xf64, #SparseVector>
    %xv = tensor.empty(%d0): tensor<?xf64, #SparseVector>
    %0 = linalg.generic #trait_vec_select
      ins(%arga: tensor<?xf64, #SparseVector>)
      outs(%xv: tensor<?xf64, #SparseVector>) {
        ^bb(%a: f64, %b: f64):
          %1 = sparse_tensor.select %a : f64 {
              ^bb0(%x: f64):
                %keep = arith.cmpf "oge", %x, %cf1 : f64
                sparse_tensor.yield %keep : i1
            }
          linalg.yield %1 : f64
    } -> tensor<?xf64, #SparseVector>
    return %0 : tensor<?xf64, #SparseVector>
  }

  func.func @matUpperTriangle(%arga: tensor<?x?xf64, #CSR>) -> tensor<?x?xf64, #CSR> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = tensor.dim %arga, %c0 : tensor<?x?xf64, #CSR>
    %d1 = tensor.dim %arga, %c1 : tensor<?x?xf64, #CSR>
    %xv = tensor.empty(%d0, %d1): tensor<?x?xf64, #CSR>
    %0 = linalg.generic #trait_mat_select
      ins(%arga: tensor<?x?xf64, #CSR>)
      outs(%xv: tensor<?x?xf64, #CSR>) {
        ^bb(%a: f64, %b: f64):
          %row = linalg.index 0 : index
          %col = linalg.index 1 : index
          %1 = sparse_tensor.select %a : f64 {
              ^bb0(%x: f64):
                %keep = arith.cmpi "ugt", %col, %row : index
                sparse_tensor.yield %keep : i1
            }
          linalg.yield %1 : f64
    } -> tensor<?x?xf64, #CSR>
    return %0 : tensor<?x?xf64, #CSR>
  }

  // Driver method to call and verify vector kernels.
  func.func @main() {
    %c0 = arith.constant 0 : index

    // Setup sparse matrices.
    %v1 = arith.constant sparse<
        [ [1], [3], [5], [7], [9] ],
        [ 1.0, 2.0, -4.0, 0.0, 5.0 ]
    > : tensor<10xf64>
    %m1 = arith.constant sparse<
        [ [0, 3], [1, 4], [2, 1], [2, 3], [3, 3], [3, 4], [4, 2] ],
        [ 1., 2., 3., 4., 5., 6., 7.]
    > : tensor<5x5xf64>
    %sv1 = sparse_tensor.convert %v1 : tensor<10xf64> to tensor<?xf64, #SparseVector>
    %sm1 = sparse_tensor.convert %m1 : tensor<5x5xf64> to tensor<?x?xf64, #CSR>

    // Call sparse matrix kernels.
    %1 = call @vecSelect(%sv1) : (tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector>
    %2 = call @matUpperTriangle(%sm1) : (tensor<?x?xf64, #CSR>) -> tensor<?x?xf64, #CSR>

    //
    // Verify the results.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 5
    // CHECK-NEXT: dim = ( 10 )
    // CHECK-NEXT: lvl = ( 10 )
    // CHECK-NEXT: pos[0] : ( 0, 5 )
    // CHECK-NEXT: crd[0] : ( 1, 3, 5, 7, 9 )
    // CHECK-NEXT: values : ( 1, 2, -4, 0, 5 )
    // CHECK-NEXT: ----
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 7
    // CHECK-NEXT: dim = ( 5, 5 )
    // CHECK-NEXT: lvl = ( 5, 5 )
    // CHECK-NEXT: pos[1] : ( 0, 1, 2, 4, 6, 7 )
    // CHECK-NEXT: crd[1] : ( 3, 4, 1, 3, 3, 4, 2 )
    // CHECK-NEXT: values : ( 1, 2, 3, 4, 5, 6, 7 )
    // CHECK-NEXT: ----
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 3
    // CHECK-NEXT: dim = ( 10 )
    // CHECK-NEXT: lvl = ( 10 )
    // CHECK-NEXT: pos[0] : ( 0, 3 )
    // CHECK-NEXT: crd[0] : ( 1, 3, 9 )
    // CHECK-NEXT: values : ( 1, 2, 5 )
    // CHECK-NEXT: ----
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 4
    // CHECK-NEXT: dim = ( 5, 5 )
    // CHECK-NEXT: lvl = ( 5, 5 )
    // CHECK-NEXT: pos[1] : ( 0, 1, 2, 3, 4, 4 )
    // CHECK-NEXT: crd[1] : ( 3, 4, 3, 4 )
    // CHECK-NEXT: values : ( 1, 2, 4, 6 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %sv1 : tensor<?xf64, #SparseVector>
    sparse_tensor.print %sm1 : tensor<?x?xf64, #CSR>
    sparse_tensor.print %1 : tensor<?xf64, #SparseVector>
    sparse_tensor.print %2 : tensor<?x?xf64, #CSR>

    // Release the resources.
    bufferization.dealloc_tensor %sv1 : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %sm1 : tensor<?x?xf64, #CSR>
    bufferization.dealloc_tensor %1 : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %2 : tensor<?x?xf64, #CSR>
    return
  }
}
