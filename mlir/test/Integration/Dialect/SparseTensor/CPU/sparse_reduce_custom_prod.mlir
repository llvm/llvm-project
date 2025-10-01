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

// Product reductions - kept in a separate file as these are not supported by
// the AArch64 SVE backend (so the set-up is a bit different to
// sparse_reducitons.mlir)

#SparseVector = #sparse_tensor.encoding<{map = (d0) -> (d0 : compressed)}>
#CSR = #sparse_tensor.encoding<{map = (d0, d1) -> (d0 : dense, d1 : compressed)}>
#CSC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : dense, d0 : compressed)
}>

//
// Traits for tensor operations.
//

#trait_mat_reduce_rowwise = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A (in)
    affine_map<(i,j) -> (i)>   // X (out)
  ],
  iterator_types = ["parallel", "reduction"],
  doc = "X(i) = PROD_j A(i,j)"
}

module {
  func.func @redProdLex(%arga: tensor<?x?xf64, #CSR>) -> tensor<?xf64, #SparseVector> {
    %c0 = arith.constant 0 : index
    %cf1 = arith.constant 1.0 : f64
    %d0 = tensor.dim %arga, %c0 : tensor<?x?xf64, #CSR>
    %xv = tensor.empty(%d0): tensor<?xf64, #SparseVector>
    %0 = linalg.generic #trait_mat_reduce_rowwise
      ins(%arga: tensor<?x?xf64, #CSR>)
      outs(%xv: tensor<?xf64, #SparseVector>) {
        ^bb(%a: f64, %b: f64):
          %1 = sparse_tensor.reduce %a, %b, %cf1 : f64 {
              ^bb0(%x: f64, %y: f64):
                %2 = arith.mulf %x, %y : f64
                sparse_tensor.yield %2 : f64
            }
          linalg.yield %1 : f64
    } -> tensor<?xf64, #SparseVector>
    return %0 : tensor<?xf64, #SparseVector>
  }

  func.func @redProdExpand(%arga: tensor<?x?xf64, #CSC>) -> tensor<?xf64, #SparseVector> {
    %c0 = arith.constant 0 : index
    %cf1 = arith.constant 1.0 : f64
    %d0 = tensor.dim %arga, %c0 : tensor<?x?xf64, #CSC>
    %xv = tensor.empty(%d0): tensor<?xf64, #SparseVector>
    %0 = linalg.generic #trait_mat_reduce_rowwise
      ins(%arga: tensor<?x?xf64, #CSC>)
      outs(%xv: tensor<?xf64, #SparseVector>) {
        ^bb(%a: f64, %b: f64):
          %1 = sparse_tensor.reduce %a, %b, %cf1 : f64 {
              ^bb0(%x: f64, %y: f64):
                %2 = arith.mulf %x, %y : f64
                sparse_tensor.yield %2 : f64
            }
          linalg.yield %1 : f64
    } -> tensor<?xf64, #SparseVector>
    return %0 : tensor<?xf64, #SparseVector>
  }


  // Driver method to call and verify vector kernels.
  func.func @main() {
    %c0 = arith.constant 0 : index

    // Setup sparse matrices.
    %m1 = arith.constant sparse<
       [ [0,0], [0,1], [1,0], [2,2], [2,3], [2,4], [3,0], [3,2], [3,3] ],
         [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 ]
    > : tensor<4x5xf64>
    %m2 = arith.constant sparse<
       [ [0,0], [1,3], [2,0], [2,3], [3,1], [4,1] ],
         [6.0, 5.0, 4.0, 3.0, 2.0, 11.0 ]
    > : tensor<5x4xf64>
    %sm1 = sparse_tensor.convert %m1 : tensor<4x5xf64> to tensor<?x?xf64, #CSR>
    %sm2r = sparse_tensor.convert %m2 : tensor<5x4xf64> to tensor<?x?xf64, #CSR>
    %sm2c = sparse_tensor.convert %m2 : tensor<5x4xf64> to tensor<?x?xf64, #CSC>

    // Call sparse matrix kernels.
    %1 = call @redProdLex(%sm1) : (tensor<?x?xf64, #CSR>) -> tensor<?xf64, #SparseVector>
    %2 = call @redProdExpand(%sm2c) : (tensor<?x?xf64, #CSC>) -> tensor<?xf64, #SparseVector>

    //
    // Verify the results.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 9
    // CHECK-NEXT: dim = ( 4, 5 )
    // CHECK-NEXT: lvl = ( 4, 5 )
    // CHECK-NEXT: pos[1] : ( 0, 2, 3, 6, 9 )
    // CHECK-NEXT: crd[1] : ( 0, 1, 0, 2, 3, 4, 0, 2, 3 )
    // CHECK-NEXT: values : ( 1, 2, 3, 4, 5, 6, 7, 8, 9 )
    // CHECK-NEXT: ----
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 6
    // CHECK-NEXT: dim = ( 5, 4 )
    // CHECK-NEXT: lvl = ( 5, 4 )
    // CHECK-NEXT: pos[1] : ( 0, 1, 2, 4, 5, 6 )
    // CHECK-NEXT: crd[1] : ( 0, 3, 0, 3, 1, 1 )
    // CHECK-NEXT: values : ( 6, 5, 4, 3, 2, 11 )
    // CHECK-NEXT: ----
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 4
    // CHECK-NEXT: dim = ( 4 )
    // CHECK-NEXT: lvl = ( 4 )
    // CHECK-NEXT: pos[0] : ( 0, 4 )
    // CHECK-NEXT: crd[0] : ( 0, 1, 2, 3 )
    // CHECK-NEXT: values : ( 2, 3, 120, 504 )
    // CHECK-NEXT: ----
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 5
    // CHECK-NEXT: dim = ( 5 )
    // CHECK-NEXT: lvl = ( 5 )
    // CHECK-NEXT: pos[0] : ( 0, 5 )
    // CHECK-NEXT: crd[0] : ( 0, 1, 2, 3, 4 )
    // CHECK-NEXT: values : ( 6, 5, 12, 2, 11 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %sm1 : tensor<?x?xf64, #CSR>
    sparse_tensor.print %sm2r : tensor<?x?xf64, #CSR>
    sparse_tensor.print %1 : tensor<?xf64, #SparseVector>
    sparse_tensor.print %2 : tensor<?xf64, #SparseVector>

    // Release the resources.
    bufferization.dealloc_tensor %sm1 : tensor<?x?xf64, #CSR>
    bufferization.dealloc_tensor %sm2r : tensor<?x?xf64, #CSR>
    bufferization.dealloc_tensor %sm2c : tensor<?x?xf64, #CSC>
    bufferization.dealloc_tensor %1 : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %2 : tensor<?xf64, #SparseVector>
    return
  }
}
