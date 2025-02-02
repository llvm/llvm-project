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

// Reduction in this file _are_ supported by the AArch64 SVE backend

#SparseVector = #sparse_tensor.encoding<{map = (d0) -> (d0 : compressed)}>
#CSR = #sparse_tensor.encoding<{map = (d0, d1) -> (d0 : dense, d1 : compressed)}>
#CSC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : dense, d0 : compressed)
}>

//
// Traits for tensor operations.
//
#trait_matmul = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,k)>, // A
    affine_map<(i,j,k) -> (k,j)>, // B
    affine_map<(i,j,k) -> (i,j)>  // C (out)
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  doc = "C(i,j) = SUM_k A(i,k) * B(k,j)"
}

module {
  func.func @min_plus_csrcsr(%arga: tensor<?x?xf64, #CSR>,
                             %argb: tensor<?x?xf64, #CSR>) -> tensor<?x?xf64, #CSR> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %maxf = arith.constant 1.0e999 : f64
    %d0 = tensor.dim %arga, %c0 : tensor<?x?xf64, #CSR>
    %d1 = tensor.dim %argb, %c1 : tensor<?x?xf64, #CSR>
    %xm = tensor.empty(%d0, %d1) : tensor<?x?xf64, #CSR>
    %0 = linalg.generic #trait_matmul
       ins(%arga, %argb: tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSR>)
        outs(%xm: tensor<?x?xf64, #CSR>) {
        ^bb(%a: f64, %b: f64, %output: f64):
          %1 = sparse_tensor.binary %a, %b : f64, f64 to f64
            overlap = {
              ^bb0(%x: f64, %y: f64):
                %3 = arith.addf %x, %y : f64
                sparse_tensor.yield %3 : f64
            }
            left={}
            right={}
          %2 = sparse_tensor.reduce %1, %output, %maxf : f64 {
              ^bb0(%x: f64, %y: f64):
                %cmp = arith.cmpf "olt", %x, %y : f64
                %3 = arith.select %cmp, %x, %y : f64
                sparse_tensor.yield %3 : f64
            }
          linalg.yield %2 : f64
    } -> tensor<?x?xf64, #CSR>
    return %0 : tensor<?x?xf64, #CSR>
  }

  func.func @min_plus_csrcsc(%arga: tensor<?x?xf64, #CSR>,
                             %argb: tensor<?x?xf64, #CSC>) -> tensor<?x?xf64, #CSR> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %maxf = arith.constant 1.0e999 : f64
    %d0 = tensor.dim %arga, %c0 : tensor<?x?xf64, #CSR>
    %d1 = tensor.dim %argb, %c1 : tensor<?x?xf64, #CSC>
    %xm = tensor.empty(%d0, %d1) : tensor<?x?xf64, #CSR>
    %0 = linalg.generic #trait_matmul
       ins(%arga, %argb: tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSC>)
        outs(%xm: tensor<?x?xf64, #CSR>) {
        ^bb(%a: f64, %b: f64, %output: f64):
          %1 = sparse_tensor.binary %a, %b : f64, f64 to f64
            overlap = {
              ^bb0(%x: f64, %y: f64):
                %3 = arith.addf %x, %y : f64
                sparse_tensor.yield %3 : f64
            }
            left={}
            right={}
          %2 = sparse_tensor.reduce %1, %output, %maxf : f64 {
              ^bb0(%x: f64, %y: f64):
                %cmp = arith.cmpf "olt", %x, %y : f64
                %3 = arith.select %cmp, %x, %y : f64
                sparse_tensor.yield %3 : f64
            }
          linalg.yield %2 : f64
    } -> tensor<?x?xf64, #CSR>
    return %0 : tensor<?x?xf64, #CSR>
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
    %5 = call @min_plus_csrcsr(%sm1, %sm2r)
      : (tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSR>) -> tensor<?x?xf64, #CSR>
    %6 = call @min_plus_csrcsc(%sm1, %sm2c)
      : (tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSC>) -> tensor<?x?xf64, #CSR>

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
    // CHECK-NEXT: nse = 9
    // CHECK-NEXT: dim = ( 4, 4 )
    // CHECK-NEXT: lvl = ( 4, 4 )
    // CHECK-NEXT: pos[1] : ( 0, 2, 3, 6, 9 )
    // CHECK-NEXT: crd[1] : ( 0, 3, 0, 0, 1, 3, 0, 1, 3 )
    // CHECK-NEXT: values : ( 7, 7, 9, 8, 7, 7, 12, 11, 11 )
    // CHECK-NEXT: ----
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 9
    // CHECK-NEXT: dim = ( 4, 4 )
    // CHECK-NEXT: lvl = ( 4, 4 )
    // CHECK-NEXT: pos[1] : ( 0, 2, 3, 6, 9 )
    // CHECK-NEXT: crd[1] : ( 0, 3, 0, 0, 1, 3, 0, 1, 3 )
    // CHECK-NEXT: values : ( 7, 7, 9, 8, 7, 7, 12, 11, 11 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %sm1 : tensor<?x?xf64, #CSR>
    sparse_tensor.print %sm2r : tensor<?x?xf64, #CSR>
    sparse_tensor.print %5 : tensor<?x?xf64, #CSR>
    sparse_tensor.print %6 : tensor<?x?xf64, #CSR>

    // Release the resources.
    bufferization.dealloc_tensor %sm1 : tensor<?x?xf64, #CSR>
    bufferization.dealloc_tensor %sm2r : tensor<?x?xf64, #CSR>
    bufferization.dealloc_tensor %sm2c : tensor<?x?xf64, #CSC>
    bufferization.dealloc_tensor %5 : tensor<?x?xf64, #CSR>
    bufferization.dealloc_tensor %6 : tensor<?x?xf64, #CSR>
    return
  }
}
