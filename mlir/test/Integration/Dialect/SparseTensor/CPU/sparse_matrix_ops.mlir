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
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#DCSR = #sparse_tensor.encoding<{map = (d0, d1) -> (d0 : compressed, d1 : compressed)}>

//
// Traits for 2-d tensor (aka matrix) operations.
//
#trait_scale = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A (in)
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(i,j) * 2.0"
}
#trait_scale_inpl = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) *= 2.0"
}
#trait_op = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A (in)
    affine_map<(i,j) -> (i,j)>,  // B (in)
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(i,j) OP B(i,j)"
}

module {
  // Scales a sparse matrix into a new sparse matrix.
  func.func @matrix_scale(%arga: tensor<?x?xf64, #DCSR>) -> tensor<?x?xf64, #DCSR> {
    %s = arith.constant 2.0 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = tensor.dim %arga, %c0 : tensor<?x?xf64, #DCSR>
    %d1 = tensor.dim %arga, %c1 : tensor<?x?xf64, #DCSR>
    %xm = tensor.empty(%d0, %d1) : tensor<?x?xf64, #DCSR>
    %0 = linalg.generic #trait_scale
       ins(%arga: tensor<?x?xf64, #DCSR>)
        outs(%xm: tensor<?x?xf64, #DCSR>) {
        ^bb(%a: f64, %x: f64):
          %1 = arith.mulf %a, %s : f64
          linalg.yield %1 : f64
    } -> tensor<?x?xf64, #DCSR>
    return %0 : tensor<?x?xf64, #DCSR>
  }

  // Scales a sparse matrix in place.
  func.func @matrix_scale_inplace(%argx: tensor<?x?xf64, #DCSR>) -> tensor<?x?xf64, #DCSR> {
    %s = arith.constant 2.0 : f64
    %0 = linalg.generic #trait_scale_inpl
      outs(%argx: tensor<?x?xf64, #DCSR>) {
        ^bb(%x: f64):
          %1 = arith.mulf %x, %s : f64
          linalg.yield %1 : f64
    } -> tensor<?x?xf64, #DCSR>
    return %0 : tensor<?x?xf64, #DCSR>
  }

  // Adds two sparse matrices element-wise into a new sparse matrix.
  func.func @matrix_add(%arga: tensor<?x?xf64, #DCSR>,
                        %argb: tensor<?x?xf64, #DCSR>) -> tensor<?x?xf64, #DCSR> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = tensor.dim %arga, %c0 : tensor<?x?xf64, #DCSR>
    %d1 = tensor.dim %arga, %c1 : tensor<?x?xf64, #DCSR>
    %xv = tensor.empty(%d0, %d1) : tensor<?x?xf64, #DCSR>
    %0 = linalg.generic #trait_op
       ins(%arga, %argb: tensor<?x?xf64, #DCSR>, tensor<?x?xf64, #DCSR>)
        outs(%xv: tensor<?x?xf64, #DCSR>) {
        ^bb(%a: f64, %b: f64, %x: f64):
          %1 = arith.addf %a, %b : f64
          linalg.yield %1 : f64
    } -> tensor<?x?xf64, #DCSR>
    return %0 : tensor<?x?xf64, #DCSR>
  }

  // Multiplies two sparse matrices element-wise into a new sparse matrix.
  func.func @matrix_mul(%arga: tensor<?x?xf64, #DCSR>,
                        %argb: tensor<?x?xf64, #DCSR>) -> tensor<?x?xf64, #DCSR> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = tensor.dim %arga, %c0 : tensor<?x?xf64, #DCSR>
    %d1 = tensor.dim %arga, %c1 : tensor<?x?xf64, #DCSR>
    %xv = tensor.empty(%d0, %d1) : tensor<?x?xf64, #DCSR>
    %0 = linalg.generic #trait_op
       ins(%arga, %argb: tensor<?x?xf64, #DCSR>, tensor<?x?xf64, #DCSR>)
        outs(%xv: tensor<?x?xf64, #DCSR>) {
        ^bb(%a: f64, %b: f64, %x: f64):
          %1 = arith.mulf %a, %b : f64
          linalg.yield %1 : f64
    } -> tensor<?x?xf64, #DCSR>
    return %0 : tensor<?x?xf64, #DCSR>
  }

  // Driver method to call and verify matrix kernels.
  func.func @main() {
    %c0 = arith.constant 0 : index
    %d1 = arith.constant 1.1 : f64

    // Setup sparse matrices.
    %m1 = arith.constant sparse<
       [ [0,0], [0,1], [1,7], [2,2], [2,4], [2,7], [3,0], [3,2], [3,3] ],
         [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 ]
    > : tensor<4x8xf64>
    %m2 = arith.constant sparse<
       [ [0,0], [0,7], [1,0], [1,6], [2,1], [2,7] ],
         [6.0, 5.0, 4.0, 3.0, 2.0, 1.0 ]
    > : tensor<4x8xf64>
    %sm1 = sparse_tensor.convert %m1 : tensor<4x8xf64> to tensor<?x?xf64, #DCSR>
    // TODO: Use %sm1 when we support sparse tensor copies.
    %sm1_dup = sparse_tensor.convert %m1 : tensor<4x8xf64> to tensor<?x?xf64, #DCSR>
    %sm2 = sparse_tensor.convert %m2 : tensor<4x8xf64> to tensor<?x?xf64, #DCSR>

    // Call sparse matrix kernels.
    %0 = call @matrix_scale(%sm1)
      : (tensor<?x?xf64, #DCSR>) -> tensor<?x?xf64, #DCSR>
    %1 = call @matrix_scale_inplace(%sm1_dup)
      : (tensor<?x?xf64, #DCSR>) -> tensor<?x?xf64, #DCSR>
    %2 = call @matrix_add(%1, %sm2)
      : (tensor<?x?xf64, #DCSR>, tensor<?x?xf64, #DCSR>) -> tensor<?x?xf64, #DCSR>
    %3 = call @matrix_mul(%1, %sm2)
      : (tensor<?x?xf64, #DCSR>, tensor<?x?xf64, #DCSR>) -> tensor<?x?xf64, #DCSR>

    //
    // Verify the results.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 9
    // CHECK-NEXT: dim = ( 4, 8 )
    // CHECK-NEXT: lvl = ( 4, 8 )
    // CHECK-NEXT: pos[0] : ( 0, 4 )
    // CHECK-NEXT: crd[0] : ( 0, 1, 2, 3 )
    // CHECK-NEXT: pos[1] : ( 0, 2, 3, 6, 9 )
    // CHECK-NEXT: crd[1] : ( 0, 1, 7, 2, 4, 7, 0, 2, 3 )
    // CHECK-NEXT: values : ( 1, 2, 3, 4, 5, 6, 7, 8, 9 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %sm1 : tensor<?x?xf64, #DCSR>

    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 6
    // CHECK-NEXT: dim = ( 4, 8 )
    // CHECK-NEXT: lvl = ( 4, 8 )
    // CHECK-NEXT: pos[0] : ( 0, 3 )
    // CHECK-NEXT: crd[0] : ( 0, 1, 2 )
    // CHECK-NEXT: pos[1] : ( 0, 2, 4, 6 )
    // CHECK-NEXT: crd[1] : ( 0, 7, 0, 6, 1, 7 )
    // CHECK-NEXT: values : ( 6, 5, 4, 3, 2, 1 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %sm2 : tensor<?x?xf64, #DCSR>

    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 9
    // CHECK-NEXT: dim = ( 4, 8 )
    // CHECK-NEXT: lvl = ( 4, 8 )
    // CHECK-NEXT: pos[0] : ( 0, 4 )
    // CHECK-NEXT: crd[0] : ( 0, 1, 2, 3 )
    // CHECK-NEXT: pos[1] : ( 0, 2, 3, 6, 9 )
    // CHECK-NEXT: crd[1] : ( 0, 1, 7, 2, 4, 7, 0, 2, 3 )
    // CHECK-NEXT: values : ( 2, 4, 6, 8, 10, 12, 14, 16, 18 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %0 : tensor<?x?xf64, #DCSR>

    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 9
    // CHECK-NEXT: dim = ( 4, 8 )
    // CHECK-NEXT: lvl = ( 4, 8 )
    // CHECK-NEXT: pos[0] : ( 0, 4 )
    // CHECK-NEXT: crd[0] : ( 0, 1, 2, 3 )
    // CHECK-NEXT: pos[1] : ( 0, 2, 3, 6, 9 )
    // CHECK-NEXT: crd[1] : ( 0, 1, 7, 2, 4, 7, 0, 2, 3 )
    // CHECK-NEXT: values : ( 2, 4, 6, 8, 10, 12, 14, 16, 18 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %1 : tensor<?x?xf64, #DCSR>

    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 13
    // CHECK-NEXT: dim = ( 4, 8 )
    // CHECK-NEXT: lvl = ( 4, 8 )
    // CHECK-NEXT: pos[0] : ( 0, 4 )
    // CHECK-NEXT: crd[0] : ( 0, 1, 2, 3 )
    // CHECK-NEXT: pos[1] : ( 0, 3, 6, 10, 13 )
    // CHECK-NEXT: crd[1] : ( 0, 1, 7, 0, 6, 7, 1, 2, 4, 7, 0, 2, 3 )
    // CHECK-NEXT: values : ( 8, 4, 5, 4, 3, 6, 2, 8, 10, 13, 14, 16, 18 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %2 : tensor<?x?xf64, #DCSR>

    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 2
    // CHECK-NEXT: dim = ( 4, 8 )
    // CHECK-NEXT: lvl = ( 4, 8 )
    // CHECK-NEXT: pos[0] : ( 0, 2 )
    // CHECK-NEXT: crd[0] : ( 0, 2 )
    // CHECK-NEXT: pos[1] : ( 0, 1, 2 )
    // CHECK-NEXT: crd[1] : ( 0, 7 )
    // CHECK-NEXT: values : ( 12, 12 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %3 : tensor<?x?xf64, #DCSR>

    // Release the resources.
    bufferization.dealloc_tensor %sm1 : tensor<?x?xf64, #DCSR>
    bufferization.dealloc_tensor %sm1_dup : tensor<?x?xf64, #DCSR>
    bufferization.dealloc_tensor %sm2 : tensor<?x?xf64, #DCSR>
    bufferization.dealloc_tensor %0 : tensor<?x?xf64, #DCSR>
    bufferization.dealloc_tensor %2 : tensor<?x?xf64, #DCSR>
    bufferization.dealloc_tensor %3 : tensor<?x?xf64, #DCSR>
    return
  }
}
