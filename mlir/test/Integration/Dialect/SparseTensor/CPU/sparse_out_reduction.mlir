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

#SparseMatrix = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed, d1 : compressed)
}>

#SparseTensor = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : compressed, d1 : compressed, d2 : compressed)
}>

#redsum = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,j,k)>, // A
    affine_map<(i,j,k) -> (i,j,k)>, // B
    affine_map<(i,j,k) -> (i,j)>    // X (out)
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  doc = "X(i,j) = SUM_k A(i,j,k) * B(i,j,k)"
}

module {
  func.func @redsum(%arga: tensor<?x?x?xi32, #SparseTensor>,
               %argb: tensor<?x?x?xi32, #SparseTensor>)
                   -> tensor<?x?xi32, #SparseMatrix> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = tensor.dim %arga, %c0 : tensor<?x?x?xi32, #SparseTensor>
    %d1 = tensor.dim %arga, %c1 : tensor<?x?x?xi32, #SparseTensor>
    %xinit = tensor.empty(%d0, %d1): tensor<?x?xi32, #SparseMatrix>
    %0 = linalg.generic #redsum
      ins(%arga, %argb: tensor<?x?x?xi32, #SparseTensor>,
                        tensor<?x?x?xi32, #SparseTensor>)
      outs(%xinit: tensor<?x?xi32, #SparseMatrix>) {
        ^bb(%a: i32, %b: i32, %x: i32):
          %0 = arith.muli %a, %b : i32
          %1 = arith.addi %x, %0 : i32
          linalg.yield %1 : i32
    } -> tensor<?x?xi32, #SparseMatrix>
    return %0 : tensor<?x?xi32, #SparseMatrix>
  }

  // Driver method to call and verify tensor kernel.
  func.func @main() {
    // Setup very sparse 3-d tensors.
    %t1 = arith.constant sparse<
       [ [1,1,3], [2,0,0], [2,2,1], [2,2,2], [2,2,3] ], [ 1, 2, 3, 4, 5 ]
    > : tensor<3x3x4xi32>
    %t2 = arith.constant sparse<
       [ [1,0,0], [1,1,3], [2,2,1], [2,2,3] ], [ 6, 7, 8, 9 ]
    > : tensor<3x3x4xi32>
    %st1 = sparse_tensor.convert %t1
      : tensor<3x3x4xi32> to tensor<?x?x?xi32, #SparseTensor>
    %st2 = sparse_tensor.convert %t2
      : tensor<3x3x4xi32> to tensor<?x?x?xi32, #SparseTensor>

    // Call kernel.
    %0 = call @redsum(%st1, %st2)
      : (tensor<?x?x?xi32, #SparseTensor>,
         tensor<?x?x?xi32, #SparseTensor>) -> tensor<?x?xi32, #SparseMatrix>

    //
    // Verify results. Only two entries stored in result. Correct structure.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 2
    // CHECK-NEXT: dim = ( 3, 3 )
    // CHECK-NEXT: lvl = ( 3, 3 )
    // CHECK-NEXT: pos[0] : ( 0, 2 )
    // CHECK-NEXT: crd[0] : ( 1, 2 )
    // CHECK-NEXT: pos[1] : ( 0, 1, 2 )
    // CHECK-NEXT: crd[1] : ( 1, 2 )
    // CHECK-NEXT: values : ( 7, 69 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %0 : tensor<?x?xi32, #SparseMatrix>

    // Release the resources.
    bufferization.dealloc_tensor %st1 : tensor<?x?x?xi32, #SparseTensor>
    bufferization.dealloc_tensor %st2 : tensor<?x?x?xi32, #SparseTensor>
    bufferization.dealloc_tensor %0 : tensor<?x?xi32, #SparseMatrix>

    return
  }
}
