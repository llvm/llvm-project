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

// REDEFINE: %{env} = TENSOR0="%mlir_src_dir/test/Integration/data/test.mtx"
// RUN: %{compile} | env %{env} %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false
// RUN: %{compile} | env %{env} %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | env %{env} %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | env %{env} %{run_sve} | FileCheck %s %}

!Filename = !llvm.ptr

#DenseMatrix = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : dense)
}>

#SparseMatrix = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
}>

#trait_assign = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>, // A
    affine_map<(i,j) -> (i,j)>  // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(i,j) * 2"
}

//
// Integration test that demonstrates assigning a sparse tensor
// to an all-dense annotated "sparse" tensor, which effectively
// result in inserting the nonzero elements into a linearized array.
//
// Note that there is a subtle difference between a non-annotated
// tensor and an all-dense annotated tensor. Both tensors are assumed
// dense, but the former remains an n-dimensional memref whereas the
// latter is linearized into a one-dimensional memref that is further
// lowered into a storage scheme that is backed by the runtime support
// library.
module {
  //
  // A kernel that assigns multiplied elements from A to X.
  //
  func.func @dense_output(%arga: tensor<?x?xf64, #SparseMatrix>) -> tensor<?x?xf64, #DenseMatrix> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2.0 : f64
    %d0 = tensor.dim %arga, %c0 : tensor<?x?xf64, #SparseMatrix>
    %d1 = tensor.dim %arga, %c1 : tensor<?x?xf64, #SparseMatrix>
    %init = tensor.empty(%d0, %d1) : tensor<?x?xf64, #DenseMatrix>
    %0 = linalg.generic #trait_assign
       ins(%arga: tensor<?x?xf64, #SparseMatrix>)
      outs(%init: tensor<?x?xf64, #DenseMatrix>) {
      ^bb(%a: f64, %x: f64):
        %0 = arith.mulf %a, %c2 : f64
        linalg.yield %0 : f64
    } -> tensor<?x?xf64, #DenseMatrix>
    return %0 : tensor<?x?xf64, #DenseMatrix>
  }

  func.func private @getTensorFilename(index) -> (!Filename)

  //
  // Main driver that reads matrix from file and calls the kernel.
  //
  func.func @main() {
    %d0 = arith.constant 0.0 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Read the sparse matrix from file, construct sparse storage.
    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)
    %a = sparse_tensor.new %fileName
      : !Filename to tensor<?x?xf64, #SparseMatrix>

    // Call the kernel.
    %0 = call @dense_output(%a)
      : (tensor<?x?xf64, #SparseMatrix>) -> tensor<?x?xf64, #DenseMatrix>

    //
    // Print the linearized 5x5 result for verification.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 25
    // CHECK-NEXT: dim = ( 5, 5 )
    // CHECK-NEXT: lvl = ( 5, 5 )
    // CHECK-NEXT: values : ( 2, 0, 0, 2.8, 0, 0, 4, 0, 0, 5, 0, 0, 6, 0, 0, 8.2, 0, 0, 8, 0, 0, 10.4, 0, 0, 10,
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %0 : tensor<?x?xf64, #DenseMatrix>

    // Release the resources.
    bufferization.dealloc_tensor %a : tensor<?x?xf64, #SparseMatrix>
    bufferization.dealloc_tensor %0 : tensor<?x?xf64, #DenseMatrix>

    return
  }
}
