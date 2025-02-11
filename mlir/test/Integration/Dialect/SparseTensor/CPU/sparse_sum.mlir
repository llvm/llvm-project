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

// REDEFINE: %{env} = TENSOR0=%mlir_src_dir/test/Integration/data/test_symmetric.mtx
// RUN: %{compile} | env %{env} %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false
// RUN: %{compile} | env %{env} %{run} | FileCheck %s
//
// Do the same run, but now with vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | env %{env} %{run} | FileCheck %s
//
// Do the same run, but now with  VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | env %{env} %{run_sve} | FileCheck %s %}

// TODO: The test currently only operates on the triangular part of the
// symmetric matrix.

!Filename = !llvm.ptr

#SparseMatrix = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed, d1 : compressed)
}>

#trait_sum_reduce = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>, // A
    affine_map<(i,j) -> ()>     // x (out)
  ],
  iterator_types = ["reduction", "reduction"],
  doc = "x += A(i,j)"
}

//
// Integration test that lowers a kernel annotated as sparse to
// actual sparse code, initializes a matching sparse storage scheme
// from file, and runs the resulting code with the JIT compiler.
//
module {
  //
  // A kernel that sum-reduces a matrix to a single scalar.
  //
  func.func @kernel_sum_reduce(%arga: tensor<?x?xf64, #SparseMatrix>,
                               %argx: tensor<f64>) -> tensor<f64> {
    %0 = linalg.generic #trait_sum_reduce
      ins(%arga: tensor<?x?xf64, #SparseMatrix>)
      outs(%argx: tensor<f64>) {
      ^bb(%a: f64, %x: f64):
        %0 = arith.addf %x, %a : f64
        linalg.yield %0 : f64
    } -> tensor<f64>
    return %0 : tensor<f64>
  }

  func.func private @getTensorFilename(index) -> (!Filename)

  //
  // Main driver that reads matrix from file and calls the sparse kernel.
  //
  func.func @main() {
    %d0 = arith.constant 0.0 : f64
    %c0 = arith.constant 0 : index

    // Setup memory for a single reduction scalar,
    // initialized to zero.
    %x = tensor.from_elements %d0 : tensor<f64>

    // Read the sparse matrix from file, construct sparse storage.
    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)
    %a = sparse_tensor.new %fileName : !Filename to tensor<?x?xf64, #SparseMatrix>

    // Call the kernel.
    %0 = call @kernel_sum_reduce(%a, %x)
      : (tensor<?x?xf64, #SparseMatrix>, tensor<f64>) -> tensor<f64>

    // Print the result for verification.
    //
    // CHECK: 24.1
    //
    %v = tensor.extract %0[] : tensor<f64>
    vector.print %v : f64

    // Release the resources.
    bufferization.dealloc_tensor %a : tensor<?x?xf64, #SparseMatrix>
    bufferization.dealloc_tensor %0 : tensor<f64>

    return
  }
}
