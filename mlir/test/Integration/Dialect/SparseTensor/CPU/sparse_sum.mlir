// RUN: mlir-opt %s --sparse-compiler | \
// RUN: TENSOR0="%mlir_src_dir/test/Integration/data/test_symmetric.mtx" \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//
// Do the same run, but now with SIMDization as well. This should not change the outcome.
//
// RUN: mlir-opt %s --sparse-compiler="vectorization-strategy=any-storage-inner-loop vl=2" | \
// RUN: TENSOR0="%mlir_src_dir/test/Integration/data/test_symmetric.mtx" \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

!Filename = !llvm.ptr<i8>

#SparseMatrix = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ]
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
  func.func @entry() {
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
    // CHECK: 30.2
    //
    %v = tensor.extract %0[] : tensor<f64>
    vector.print %v : f64

    // Release the resources.
    bufferization.dealloc_tensor %a : tensor<?x?xf64, #SparseMatrix>

    return
  }
}
