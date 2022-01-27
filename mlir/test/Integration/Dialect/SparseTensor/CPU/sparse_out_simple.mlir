// RUN: mlir-opt %s --sparse-compiler | \
// RUN: TENSOR0="%mlir_integration_test_dir/data/test.mtx" \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//
// Do the same run, but now with SIMDization as well. This should not change the outcome.
//
// RUN: mlir-opt %s --sparse-compiler="vectorization-strategy=2 vl=4" | \
// RUN: TENSOR0="%mlir_integration_test_dir/data/test.mtx" \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

!Filename = type !llvm.ptr<i8>

#DCSR = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>
}>

#eltwise_mult = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>  // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) *= X(i,j)"
}

//
// Integration test that lowers a kernel annotated as sparse to
// actual sparse code, initializes a matching sparse storage scheme
// from file, and runs the resulting code with the JIT compiler.
//
module {
  //
  // A kernel that multiplies a sparse matrix A with itself
  // in an element-wise fashion. In this operation, we have
  // a sparse tensor as output, but although the values of the
  // sparse tensor change, its nonzero structure remains the same.
  //
  func @kernel_eltwise_mult(%argx: tensor<?x?xf64, #DCSR> {linalg.inplaceable = true})
    -> tensor<?x?xf64, #DCSR> {
    %0 = linalg.generic #eltwise_mult
      outs(%argx: tensor<?x?xf64, #DCSR>) {
      ^bb(%x: f64):
        %0 = arith.mulf %x, %x : f64
        linalg.yield %0 : f64
    } -> tensor<?x?xf64, #DCSR>
    return %0 : tensor<?x?xf64, #DCSR>
  }

  func private @getTensorFilename(index) -> (!Filename)

  //
  // Main driver that reads matrix from file and calls the sparse kernel.
  //
  func @entry() {
    %d0 = arith.constant 0.0 : f64
    %c0 = arith.constant 0 : index

    // Read the sparse matrix from file, construct sparse storage.
    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)
    %x = sparse_tensor.new %fileName : !Filename to tensor<?x?xf64, #DCSR>

    // Call kernel.
    %0 = call @kernel_eltwise_mult(%x) : (tensor<?x?xf64, #DCSR>) -> tensor<?x?xf64, #DCSR>

    // Print the result for verification.
    //
    // CHECK: ( 1, 1.96, 4, 6.25, 9, 16.81, 16, 27.04, 25 )
    //
    %m = sparse_tensor.values %0 : tensor<?x?xf64, #DCSR> to memref<?xf64>
    %v = vector.transfer_read %m[%c0], %d0: memref<?xf64>, vector<9xf64>
    vector.print %v : vector<9xf64>

    // Release the resources.
    sparse_tensor.release %x : tensor<?x?xf64, #DCSR>

    return
  }
}
