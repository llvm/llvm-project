// RUN: mlir-opt %s --sparse-compiler | \
// RUN: TENSOR0="%mlir_integration_test_dir/data/test.mtx" \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

!Filename = !llvm.ptr<i8>

#DenseMatrix = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "dense" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>
}>

#SparseMatrix = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>
}>

#trait_assign = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>, // A
    affine_map<(i,j) -> (i,j)>  // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(i,j)"
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
  // A kernel that assigns elements from A to X.
  //
  func.func @dense_output(%arga: tensor<?x?xf64, #SparseMatrix>) -> tensor<?x?xf64, #DenseMatrix> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = tensor.dim %arga, %c0 : tensor<?x?xf64, #SparseMatrix>
    %d1 = tensor.dim %arga, %c1 : tensor<?x?xf64, #SparseMatrix>
    %init = bufferization.alloc_tensor(%d0, %d1) : tensor<?x?xf64, #DenseMatrix>
    %0 = linalg.generic #trait_assign
       ins(%arga: tensor<?x?xf64, #SparseMatrix>)
      outs(%init: tensor<?x?xf64, #DenseMatrix>) {
      ^bb(%a: f64, %x: f64):
        linalg.yield %a : f64
    } -> tensor<?x?xf64, #DenseMatrix>
    return %0 : tensor<?x?xf64, #DenseMatrix>
  }

  func.func private @getTensorFilename(index) -> (!Filename)

  //
  // Main driver that reads matrix from file and calls the kernel.
  //
  func.func @entry() {
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
    // CHECK: ( 1, 0, 0, 1.4, 0, 0, 2, 0, 0, 2.5, 0, 0, 3, 0, 0, 4.1, 0, 0, 4, 0, 0, 5.2, 0, 0, 5 )
    //
    %m = sparse_tensor.values %0
      : tensor<?x?xf64, #DenseMatrix> to memref<?xf64>
    %v = vector.load %m[%c0] : memref<?xf64>, vector<25xf64>
    vector.print %v : vector<25xf64>

    // Release the resources.
    sparse_tensor.release %a : tensor<?x?xf64, #SparseMatrix>
    sparse_tensor.release %0 : tensor<?x?xf64, #DenseMatrix>

    return
  }
}
