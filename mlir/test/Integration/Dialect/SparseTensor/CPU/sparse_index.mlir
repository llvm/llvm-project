// RUN: mlir-opt %s --sparse-compiler | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#SparseMatrix = #sparse_tensor.encoding<{
  dimLevelType = ["compressed", "compressed"]
}>

#trait = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(i,j) * i * j"
}

module {

  //
  // Kernel that uses indices in the index notation.
  //
  func @sparse_index(%arga: tensor<3x4xi64, #SparseMatrix>)
                         -> tensor<3x4xi64, #SparseMatrix> {
    %d0 = arith.constant 3 : index
    %d1 = arith.constant 4 : index
    %init = sparse_tensor.init [%d0, %d1] : tensor<3x4xi64, #SparseMatrix>
    %r = linalg.generic #trait
        ins(%arga: tensor<3x4xi64, #SparseMatrix>)
       outs(%init: tensor<3x4xi64, #SparseMatrix>) {
        ^bb(%a: i64, %x: i64):
          %i = linalg.index 0 : index
          %j = linalg.index 1 : index
          %ii = arith.index_cast %i : index to i64
          %jj = arith.index_cast %j : index to i64
          %m1 = arith.muli %ii, %a : i64
          %m2 = arith.muli %jj, %m1 : i64
          linalg.yield %m2 : i64
    } -> tensor<3x4xi64, #SparseMatrix>
    return %r : tensor<3x4xi64, #SparseMatrix>
  }

  //
  // Main driver.
  //
  func @entry() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %du = arith.constant -1 : i64

    // Setup input "sparse" matrix.
    %d = arith.constant dense <[
       [ 1,  1,  1,  1 ],
       [ 1,  1,  1,  1 ],
       [ 1,  1,  1,  1 ]
    ]> : tensor<3x4xi64>
    %a = sparse_tensor.convert %d : tensor<3x4xi64> to tensor<3x4xi64, #SparseMatrix>

    // Call the kernel.
    %0 = call @sparse_index(%a) : (tensor<3x4xi64, #SparseMatrix>) -> tensor<3x4xi64, #SparseMatrix>

    //
    // Verify result.
    //
    // CHECK: ( ( 0, 0, 0, 0 ), ( 0, 1, 2, 3 ), ( 0, 2, 4, 6 ) )
    //
    %x = sparse_tensor.convert %0 : tensor<3x4xi64, #SparseMatrix> to tensor<3x4xi64>
    %m = bufferization.to_memref %x : memref<3x4xi64>
    %v = vector.transfer_read %m[%c0, %c0], %du: memref<3x4xi64>, vector<3x4xi64>
    vector.print %v : vector<3x4xi64>

    // Release resources.
    sparse_tensor.release %a : tensor<3x4xi64, #SparseMatrix>
    sparse_tensor.release %0 : tensor<3x4xi64, #SparseMatrix>
    memref.dealloc %m : memref<3x4xi64>

    return
  }
}
