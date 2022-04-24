// RUN: mlir-opt %s --sparse-compiler | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//
// Do the same run, but now with SIMDization as well. This should not change the outcome.
//
// RUN: mlir-opt %s --sparse-compiler="vectorization-strategy=any-storage-inner-loop vl=4" | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#SparseVector = #sparse_tensor.encoding<{
  dimLevelType = ["compressed"]
}>

#SparseMatrix = #sparse_tensor.encoding<{
  dimLevelType = ["compressed", "compressed"]
}>

#trait_1d = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "X(i) = a(i) op i"
}

#trait_2d = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(i,j) op i op j"
}

//
// Test with indices and sparse inputs. All outputs are dense.
//
module {

  //
  // Kernel that uses index in the index notation (conjunction).
  //
  func.func @sparse_index_1d_conj(%arga: tensor<8xi64, #SparseVector>) -> tensor<8xi64> {
    %init = linalg.init_tensor [8] : tensor<8xi64>
    %r = linalg.generic #trait_1d
        ins(%arga: tensor<8xi64, #SparseVector>)
       outs(%init: tensor<8xi64>) {
        ^bb(%a: i64, %x: i64):
          %i = linalg.index 0 : index
          %ii = arith.index_cast %i : index to i64
          %m1 = arith.muli %a, %ii : i64
          linalg.yield %m1 : i64
    } -> tensor<8xi64>
    return %r : tensor<8xi64>
  }

  //
  // Kernel that uses index in the index notation (disjunction).
  //
  func.func @sparse_index_1d_disj(%arga: tensor<8xi64, #SparseVector>) -> tensor<8xi64> {
    %init = linalg.init_tensor [8] : tensor<8xi64>
    %r = linalg.generic #trait_1d
        ins(%arga: tensor<8xi64, #SparseVector>)
       outs(%init: tensor<8xi64>) {
        ^bb(%a: i64, %x: i64):
          %i = linalg.index 0 : index
          %ii = arith.index_cast %i : index to i64
          %m1 = arith.addi %a, %ii : i64
          linalg.yield %m1 : i64
    } -> tensor<8xi64>
    return %r : tensor<8xi64>
  }

  //
  // Kernel that uses indices in the index notation (conjunction).
  //
  func.func @sparse_index_2d_conj(%arga: tensor<3x4xi64, #SparseMatrix>) -> tensor<3x4xi64> {
    %init = linalg.init_tensor [3,4] : tensor<3x4xi64>
    %r = linalg.generic #trait_2d
        ins(%arga: tensor<3x4xi64, #SparseMatrix>)
       outs(%init: tensor<3x4xi64>) {
        ^bb(%a: i64, %x: i64):
          %i = linalg.index 0 : index
          %j = linalg.index 1 : index
          %ii = arith.index_cast %i : index to i64
          %jj = arith.index_cast %j : index to i64
          %m1 = arith.muli %ii, %a : i64
          %m2 = arith.muli %jj, %m1 : i64
          linalg.yield %m2 : i64
    } -> tensor<3x4xi64>
    return %r : tensor<3x4xi64>
  }

  //
  // Kernel that uses indices in the index notation (disjunction).
  //
  func.func @sparse_index_2d_disj(%arga: tensor<3x4xi64, #SparseMatrix>) -> tensor<3x4xi64> {
    %init = linalg.init_tensor [3,4] : tensor<3x4xi64>
    %r = linalg.generic #trait_2d
        ins(%arga: tensor<3x4xi64, #SparseMatrix>)
       outs(%init: tensor<3x4xi64>) {
        ^bb(%a: i64, %x: i64):
          %i = linalg.index 0 : index
          %j = linalg.index 1 : index
          %ii = arith.index_cast %i : index to i64
          %jj = arith.index_cast %j : index to i64
          %m1 = arith.addi %ii, %a : i64
          %m2 = arith.addi %jj, %m1 : i64
          linalg.yield %m2 : i64
    } -> tensor<3x4xi64>
    return %r : tensor<3x4xi64>
  }

  //
  // Main driver.
  //
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %du = arith.constant -1 : i64

    // Setup input sparse vector.
    %v1 = arith.constant sparse<[[2], [4]], [ 10, 20]> : tensor<8xi64>
    %sv = sparse_tensor.convert %v1 : tensor<8xi64> to tensor<8xi64, #SparseVector>

    // Setup input "sparse" vector.
    %v2 = arith.constant dense<[ 1,  2,  4,  8,  16,  32,  64,  128 ]> : tensor<8xi64>
    %dv = sparse_tensor.convert %v2 : tensor<8xi64> to tensor<8xi64, #SparseVector>

    // Setup input sparse matrix.
    %m1 = arith.constant sparse<[[1,1], [2,3]], [10, 20]> : tensor<3x4xi64>
    %sm = sparse_tensor.convert %m1 : tensor<3x4xi64> to tensor<3x4xi64, #SparseMatrix>

    // Setup input "sparse" matrix.
    %m2 = arith.constant dense <[ [ 1,  1,  1,  1 ],
                                  [ 1,  2,  1,  1 ],
                                  [ 1,  1,  3,  4 ] ]> : tensor<3x4xi64>
    %dm = sparse_tensor.convert %m2 : tensor<3x4xi64> to tensor<3x4xi64, #SparseMatrix>

    // Call the kernels.
    %0 = call @sparse_index_1d_conj(%sv) : (tensor<8xi64, #SparseVector>) -> tensor<8xi64>
    %1 = call @sparse_index_1d_disj(%sv) : (tensor<8xi64, #SparseVector>) -> tensor<8xi64>
    %2 = call @sparse_index_1d_conj(%dv) : (tensor<8xi64, #SparseVector>) -> tensor<8xi64>
    %3 = call @sparse_index_1d_disj(%dv) : (tensor<8xi64, #SparseVector>) -> tensor<8xi64>
    %4 = call @sparse_index_2d_conj(%sm) : (tensor<3x4xi64, #SparseMatrix>) -> tensor<3x4xi64>
    %5 = call @sparse_index_2d_disj(%sm) : (tensor<3x4xi64, #SparseMatrix>) -> tensor<3x4xi64>
    %6 = call @sparse_index_2d_conj(%dm) : (tensor<3x4xi64, #SparseMatrix>) -> tensor<3x4xi64>
    %7 = call @sparse_index_2d_disj(%dm) : (tensor<3x4xi64, #SparseMatrix>) -> tensor<3x4xi64>

    // Get the backing buffers.
    %mem0 = bufferization.to_memref %0 : memref<8xi64>
    %mem1 = bufferization.to_memref %1 : memref<8xi64>
    %mem2 = bufferization.to_memref %2 : memref<8xi64>
    %mem3 = bufferization.to_memref %3 : memref<8xi64>
    %mem4 = bufferization.to_memref %4 : memref<3x4xi64>
    %mem5 = bufferization.to_memref %5 : memref<3x4xi64>
    %mem6 = bufferization.to_memref %6 : memref<3x4xi64>
    %mem7 = bufferization.to_memref %7 : memref<3x4xi64>

    //
    // Verify result.
    //
    // CHECK:      ( 0, 0, 20, 0, 80, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 12, 3, 24, 5, 6, 7 )
    // CHECK-NEXT: ( 0, 2, 8, 24, 64, 160, 384, 896 )
    // CHECK-NEXT: ( 1, 3, 6, 11, 20, 37, 70, 135 )
    // CHECK-NEXT: ( ( 0, 0, 0, 0 ), ( 0, 10, 0, 0 ), ( 0, 0, 0, 120 ) )
    // CHECK-NEXT: ( ( 0, 1, 2, 3 ), ( 1, 12, 3, 4 ), ( 2, 3, 4, 25 ) )
    // CHECK-NEXT: ( ( 0, 0, 0, 0 ), ( 0, 2, 2, 3 ), ( 0, 2, 12, 24 ) )
    // CHECK-NEXT: ( ( 1, 2, 3, 4 ), ( 2, 4, 4, 5 ), ( 3, 4, 7, 9 ) )
    //
    %vv0 = vector.transfer_read %mem0[%c0], %du: memref<8xi64>, vector<8xi64>
    %vv1 = vector.transfer_read %mem1[%c0], %du: memref<8xi64>, vector<8xi64>
    %vv2 = vector.transfer_read %mem2[%c0], %du: memref<8xi64>, vector<8xi64>
    %vv3 = vector.transfer_read %mem3[%c0], %du: memref<8xi64>, vector<8xi64>
    %vv4 = vector.transfer_read %mem4[%c0,%c0], %du: memref<3x4xi64>, vector<3x4xi64>
    %vv5 = vector.transfer_read %mem5[%c0,%c0], %du: memref<3x4xi64>, vector<3x4xi64>
    %vv6 = vector.transfer_read %mem6[%c0,%c0], %du: memref<3x4xi64>, vector<3x4xi64>
    %vv7 = vector.transfer_read %mem7[%c0,%c0], %du: memref<3x4xi64>, vector<3x4xi64>
    vector.print %vv0 : vector<8xi64>
    vector.print %vv1 : vector<8xi64>
    vector.print %vv2 : vector<8xi64>
    vector.print %vv3 : vector<8xi64>
    vector.print %vv4 : vector<3x4xi64>
    vector.print %vv5 : vector<3x4xi64>
    vector.print %vv6 : vector<3x4xi64>
    vector.print %vv7 : vector<3x4xi64>

    // Release resources.
    sparse_tensor.release %sv : tensor<8xi64, #SparseVector>
    sparse_tensor.release %dv : tensor<8xi64, #SparseVector>
    sparse_tensor.release %sm : tensor<3x4xi64, #SparseMatrix>
    sparse_tensor.release %dm : tensor<3x4xi64, #SparseMatrix>
    memref.dealloc %mem0 : memref<8xi64>
    memref.dealloc %mem1 : memref<8xi64>
    memref.dealloc %mem2 : memref<8xi64>
    memref.dealloc %mem3 : memref<8xi64>
    memref.dealloc %mem4 : memref<3x4xi64>
    memref.dealloc %mem5 : memref<3x4xi64>
    memref.dealloc %mem6 : memref<3x4xi64>
    memref.dealloc %mem7 : memref<3x4xi64>

    return
  }
}
