// RUN: mlir-opt %s --sparse-compiler | \
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
// Test with indices. Note that a lot of results are actually
// dense, but this is done to stress test all the operations.
//
module {

  //
  // Kernel that uses index in the index notation (conjunction).
  //
  func.func @sparse_index_1d_conj(%arga: tensor<8xi64, #SparseVector>)
                                 -> tensor<8xi64, #SparseVector> {
    %init = bufferization.alloc_tensor() : tensor<8xi64, #SparseVector>
    %r = linalg.generic #trait_1d
        ins(%arga: tensor<8xi64, #SparseVector>)
       outs(%init: tensor<8xi64, #SparseVector>) {
        ^bb(%a: i64, %x: i64):
          %i = linalg.index 0 : index
          %ii = arith.index_cast %i : index to i64
          %m1 = arith.muli %a, %ii : i64
          linalg.yield %m1 : i64
    } -> tensor<8xi64, #SparseVector>
    return %r : tensor<8xi64, #SparseVector>
  }

  //
  // Kernel that uses index in the index notation (disjunction).
  //
  func.func @sparse_index_1d_disj(%arga: tensor<8xi64, #SparseVector>)
                                 -> tensor<8xi64, #SparseVector> {
    %init = bufferization.alloc_tensor() : tensor<8xi64, #SparseVector>
    %r = linalg.generic #trait_1d
        ins(%arga: tensor<8xi64, #SparseVector>)
       outs(%init: tensor<8xi64, #SparseVector>) {
        ^bb(%a: i64, %x: i64):
          %i = linalg.index 0 : index
          %ii = arith.index_cast %i : index to i64
          %m1 = arith.addi %a, %ii : i64
          linalg.yield %m1 : i64
    } -> tensor<8xi64, #SparseVector>
    return %r : tensor<8xi64, #SparseVector>
  }

  //
  // Kernel that uses indices in the index notation (conjunction).
  //
  func.func @sparse_index_2d_conj(%arga: tensor<3x4xi64, #SparseMatrix>)
                                 -> tensor<3x4xi64, #SparseMatrix> {
    %init = bufferization.alloc_tensor() : tensor<3x4xi64, #SparseMatrix>
    %r = linalg.generic #trait_2d
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
  // Kernel that uses indices in the index notation (disjunction).
  //
  func.func @sparse_index_2d_disj(%arga: tensor<3x4xi64, #SparseMatrix>)
                                 -> tensor<3x4xi64, #SparseMatrix> {
    %init = bufferization.alloc_tensor() : tensor<3x4xi64, #SparseMatrix>
    %r = linalg.generic #trait_2d
        ins(%arga: tensor<3x4xi64, #SparseMatrix>)
       outs(%init: tensor<3x4xi64, #SparseMatrix>) {
        ^bb(%a: i64, %x: i64):
          %i = linalg.index 0 : index
          %j = linalg.index 1 : index
          %ii = arith.index_cast %i : index to i64
          %jj = arith.index_cast %j : index to i64
          %m1 = arith.addi %ii, %a : i64
          %m2 = arith.addi %jj, %m1 : i64
          linalg.yield %m2 : i64
    } -> tensor<3x4xi64, #SparseMatrix>
    return %r : tensor<3x4xi64, #SparseMatrix>
  }

  func.func @add_outer_2d(%arg0: tensor<2x3xf32, #SparseMatrix>)
                         -> tensor<2x3xf32, #SparseMatrix> {
    %0 = bufferization.alloc_tensor() : tensor<2x3xf32, #SparseMatrix>
    %1 = linalg.generic #trait_2d
      ins(%arg0 : tensor<2x3xf32, #SparseMatrix>)
      outs(%0 : tensor<2x3xf32, #SparseMatrix>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %2 = linalg.index 0 : index
      %3 = arith.index_cast %2 : index to i64
      %4 = arith.uitofp %3 : i64 to f32
      %5 = arith.addf %arg1, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<2x3xf32, #SparseMatrix>
    return %1 : tensor<2x3xf32, #SparseMatrix>
  }

  //
  // Main driver.
  //
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %du = arith.constant -1 : i64
    %df = arith.constant -1.0 : f32

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

    // Setup input sparse f32 matrix.
    %mf32 = arith.constant sparse<[[0,1], [1,2]], [10.0, 41.0]> : tensor<2x3xf32>
    %sf32 = sparse_tensor.convert %mf32 : tensor<2x3xf32> to tensor<2x3xf32, #SparseMatrix>

    // Call the kernels.
    %0 = call @sparse_index_1d_conj(%sv) : (tensor<8xi64, #SparseVector>)
      -> tensor<8xi64, #SparseVector>
    %1 = call @sparse_index_1d_disj(%sv) : (tensor<8xi64, #SparseVector>)
      -> tensor<8xi64, #SparseVector>
    %2 = call @sparse_index_1d_conj(%dv) : (tensor<8xi64, #SparseVector>)
      -> tensor<8xi64, #SparseVector>
    %3 = call @sparse_index_1d_disj(%dv) : (tensor<8xi64, #SparseVector>)
      -> tensor<8xi64, #SparseVector>
    %4 = call @sparse_index_2d_conj(%sm) : (tensor<3x4xi64, #SparseMatrix>)
      -> tensor<3x4xi64, #SparseMatrix>
    %5 = call @sparse_index_2d_disj(%sm) : (tensor<3x4xi64, #SparseMatrix>)
      -> tensor<3x4xi64, #SparseMatrix>
    %6 = call @sparse_index_2d_conj(%dm) : (tensor<3x4xi64, #SparseMatrix>)
      -> tensor<3x4xi64, #SparseMatrix>
    %7 = call @sparse_index_2d_disj(%dm) : (tensor<3x4xi64, #SparseMatrix>)
      -> tensor<3x4xi64, #SparseMatrix>

    //
    // Verify result.
    //
    // CHECK:      ( 20, 80, -1, -1, -1, -1, -1, -1 )
    // CHECK-NEXT: ( 0, 1, 12, 3, 24, 5, 6, 7 )
    // CHECK-NEXT: ( 0, 2, 8, 24, 64, 160, 384, 896 )
    // CHECK-NEXT: ( 1, 3, 6, 11, 20, 37, 70, 135 )
    // CHECK-NEXT: ( 10, 120, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 )
    // CHECK-NEXT: ( 0, 1, 2, 3, 1, 12, 3, 4, 2, 3, 4, 25 )
    // CHECK-NEXT: ( 0, 0, 0, 0, 0, 2, 2, 3, 0, 2, 12, 24 )
    // CHECK-NEXT: ( 1, 2, 3, 4, 2, 4, 4, 5, 3, 4, 7, 9 )
    //
    %8 = sparse_tensor.values %0 : tensor<8xi64, #SparseVector> to memref<?xi64>
    %9 = sparse_tensor.values %1 : tensor<8xi64, #SparseVector> to memref<?xi64>
    %10 = sparse_tensor.values %2 : tensor<8xi64, #SparseVector> to memref<?xi64>
    %11 = sparse_tensor.values %3 : tensor<8xi64, #SparseVector> to memref<?xi64>
    %12 = sparse_tensor.values %4 : tensor<3x4xi64, #SparseMatrix> to memref<?xi64>
    %13 = sparse_tensor.values %5 : tensor<3x4xi64, #SparseMatrix> to memref<?xi64>
    %14 = sparse_tensor.values %6 : tensor<3x4xi64, #SparseMatrix> to memref<?xi64>
    %15 = sparse_tensor.values %7 : tensor<3x4xi64, #SparseMatrix> to memref<?xi64>
    %16 = vector.transfer_read %8[%c0], %du: memref<?xi64>, vector<8xi64>
    %17 = vector.transfer_read %9[%c0], %du: memref<?xi64>, vector<8xi64>
    %18 = vector.transfer_read %10[%c0], %du: memref<?xi64>, vector<8xi64>
    %19 = vector.transfer_read %11[%c0], %du: memref<?xi64>, vector<8xi64>
    %20 = vector.transfer_read %12[%c0], %du: memref<?xi64>, vector<12xi64>
    %21 = vector.transfer_read %13[%c0], %du: memref<?xi64>, vector<12xi64>
    %22 = vector.transfer_read %14[%c0], %du: memref<?xi64>, vector<12xi64>
    %23 = vector.transfer_read %15[%c0], %du: memref<?xi64>, vector<12xi64>
    vector.print %16 : vector<8xi64>
    vector.print %17 : vector<8xi64>
    vector.print %18 : vector<8xi64>
    vector.print %19 : vector<8xi64>
    vector.print %20 : vector<12xi64>
    vector.print %21 : vector<12xi64>
    vector.print %22 : vector<12xi64>
    vector.print %23 : vector<12xi64>

    // Release resources.
    bufferization.dealloc_tensor %sv : tensor<8xi64, #SparseVector>
    bufferization.dealloc_tensor %dv : tensor<8xi64, #SparseVector>
    bufferization.dealloc_tensor %0 : tensor<8xi64, #SparseVector>
    bufferization.dealloc_tensor %1 : tensor<8xi64, #SparseVector>
    bufferization.dealloc_tensor %2 : tensor<8xi64, #SparseVector>
    bufferization.dealloc_tensor %3 : tensor<8xi64, #SparseVector>
    bufferization.dealloc_tensor %sm : tensor<3x4xi64, #SparseMatrix>
    bufferization.dealloc_tensor %dm : tensor<3x4xi64, #SparseMatrix>
    bufferization.dealloc_tensor %4 : tensor<3x4xi64, #SparseMatrix>
    bufferization.dealloc_tensor %5 : tensor<3x4xi64, #SparseMatrix>
    bufferization.dealloc_tensor %6 : tensor<3x4xi64, #SparseMatrix>
    bufferization.dealloc_tensor %7 : tensor<3x4xi64, #SparseMatrix>

    //
    // Call the f32 kernel, verify the result, release the resources.
    //
    // CHECK-NEXT: ( 0, 10, 0, 1, 1, 42 )
    //
    %100 = call @add_outer_2d(%sf32) : (tensor<2x3xf32, #SparseMatrix>)
      -> tensor<2x3xf32, #SparseMatrix>
    %101 = sparse_tensor.values %100 : tensor<2x3xf32, #SparseMatrix> to memref<?xf32>
    %102 = vector.transfer_read %101[%c0], %df: memref<?xf32>, vector<6xf32>
    vector.print %102 : vector<6xf32>
    bufferization.dealloc_tensor %sf32 : tensor<2x3xf32, #SparseMatrix>
    bufferization.dealloc_tensor %100 : tensor<2x3xf32, #SparseMatrix>

    return
  }
}
