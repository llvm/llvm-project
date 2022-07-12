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

//
// Test with various forms of the two most elementary reshape
// operations: expand/collapse.
//
module {

  func.func @expand_dense(%arg0: tensor<12xf64>) -> tensor<3x4xf64> {
    %0 = tensor.expand_shape %arg0 [[0, 1]] : tensor<12xf64> into tensor<3x4xf64>
    return %0 : tensor<3x4xf64>
  }

  func.func @expand_from_sparse(%arg0: tensor<12xf64, #SparseVector>) -> tensor<3x4xf64> {
    %0 = tensor.expand_shape %arg0 [[0, 1]] : tensor<12xf64, #SparseVector> into tensor<3x4xf64>
    return %0 : tensor<3x4xf64>
  }

  func.func @expand_to_sparse(%arg0: tensor<12xf64>) -> tensor<3x4xf64, #SparseMatrix> {
    %0 = tensor.expand_shape %arg0 [[0, 1]] : tensor<12xf64> into tensor<3x4xf64, #SparseMatrix>
    return %0 : tensor<3x4xf64, #SparseMatrix>
  }

  func.func @expand_sparse2sparse(%arg0: tensor<12xf64, #SparseVector>) -> tensor<3x4xf64, #SparseMatrix> {
    %0 = tensor.expand_shape %arg0 [[0, 1]] : tensor<12xf64, #SparseVector> into tensor<3x4xf64, #SparseMatrix>
    return %0 : tensor<3x4xf64, #SparseMatrix>
  }

  func.func @collapse_dense(%arg0: tensor<3x4xf64>) -> tensor<12xf64> {
    %0 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<3x4xf64> into tensor<12xf64>
    return %0 : tensor<12xf64>
  }

  func.func @collapse_from_sparse(%arg0: tensor<3x4xf64, #SparseMatrix>) -> tensor<12xf64> {
    %0 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<3x4xf64, #SparseMatrix> into tensor<12xf64>
    return %0 : tensor<12xf64>
  }

  func.func @collapse_to_sparse(%arg0: tensor<3x4xf64>) -> tensor<12xf64, #SparseVector> {
    %0 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<3x4xf64> into tensor<12xf64, #SparseVector>
    return %0 : tensor<12xf64, #SparseVector>
  }

  func.func @collapse_sparse2sparse(%arg0: tensor<3x4xf64, #SparseMatrix>) -> tensor<12xf64, #SparseVector> {
    %0 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<3x4xf64, #SparseMatrix> into tensor<12xf64, #SparseVector>
    return %0 : tensor<12xf64, #SparseVector>
  }


  //
  // Main driver.
  //
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %df = arith.constant -1.0 : f64

    // Setup test vectors and matrices..
    %v = arith.constant dense <[ 1.0, 2.0, 3.0,  4.0,  5.0,  6.0,
                                 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]> : tensor<12xf64>
    %m = arith.constant dense <[ [ 1.1,  1.2,  1.3,  1.4 ],
                                 [ 2.1,  2.2,  2.3,  2.4 ],
                                 [ 3.1,  3.2,  3.3,  3.4 ]]> : tensor<3x4xf64>
    %sv = sparse_tensor.convert %v : tensor<12xf64> to tensor<12xf64, #SparseVector>
    %sm = sparse_tensor.convert %m : tensor<3x4xf64> to tensor<3x4xf64, #SparseMatrix>


    // Call the kernels.
    %expand0 = call @expand_dense(%v) : (tensor<12xf64>) -> tensor<3x4xf64>
    %expand1 = call @expand_from_sparse(%sv) : (tensor<12xf64, #SparseVector>) -> tensor<3x4xf64>
    %expand2 = call @expand_to_sparse(%v) : (tensor<12xf64>) -> tensor<3x4xf64, #SparseMatrix>
    %expand3 = call @expand_sparse2sparse(%sv) : (tensor<12xf64, #SparseVector>) -> tensor<3x4xf64, #SparseMatrix>

    %collapse0 = call @collapse_dense(%m) : (tensor<3x4xf64>) -> tensor<12xf64>
    %collapse1 = call @collapse_from_sparse(%sm) : (tensor<3x4xf64, #SparseMatrix>) -> tensor<12xf64>
    %collapse2 = call @collapse_to_sparse(%m) : (tensor<3x4xf64>) -> tensor<12xf64, #SparseVector>
    %collapse3 = call @collapse_sparse2sparse(%sm) : (tensor<3x4xf64, #SparseMatrix>) -> tensor<12xf64, #SparseVector>

    //
    // Verify result.
    //
    // CHECK:      ( ( 1, 2, 3, 4 ), ( 5, 6, 7, 8 ), ( 9, 10, 11, 12 ) )
    // CHECK-NEXT: ( ( 1, 2, 3, 4 ), ( 5, 6, 7, 8 ), ( 9, 10, 11, 12 ) )
    // CHECK-NEXT: ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1, -1, -1, -1 )
    // CHECK-NEXT: ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1, -1, -1, -1 )
    // CHECK-NEXT: ( 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4 )
    // CHECK-NEXT: ( 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4 )
    // CHECK-NEXT: ( 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, -1, -1, -1, -1 )
    // CHECK-NEXT: ( 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, -1, -1, -1, -1 )
    //
    %m0 = vector.transfer_read %expand0[%c0, %c0], %df: tensor<3x4xf64>, vector<3x4xf64>
    vector.print %m0 : vector<3x4xf64>
    %m1 = vector.transfer_read %expand1[%c0, %c0], %df: tensor<3x4xf64>, vector<3x4xf64>
    vector.print %m1 : vector<3x4xf64>
    %a2 = sparse_tensor.values %expand2 : tensor<3x4xf64, #SparseMatrix> to memref<?xf64>
    %m2 = vector.transfer_read %a2[%c0], %df: memref<?xf64>, vector<16xf64>
    vector.print %m2 : vector<16xf64>
    %a3 = sparse_tensor.values %expand3 : tensor<3x4xf64, #SparseMatrix> to memref<?xf64>
    %m3 = vector.transfer_read %a3[%c0], %df: memref<?xf64>, vector<16xf64>
    vector.print %m3 : vector<16xf64>

    %v0 = vector.transfer_read %collapse0[%c0], %df: tensor<12xf64>, vector<12xf64>
    vector.print %v0 : vector<12xf64>
    %v1 = vector.transfer_read %collapse1[%c0], %df: tensor<12xf64>, vector<12xf64>
    vector.print %v1 : vector<12xf64>
    %b2 = sparse_tensor.values %collapse2 : tensor<12xf64, #SparseVector> to memref<?xf64>
    %v2 = vector.transfer_read %b2[%c0], %df: memref<?xf64>, vector<16xf64>
    vector.print %v2 : vector<16xf64>
    %b3 = sparse_tensor.values %collapse3 : tensor<12xf64, #SparseVector> to memref<?xf64>
    %v3 = vector.transfer_read %b3[%c0], %df: memref<?xf64>, vector<16xf64>
    vector.print %v3 : vector<16xf64>

    // Release sparse resources.
    sparse_tensor.release %sv : tensor<12xf64, #SparseVector>
    sparse_tensor.release %sm : tensor<3x4xf64, #SparseMatrix>
    sparse_tensor.release %expand2 : tensor<3x4xf64, #SparseMatrix>
    sparse_tensor.release %expand3 : tensor<3x4xf64, #SparseMatrix>
    sparse_tensor.release %collapse2 : tensor<12xf64, #SparseVector>
    sparse_tensor.release %collapse3 : tensor<12xf64, #SparseVector>

    // Release dense resources.
    %meme1 = bufferization.to_memref %expand1 : memref<3x4xf64>
    memref.dealloc %meme1 : memref<3x4xf64>
    %memc1 = bufferization.to_memref %collapse1 : memref<12xf64>
    memref.dealloc %memc1 : memref<12xf64>

    return
  }
}
