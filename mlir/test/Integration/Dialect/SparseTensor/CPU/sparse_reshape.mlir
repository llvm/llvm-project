// RUN: mlir-opt %s --sparse-compiler | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#SparseVector = #sparse_tensor.encoding<{
  dimLevelType = ["compressed"]
}>

#SparseMatrix = #sparse_tensor.encoding<{
  dimLevelType = ["compressed", "compressed"]
}>

#Sparse3dTensor = #sparse_tensor.encoding<{
  dimLevelType = ["compressed", "compressed", "compressed"]
}>

#Sparse4dTensor = #sparse_tensor.encoding<{
  dimLevelType = ["compressed", "compressed", "compressed", "compressed"]
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

  func.func @expand_dense_3x2x2(%arg0: tensor<3x4xf64>) -> tensor<3x2x2xf64> {
    %0 = tensor.expand_shape %arg0 [[0], [1, 2]] : tensor<3x4xf64> into tensor<3x2x2xf64>
    return %0 : tensor<3x2x2xf64>
  }

  func.func @expand_from_sparse_3x2x2(%arg0: tensor<3x4xf64, #SparseMatrix>) -> tensor<3x2x2xf64> {
    %0 = tensor.expand_shape %arg0 [[0], [1, 2]] : tensor<3x4xf64, #SparseMatrix> into tensor<3x2x2xf64>
    return %0 : tensor<3x2x2xf64>
  }

  func.func @expand_to_sparse_3x2x2(%arg0: tensor<3x4xf64>) -> tensor<3x2x2xf64, #Sparse3dTensor> {
    %0 = tensor.expand_shape %arg0 [[0], [1, 2]] : tensor<3x4xf64> into tensor<3x2x2xf64, #Sparse3dTensor>
    return %0 : tensor<3x2x2xf64, #Sparse3dTensor>
  }

  func.func @expand_sparse2sparse_3x2x2(%arg0: tensor<3x4xf64, #SparseMatrix>) -> tensor<3x2x2xf64, #Sparse3dTensor> {
    %0 = tensor.expand_shape %arg0 [[0], [1, 2]] : tensor<3x4xf64, #SparseMatrix> into tensor<3x2x2xf64, #Sparse3dTensor>
    return %0 : tensor<3x2x2xf64, #Sparse3dTensor>
  }

  func.func @collapse_dense_6x10(%arg0: tensor<2x3x5x2xf64>) -> tensor<6x10xf64> {
    %0 = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<2x3x5x2xf64> into tensor<6x10xf64>
    return %0 : tensor<6x10xf64>
  }

  func.func @collapse_from_sparse_6x10(%arg0: tensor<2x3x5x2xf64, #Sparse4dTensor>) -> tensor<6x10xf64> {
    %0 = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<2x3x5x2xf64, #Sparse4dTensor> into tensor<6x10xf64>
    return %0 : tensor<6x10xf64>
  }

  func.func @collapse_to_sparse_6x10(%arg0: tensor<2x3x5x2xf64>) -> tensor<6x10xf64, #SparseMatrix> {
    %0 = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<2x3x5x2xf64> into tensor<6x10xf64, #SparseMatrix>
    return %0 : tensor<6x10xf64, #SparseMatrix>
  }

  func.func @collapse_sparse2sparse_6x10(%arg0: tensor<2x3x5x2xf64, #Sparse4dTensor>) -> tensor<6x10xf64, #SparseMatrix> {
    %0 = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<2x3x5x2xf64, #Sparse4dTensor> into tensor<6x10xf64, #SparseMatrix>
    return %0 : tensor<6x10xf64, #SparseMatrix>
  }

  func.func @expand_dense_dyn(%arg0: tensor<?x?xf64>) -> tensor<?x2x?xf64> {
    %0 = tensor.expand_shape %arg0 [[0], [1, 2]] : tensor<?x?xf64> into tensor<?x2x?xf64>
    return %0 : tensor<?x2x?xf64>
  }

  func.func @expand_from_sparse_dyn(%arg0: tensor<?x?xf64, #SparseMatrix>) -> tensor<?x2x?xf64> {
    %0 = tensor.expand_shape %arg0 [[0], [1, 2]] : tensor<?x?xf64, #SparseMatrix> into tensor<?x2x?xf64>
    return %0 : tensor<?x2x?xf64>
  }

  func.func @expand_to_sparse_dyn(%arg0: tensor<?x?xf64>) -> tensor<?x2x?xf64, #Sparse3dTensor> {
    %0 = tensor.expand_shape %arg0 [[0], [1, 2]] : tensor<?x?xf64> into tensor<?x2x?xf64, #Sparse3dTensor>
    return %0 : tensor<?x2x?xf64, #Sparse3dTensor>
  }

  func.func @expand_sparse2sparse_dyn(%arg0: tensor<?x?xf64, #SparseMatrix>) -> tensor<?x2x?xf64, #Sparse3dTensor> {
    %0 = tensor.expand_shape %arg0 [[0], [1, 2]] : tensor<?x?xf64, #SparseMatrix> into tensor<?x2x?xf64, #Sparse3dTensor>
    return %0 : tensor<?x2x?xf64, #Sparse3dTensor>
  }

  func.func @collapse_dense_dyn(%arg0: tensor<?x?x?x?xf64>) -> tensor<?x?xf64> {
    %0 = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<?x?x?x?xf64> into tensor<?x?xf64>
    return %0 : tensor<?x?xf64>
  }

  func.func @collapse_from_sparse_dyn(%arg0: tensor<?x?x?x?xf64, #Sparse4dTensor>) -> tensor<?x?xf64> {
    %0 = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<?x?x?x?xf64, #Sparse4dTensor> into tensor<?x?xf64>
    return %0 : tensor<?x?xf64>
  }

  func.func @collapse_to_sparse_dyn(%arg0: tensor<?x?x?x?xf64>) -> tensor<?x?xf64, #SparseMatrix> {
    %0 = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<?x?x?x?xf64> into tensor<?x?xf64, #SparseMatrix>
    return %0 : tensor<?x?xf64, #SparseMatrix>
  }

  func.func @collapse_sparse2sparse_dyn(%arg0: tensor<?x?x?x?xf64, #Sparse4dTensor>) -> tensor<?x?xf64, #SparseMatrix> {
    %0 = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<?x?x?x?xf64, #Sparse4dTensor> into tensor<?x?xf64, #SparseMatrix>
    return %0 : tensor<?x?xf64, #SparseMatrix>
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
    %n = arith.constant dense <[ 
      [ [[1.0,  2.0],  [3.0,  4.0],  [5.0,  6.0],  [7.0,  8.0],  [9.0, 10.0]],
        [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0], [19.0, 20.0]],
        [[21.0, 22.0], [23.0, 24.0], [25.0, 26.0], [27.0, 28.0], [29.0, 30.0]] ],
      [ [[31.0, 32.0], [33.0, 34.0], [35.0, 36.0], [37.0, 38.0], [39.0, 40.0]],
        [[41.0, 42.0], [43.0, 44.0], [45.0, 26.0], [47.0, 48.0], [49.0, 50.0]],
        [[51.0, 52.0], [53.0, 54.0], [55.0, 56.0], [57.0, 58.0], [59.0, 60.0]] ] ]> : tensor<2x3x5x2xf64>
    %sv = sparse_tensor.convert %v : tensor<12xf64> to tensor<12xf64, #SparseVector>
    %sm = sparse_tensor.convert %m : tensor<3x4xf64> to tensor<3x4xf64, #SparseMatrix>
    %sn = sparse_tensor.convert %n : tensor<2x3x5x2xf64> to tensor<2x3x5x2xf64, #Sparse4dTensor>

    %dm = tensor.cast %m : tensor<3x4xf64> to tensor<?x?xf64>
    %sdm = sparse_tensor.convert %dm : tensor<?x?xf64> to tensor<?x?xf64, #SparseMatrix>

    %dn = tensor.cast %n : tensor<2x3x5x2xf64> to tensor<?x?x?x?xf64>
    %sdn = sparse_tensor.convert %dn : tensor<?x?x?x?xf64> to tensor<?x?x?x?xf64, #Sparse4dTensor>

    // Call the kernels.
    %expand0 = call @expand_dense(%v) : (tensor<12xf64>) -> tensor<3x4xf64>
    %expand1 = call @expand_from_sparse(%sv) : (tensor<12xf64, #SparseVector>) -> tensor<3x4xf64>
    %expand2 = call @expand_to_sparse(%v) : (tensor<12xf64>) -> tensor<3x4xf64, #SparseMatrix>
    %expand3 = call @expand_sparse2sparse(%sv) : (tensor<12xf64, #SparseVector>) -> tensor<3x4xf64, #SparseMatrix>
    %expand4 = call @expand_dense_3x2x2(%m) : (tensor<3x4xf64>) -> tensor<3x2x2xf64>
    %expand5 = call @expand_from_sparse_3x2x2(%sm) : (tensor<3x4xf64, #SparseMatrix>) -> tensor<3x2x2xf64>
    %expand6 = call @expand_to_sparse_3x2x2(%m) : (tensor<3x4xf64>) -> tensor<3x2x2xf64, #Sparse3dTensor>
    %expand7 = call @expand_sparse2sparse_3x2x2(%sm) : (tensor<3x4xf64, #SparseMatrix>) -> tensor<3x2x2xf64, #Sparse3dTensor>
    %expand8 = call @expand_dense_dyn(%dm) : (tensor<?x?xf64>) -> tensor<?x2x?xf64>
    %expand9 = call @expand_from_sparse_dyn(%sdm) : (tensor<?x?xf64, #SparseMatrix>) -> tensor<?x2x?xf64>
    %expand10 = call @expand_to_sparse_dyn(%dm) : (tensor<?x?xf64>) -> tensor<?x2x?xf64, #Sparse3dTensor>
    %expand11 = call @expand_sparse2sparse_dyn(%sdm) : (tensor<?x?xf64, #SparseMatrix>) -> tensor<?x2x?xf64, #Sparse3dTensor>

    %collapse0 = call @collapse_dense(%m) : (tensor<3x4xf64>) -> tensor<12xf64>
    %collapse1 = call @collapse_from_sparse(%sm) : (tensor<3x4xf64, #SparseMatrix>) -> tensor<12xf64>
    %collapse2 = call @collapse_to_sparse(%m) : (tensor<3x4xf64>) -> tensor<12xf64, #SparseVector>
    %collapse3 = call @collapse_sparse2sparse(%sm) : (tensor<3x4xf64, #SparseMatrix>) -> tensor<12xf64, #SparseVector>
    %collapse4 = call @collapse_dense_6x10(%n) : (tensor<2x3x5x2xf64>) -> tensor<6x10xf64>
    %collapse5 = call @collapse_from_sparse_6x10(%sn) : (tensor<2x3x5x2xf64, #Sparse4dTensor>) -> tensor<6x10xf64>
    %collapse6 = call @collapse_to_sparse_6x10(%n) : (tensor<2x3x5x2xf64>) -> tensor<6x10xf64, #SparseMatrix>
    %collapse7 = call @collapse_sparse2sparse_6x10(%sn) : (tensor<2x3x5x2xf64, #Sparse4dTensor>) -> tensor<6x10xf64, #SparseMatrix>
    %collapse8 = call @collapse_dense_dyn(%dn) : (tensor<?x?x?x?xf64>) -> tensor<?x?xf64>
    %collapse9 = call @collapse_from_sparse_dyn(%sdn) : (tensor<?x?x?x?xf64, #Sparse4dTensor>) -> tensor<?x?xf64>
    %collapse10 = call @collapse_to_sparse_dyn(%dn) : (tensor<?x?x?x?xf64>) -> tensor<?x?xf64, #SparseMatrix>
    %collapse11 = call @collapse_sparse2sparse_dyn(%sdn) : (tensor<?x?x?x?xf64, #Sparse4dTensor>) -> tensor<?x?xf64, #SparseMatrix>

    //
    // Verify results of expand
    //
    // CHECK:      ( ( 1, 2, 3, 4 ), ( 5, 6, 7, 8 ), ( 9, 10, 11, 12 ) )
    // CHECK-NEXT: ( ( 1, 2, 3, 4 ), ( 5, 6, 7, 8 ), ( 9, 10, 11, 12 ) )
    // CHECK-NEXT: ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1, -1, -1, -1 )
    // CHECK-NEXT: ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1, -1, -1, -1 )
    // CHECK-NEXT: ( ( ( 1.1, 1.2 ), ( 1.3, 1.4 ) ), ( ( 2.1, 2.2 ), ( 2.3, 2.4 ) ), ( ( 3.1, 3.2 ), ( 3.3, 3.4 ) ) )
    // CHECK-NEXT: ( ( ( 1.1, 1.2 ), ( 1.3, 1.4 ) ), ( ( 2.1, 2.2 ), ( 2.3, 2.4 ) ), ( ( 3.1, 3.2 ), ( 3.3, 3.4 ) ) )
    // CHECK-NEXT: ( 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, -1, -1, -1, -1 )
    // CHECK-NEXT: ( 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, -1, -1, -1, -1 )
    // CHECK-NEXT: ( ( ( 1.1, 1.2 ), ( 1.3, 1.4 ) ), ( ( 2.1, 2.2 ), ( 2.3, 2.4 ) ), ( ( 3.1, 3.2 ), ( 3.3, 3.4 ) ) )
    // CHECK-NEXT: ( ( ( 1.1, 1.2 ), ( 1.3, 1.4 ) ), ( ( 2.1, 2.2 ), ( 2.3, 2.4 ) ), ( ( 3.1, 3.2 ), ( 3.3, 3.4 ) ) )
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

    %m4 = vector.transfer_read %expand4[%c0, %c0, %c0], %df: tensor<3x2x2xf64>, vector<3x2x2xf64>
    vector.print %m4 : vector<3x2x2xf64>
    %m5 = vector.transfer_read %expand5[%c0, %c0, %c0], %df: tensor<3x2x2xf64>, vector<3x2x2xf64>
    vector.print %m5 : vector<3x2x2xf64>
    %a6 = sparse_tensor.values %expand6 : tensor<3x2x2xf64, #Sparse3dTensor> to memref<?xf64>
    %m6 = vector.transfer_read %a6[%c0], %df: memref<?xf64>, vector<16xf64>
    vector.print %m6 : vector<16xf64>
    %a7 = sparse_tensor.values %expand7 : tensor<3x2x2xf64, #Sparse3dTensor> to memref<?xf64>
    %m7 = vector.transfer_read %a7[%c0], %df: memref<?xf64>, vector<16xf64>
    vector.print %m7 : vector<16xf64>

    %m8 = vector.transfer_read %expand8[%c0, %c0, %c0], %df: tensor<?x2x?xf64>, vector<3x2x2xf64>
    vector.print %m8 : vector<3x2x2xf64>
    %m9 = vector.transfer_read %expand9[%c0, %c0, %c0], %df: tensor<?x2x?xf64>, vector<3x2x2xf64>
    vector.print %m9 : vector<3x2x2xf64>
    %a10 = sparse_tensor.values %expand10 : tensor<?x2x?xf64, #Sparse3dTensor> to memref<?xf64>
    %m10 = vector.transfer_read %a10[%c0], %df: memref<?xf64>, vector<16xf64>
    vector.print %m10 : vector<16xf64>
    %a11 = sparse_tensor.values %expand11 : tensor<?x2x?xf64, #Sparse3dTensor> to memref<?xf64>
    %m11 = vector.transfer_read %a11[%c0], %df: memref<?xf64>, vector<16xf64>
    vector.print %m11 : vector<16xf64>


    // 
    // Verify results of collapse
    // 
    // CHECK-NEXT: ( 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4 )
    // CHECK-NEXT: ( 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4 )
    // CHECK-NEXT: ( 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, -1, -1, -1, -1 )
    // CHECK-NEXT: ( 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, -1, -1, -1, -1 )
    // CHECK-NEXT: ( ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ), ( 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 ), ( 21, 22, 23, 24, 25, 26, 27, 28, 29, 30 ), ( 31, 32, 33, 34, 35, 36, 37, 38, 39, 40 ), ( 41, 42, 43, 44, 45, 26, 47, 48, 49, 50 ), ( 51, 52, 53, 54, 55, 56, 57, 58, 59, 60 ) )
    // CHECK-NEXT: ( ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ), ( 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 ), ( 21, 22, 23, 24, 25, 26, 27, 28, 29, 30 ), ( 31, 32, 33, 34, 35, 36, 37, 38, 39, 40 ), ( 41, 42, 43, 44, 45, 26, 47, 48, 49, 50 ), ( 51, 52, 53, 54, 55, 56, 57, 58, 59, 60 ) )
    // CHECK-NEXT: ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 26, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, -1, -1, -1, -1 )
    // CHECK-NEXT: ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 26, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, -1, -1, -1, -1 )
    // CHECK-NEXT: ( ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ), ( 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 ), ( 21, 22, 23, 24, 25, 26, 27, 28, 29, 30 ), ( 31, 32, 33, 34, 35, 36, 37, 38, 39, 40 ), ( 41, 42, 43, 44, 45, 26, 47, 48, 49, 50 ), ( 51, 52, 53, 54, 55, 56, 57, 58, 59, 60 ) )
    // CHECK-NEXT: ( ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ), ( 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 ), ( 21, 22, 23, 24, 25, 26, 27, 28, 29, 30 ), ( 31, 32, 33, 34, 35, 36, 37, 38, 39, 40 ), ( 41, 42, 43, 44, 45, 26, 47, 48, 49, 50 ), ( 51, 52, 53, 54, 55, 56, 57, 58, 59, 60 ) )
    // CHECK-NEXT: ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 26, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, -1, -1, -1, -1 )
    // CHECK-NEXT: ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 26, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, -1, -1, -1, -1 )
    //

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

    %v4 = vector.transfer_read %collapse4[%c0, %c0], %df: tensor<6x10xf64>, vector<6x10xf64>
    vector.print %v4 : vector<6x10xf64>
    %v5 = vector.transfer_read %collapse5[%c0, %c0], %df: tensor<6x10xf64>, vector<6x10xf64>
    vector.print %v5 : vector<6x10xf64>
    %b6 = sparse_tensor.values %collapse6 : tensor<6x10xf64, #SparseMatrix> to memref<?xf64>
    %v6 = vector.transfer_read %b6[%c0], %df: memref<?xf64>, vector<64xf64>
    vector.print %v6 : vector<64xf64>
    %b7 = sparse_tensor.values %collapse7 : tensor<6x10xf64, #SparseMatrix> to memref<?xf64>
    %v7 = vector.transfer_read %b7[%c0], %df: memref<?xf64>, vector<64xf64>
    vector.print %v7 : vector<64xf64>

    %v8 = vector.transfer_read %collapse8[%c0, %c0], %df: tensor<?x?xf64>, vector<6x10xf64>
    vector.print %v8 : vector<6x10xf64>
    %v9 = vector.transfer_read %collapse9[%c0, %c0], %df: tensor<?x?xf64>, vector<6x10xf64>
    vector.print %v9 : vector<6x10xf64>
    %b10 = sparse_tensor.values %collapse10 : tensor<?x?xf64, #SparseMatrix> to memref<?xf64>
    %v10 = vector.transfer_read %b10[%c0], %df: memref<?xf64>, vector<64xf64>
    vector.print %v10 : vector<64xf64>
    %b11 = sparse_tensor.values %collapse11 : tensor<?x?xf64, #SparseMatrix> to memref<?xf64>
    %v11 = vector.transfer_read %b11[%c0], %df: memref<?xf64>, vector<64xf64>
    vector.print %v11 : vector<64xf64>


    // Release sparse resources.
    bufferization.dealloc_tensor %sv : tensor<12xf64, #SparseVector>
    bufferization.dealloc_tensor %sm : tensor<3x4xf64, #SparseMatrix>
    bufferization.dealloc_tensor %sn : tensor<2x3x5x2xf64, #Sparse4dTensor>
    bufferization.dealloc_tensor %sdm : tensor<?x?xf64, #SparseMatrix>
    bufferization.dealloc_tensor %sdn : tensor<?x?x?x?xf64, #Sparse4dTensor>
    bufferization.dealloc_tensor %expand2 : tensor<3x4xf64, #SparseMatrix>
    bufferization.dealloc_tensor %expand3 : tensor<3x4xf64, #SparseMatrix>
    bufferization.dealloc_tensor %expand6 : tensor<3x2x2xf64, #Sparse3dTensor>
    bufferization.dealloc_tensor %expand7 : tensor<3x2x2xf64, #Sparse3dTensor>
    bufferization.dealloc_tensor %expand10 : tensor<?x2x?xf64, #Sparse3dTensor>
    bufferization.dealloc_tensor %expand11 : tensor<?x2x?xf64, #Sparse3dTensor>
    bufferization.dealloc_tensor %collapse2 : tensor<12xf64, #SparseVector>
    bufferization.dealloc_tensor %collapse3 : tensor<12xf64, #SparseVector>
    bufferization.dealloc_tensor %collapse6 : tensor<6x10xf64, #SparseMatrix>
    bufferization.dealloc_tensor %collapse7 : tensor<6x10xf64, #SparseMatrix>
    bufferization.dealloc_tensor %collapse10 : tensor<?x?xf64, #SparseMatrix>
    bufferization.dealloc_tensor %collapse11 : tensor<?x?xf64, #SparseMatrix>

    // Release dense resources.
    bufferization.dealloc_tensor %expand1 : tensor<3x4xf64>
    bufferization.dealloc_tensor %collapse1 : tensor<12xf64>
    bufferization.dealloc_tensor %expand5 : tensor<3x2x2xf64>
    bufferization.dealloc_tensor %collapse5 : tensor<6x10xf64>
    bufferization.dealloc_tensor %expand9 : tensor<?x2x?xf64>
    bufferization.dealloc_tensor %collapse9: tensor<?x?xf64>

    return
  }
}
