// RUN: mlir-opt %s --sparse-compiler | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>
#DCSR = #sparse_tensor.encoding<{dimLevelType = ["compressed", "compressed"]}>

//
// Traits for tensor operations.
//
#trait_vec_scale = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a (in)
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"]
}
#trait_mat_scale = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A (in)
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"]
}

module {
  // Invert the structure of a sparse vector. Present values become missing.
  // Missing values are filled with 1 (i32).
  func.func @vector_complement(%arga: tensor<?xf64, #SparseVector>) -> tensor<?xi32, #SparseVector> {
    %c = arith.constant 0 : index
    %ci1 = arith.constant 1 : i32
    %d = tensor.dim %arga, %c : tensor<?xf64, #SparseVector>
    %xv = bufferization.alloc_tensor(%d) : tensor<?xi32, #SparseVector>
    %0 = linalg.generic #trait_vec_scale
       ins(%arga: tensor<?xf64, #SparseVector>)
        outs(%xv: tensor<?xi32, #SparseVector>) {
        ^bb(%a: f64, %x: i32):
          %1 = sparse_tensor.unary %a : f64 to i32
            present={}
            absent={
              sparse_tensor.yield %ci1 : i32
            }
          linalg.yield %1 : i32
    } -> tensor<?xi32, #SparseVector>
    return %0 : tensor<?xi32, #SparseVector>
  }

  // Negate existing values. Fill missing ones with +1.
  func.func @vector_negation(%arga: tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector> {
    %c = arith.constant 0 : index
    %cf1 = arith.constant 1.0 : f64
    %d = tensor.dim %arga, %c : tensor<?xf64, #SparseVector>
    %xv = bufferization.alloc_tensor(%d) : tensor<?xf64, #SparseVector>
    %0 = linalg.generic #trait_vec_scale
       ins(%arga: tensor<?xf64, #SparseVector>)
        outs(%xv: tensor<?xf64, #SparseVector>) {
        ^bb(%a: f64, %x: f64):
          %1 = sparse_tensor.unary %a : f64 to f64
            present={
              ^bb0(%x0: f64):
                %ret = arith.negf %x0 : f64
                sparse_tensor.yield %ret : f64
            }
            absent={
              sparse_tensor.yield %cf1 : f64
            }
          linalg.yield %1 : f64
    } -> tensor<?xf64, #SparseVector>
    return %0 : tensor<?xf64, #SparseVector>
  }

  // Clips values to the range [3, 7].
  func.func @matrix_clip(%argx: tensor<?x?xf64, #DCSR>) -> tensor<?x?xf64, #DCSR> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cfmin = arith.constant 3.0 : f64
    %cfmax = arith.constant 7.0 : f64
    %d0 = tensor.dim %argx, %c0 : tensor<?x?xf64, #DCSR>
    %d1 = tensor.dim %argx, %c1 : tensor<?x?xf64, #DCSR>
    %xv = bufferization.alloc_tensor(%d0, %d1) : tensor<?x?xf64, #DCSR>
    %0 = linalg.generic #trait_mat_scale
       ins(%argx: tensor<?x?xf64, #DCSR>)
        outs(%xv: tensor<?x?xf64, #DCSR>) {
        ^bb(%a: f64, %x: f64):
          %1 = sparse_tensor.unary %a: f64 to f64
            present={
              ^bb0(%x0: f64):
                %mincmp = arith.cmpf "ogt", %x0, %cfmin : f64
                %x1 = arith.select %mincmp, %x0, %cfmin : f64
                %maxcmp = arith.cmpf "olt", %x1, %cfmax : f64
                %x2 = arith.select %maxcmp, %x1, %cfmax : f64
                sparse_tensor.yield %x2 : f64
            }
            absent={}
          linalg.yield %1 : f64
    } -> tensor<?x?xf64, #DCSR>
    return %0 : tensor<?x?xf64, #DCSR>
  }

  // Dumps a sparse vector of type f64.
  func.func @dump_vec_f64(%arg0: tensor<?xf64, #SparseVector>) {
    // Dump the values array to verify only sparse contents are stored.
    %c0 = arith.constant 0 : index
    %d0 = arith.constant -1.0 : f64
    %0 = sparse_tensor.values %arg0 : tensor<?xf64, #SparseVector> to memref<?xf64>
    %1 = vector.transfer_read %0[%c0], %d0: memref<?xf64>, vector<32xf64>
    vector.print %1 : vector<32xf64>
    // Dump the dense vector to verify structure is correct.
    %dv = sparse_tensor.convert %arg0 : tensor<?xf64, #SparseVector> to tensor<?xf64>
    %2 = bufferization.to_memref %dv : memref<?xf64>
    %3 = vector.transfer_read %2[%c0], %d0: memref<?xf64>, vector<32xf64>
    vector.print %3 : vector<32xf64>
    memref.dealloc %2 : memref<?xf64>
    return
  }

  // Dumps a sparse vector of type i32.
  func.func @dump_vec_i32(%arg0: tensor<?xi32, #SparseVector>) {
    // Dump the values array to verify only sparse contents are stored.
    %c0 = arith.constant 0 : index
    %d0 = arith.constant -1 : i32
    %0 = sparse_tensor.values %arg0 : tensor<?xi32, #SparseVector> to memref<?xi32>
    %1 = vector.transfer_read %0[%c0], %d0: memref<?xi32>, vector<24xi32>
    vector.print %1 : vector<24xi32>
    // Dump the dense vector to verify structure is correct.
    %dv = sparse_tensor.convert %arg0 : tensor<?xi32, #SparseVector> to tensor<?xi32>
    %2 = bufferization.to_memref %dv : memref<?xi32>
    %3 = vector.transfer_read %2[%c0], %d0: memref<?xi32>, vector<32xi32>
    vector.print %3 : vector<32xi32>
    memref.dealloc %2 : memref<?xi32>
    return
  }

  // Dump a sparse matrix.
  func.func @dump_mat(%arg0: tensor<?x?xf64, #DCSR>) {
    %c0 = arith.constant 0 : index
    %d0 = arith.constant -1.0 : f64
    %0 = sparse_tensor.values %arg0 : tensor<?x?xf64, #DCSR> to memref<?xf64>
    %1 = vector.transfer_read %0[%c0], %d0: memref<?xf64>, vector<16xf64>
    vector.print %1 : vector<16xf64>
    %dm = sparse_tensor.convert %arg0 : tensor<?x?xf64, #DCSR> to tensor<?x?xf64>
    %2 = bufferization.to_memref %dm : memref<?x?xf64>
    %3 = vector.transfer_read %2[%c0, %c0], %d0: memref<?x?xf64>, vector<4x8xf64>
    vector.print %3 : vector<4x8xf64>
    memref.dealloc %2 : memref<?x?xf64>
    return
  }

  // Driver method to call and verify vector kernels.
  func.func @entry() {
    %c0 = arith.constant 0 : index

    // Setup sparse vectors.
    %v1 = arith.constant sparse<
       [ [0], [3], [11], [17], [20], [21], [28], [29], [31] ],
         [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 ]
    > : tensor<32xf64>
    %sv1 = sparse_tensor.convert %v1 : tensor<32xf64> to tensor<?xf64, #SparseVector>

    // Setup sparse matrices.
    %m1 = arith.constant sparse<
       [ [0,0], [0,1], [1,7], [2,2], [2,4], [2,7], [3,0], [3,2], [3,3] ],
         [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 ]
    > : tensor<4x8xf64>
    %sm1 = sparse_tensor.convert %m1 : tensor<4x8xf64> to tensor<?x?xf64, #DCSR>

    // Call sparse vector kernels.
    %0 = call @vector_complement(%sv1)
       : (tensor<?xf64, #SparseVector>) -> tensor<?xi32, #SparseVector>
    %1 = call @vector_negation(%sv1)
       : (tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector>


    // Call sparse matrix kernels.
    %2 = call @matrix_clip(%sm1)
      : (tensor<?x?xf64, #DCSR>) -> tensor<?x?xf64, #DCSR>

    //
    // Verify the results.
    //
    // CHECK:      ( 1, 2, 3, 4, 5, 6, 7, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 )
    // CHECK-NEXT: ( 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 4, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 7, 8, 0, 9 )
    // CHECK-NEXT: ( 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1 )
    // CHECK-NEXT: ( 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0 )
    // CHECK-NEXT: ( -1, 1, 1, -2, 1, 1, 1, 1, 1, 1, 1, -3, 1, 1, 1, 1, 1, -4, 1, 1, -5, -6, 1, 1, 1, 1, 1, 1, -7, -8, 1, -9 )
    // CHECK-NEXT: ( -1, 1, 1, -2, 1, 1, 1, 1, 1, 1, 1, -3, 1, 1, 1, 1, 1, -4, 1, 1, -5, -6, 1, 1, 1, 1, 1, 1, -7, -8, 1, -9 )
    // CHECK-NEXT: ( 3, 3, 3, 4, 5, 6, 7, 7, 7, -1, -1, -1, -1, -1, -1, -1 )
    // CHECK-NEXT: ( ( 3, 3, 0, 0, 0, 0, 0, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 3 ), ( 0, 0, 4, 0, 5, 0, 0, 6 ), ( 7, 0, 7, 7, 0, 0, 0, 0 ) )
    //
    call @dump_vec_f64(%sv1) : (tensor<?xf64, #SparseVector>) -> ()
    call @dump_vec_i32(%0) : (tensor<?xi32, #SparseVector>) -> ()
    call @dump_vec_f64(%1) : (tensor<?xf64, #SparseVector>) -> ()
    call @dump_mat(%2) : (tensor<?x?xf64, #DCSR>) -> ()

    // Release the resources.
    sparse_tensor.release %sv1 : tensor<?xf64, #SparseVector>
    sparse_tensor.release %sm1 : tensor<?x?xf64, #DCSR>
    sparse_tensor.release %0 : tensor<?xi32, #SparseVector>
    sparse_tensor.release %1 : tensor<?xf64, #SparseVector>
    sparse_tensor.release %2 : tensor<?x?xf64, #DCSR>
    return
  }
}
