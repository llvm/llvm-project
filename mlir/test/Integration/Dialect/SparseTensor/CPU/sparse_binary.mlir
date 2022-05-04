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
#trait_vec_op = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a (in)
    affine_map<(i) -> (i)>,  // b (in)
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"]
}
#trait_mat_op = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A (in)
    affine_map<(i,j) -> (i,j)>,  // B (in)
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(i,j) OP B(i,j)"
}

module {
  // Creates a new sparse vector using the minimum values from two input sparse vectors.
  // When there is no overlap, include the present value in the output.
  func @vector_min(%arga: tensor<?xf64, #SparseVector>,
                   %argb: tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector> {
    %c = arith.constant 0 : index
    %d = tensor.dim %arga, %c : tensor<?xf64, #SparseVector>
    %xv = sparse_tensor.init [%d] : tensor<?xf64, #SparseVector>
    %0 = linalg.generic #trait_vec_op
       ins(%arga, %argb: tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>)
        outs(%xv: tensor<?xf64, #SparseVector>) {
        ^bb(%a: f64, %b: f64, %x: f64):
          %1 = sparse_tensor.binary %a, %b : f64, f64 to f64
            overlap={
              ^bb0(%a0: f64, %b0: f64):
                %cmp = arith.cmpf "olt", %a0, %b0 : f64
                %2 = arith.select %cmp, %a0, %b0: f64
                sparse_tensor.yield %2 : f64
            }
            left=identity
            right=identity
          linalg.yield %1 : f64
    } -> tensor<?xf64, #SparseVector>
    return %0 : tensor<?xf64, #SparseVector>
  }

  // Creates a new sparse vector by multiplying a sparse vector with a dense vector.
  // When there is no overlap, leave the result empty.
  func @vector_mul(%arga: tensor<?xf64, #SparseVector>,
                   %argb: tensor<?xf64>) -> tensor<?xf64, #SparseVector> {
    %c = arith.constant 0 : index
    %d = tensor.dim %arga, %c : tensor<?xf64, #SparseVector>
    %xv = sparse_tensor.init [%d] : tensor<?xf64, #SparseVector>
    %0 = linalg.generic #trait_vec_op
       ins(%arga, %argb: tensor<?xf64, #SparseVector>, tensor<?xf64>)
        outs(%xv: tensor<?xf64, #SparseVector>) {
        ^bb(%a: f64, %b: f64, %x: f64):
          %1 = sparse_tensor.binary %a, %b : f64, f64 to f64
            overlap={
              ^bb0(%a0: f64, %b0: f64):
                %ret = arith.mulf %a0, %b0 : f64
                sparse_tensor.yield %ret : f64
            }
            left={}
            right={}
          linalg.yield %1 : f64
    } -> tensor<?xf64, #SparseVector>
    return %0 : tensor<?xf64, #SparseVector>
  }

  // Take a set difference of two sparse vectors. The result will include only those
  // sparse elements present in the first, but not the second vector.
  func @vector_setdiff(%arga: tensor<?xf64, #SparseVector>,
                       %argb: tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector> {
    %c = arith.constant 0 : index
    %d = tensor.dim %arga, %c : tensor<?xf64, #SparseVector>
    %xv = sparse_tensor.init [%d] : tensor<?xf64, #SparseVector>
    %0 = linalg.generic #trait_vec_op
       ins(%arga, %argb: tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>)
        outs(%xv: tensor<?xf64, #SparseVector>) {
        ^bb(%a: f64, %b: f64, %x: f64):
          %1 = sparse_tensor.binary %a, %b : f64, f64 to f64
            overlap={}
            left=identity
            right={}
          linalg.yield %1 : f64
    } -> tensor<?xf64, #SparseVector>
    return %0 : tensor<?xf64, #SparseVector>
  }

  // Return the index of each entry
  func @vector_index(%arga: tensor<?xf64, #SparseVector>) -> tensor<?xi32, #SparseVector> {
    %c = arith.constant 0 : index
    %d = tensor.dim %arga, %c : tensor<?xf64, #SparseVector>
    %xv = sparse_tensor.init [%d] : tensor<?xi32, #SparseVector>
    %0 = linalg.generic #trait_vec_scale
       ins(%arga: tensor<?xf64, #SparseVector>)
        outs(%xv: tensor<?xi32, #SparseVector>) {
        ^bb(%a: f64, %x: i32):
          %idx = linalg.index 0 : index
          %1 = sparse_tensor.binary %a, %idx : f64, index to i32
            overlap={
              ^bb0(%x0: f64, %i: index):
                %ret = arith.index_cast %i : index to i32
                sparse_tensor.yield %ret : i32
            }
            left={}
            right={}
          linalg.yield %1 : i32
    } -> tensor<?xi32, #SparseVector>
    return %0 : tensor<?xi32, #SparseVector>
  }

  // Adds two sparse matrices when they intersect. Where they don't intersect,
  // negate the 2nd argument's values; ignore 1st argument-only values.
  func @matrix_intersect(%arga: tensor<?x?xf64, #DCSR>,
                         %argb: tensor<?x?xf64, #DCSR>) -> tensor<?x?xf64, #DCSR> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = tensor.dim %arga, %c0 : tensor<?x?xf64, #DCSR>
    %d1 = tensor.dim %arga, %c1 : tensor<?x?xf64, #DCSR>
    %xv = sparse_tensor.init [%d0, %d1] : tensor<?x?xf64, #DCSR>
    %0 = linalg.generic #trait_mat_op
       ins(%arga, %argb: tensor<?x?xf64, #DCSR>, tensor<?x?xf64, #DCSR>)
        outs(%xv: tensor<?x?xf64, #DCSR>) {
        ^bb(%a: f64, %b: f64, %x: f64):
          %1 = sparse_tensor.binary %a, %b: f64, f64 to f64
            overlap={
              ^bb0(%x0: f64, %y0: f64):
                %ret = arith.addf %x0, %y0 : f64
                sparse_tensor.yield %ret : f64
            }
            left={}
            right={
              ^bb0(%x1: f64):
                %lret = arith.negf %x1 : f64
                sparse_tensor.yield %lret : f64
            }
          linalg.yield %1 : f64
    } -> tensor<?x?xf64, #DCSR>
    return %0 : tensor<?x?xf64, #DCSR>
  }

  // Dumps a sparse vector of type f64.
  func @dump_vec(%arg0: tensor<?xf64, #SparseVector>) {
    // Dump the values array to verify only sparse contents are stored.
    %c0 = arith.constant 0 : index
    %d0 = arith.constant -1.0 : f64
    %0 = sparse_tensor.values %arg0 : tensor<?xf64, #SparseVector> to memref<?xf64>
    %1 = vector.transfer_read %0[%c0], %d0: memref<?xf64>, vector<16xf64>
    vector.print %1 : vector<16xf64>
    // Dump the dense vector to verify structure is correct.
    %dv = sparse_tensor.convert %arg0 : tensor<?xf64, #SparseVector> to tensor<?xf64>
    %2 = bufferization.to_memref %dv : memref<?xf64>
    %3 = vector.transfer_read %2[%c0], %d0: memref<?xf64>, vector<32xf64>
    vector.print %3 : vector<32xf64>
    memref.dealloc %2 : memref<?xf64>
    return
  }

  // Dumps a sparse vector of type i32.
  func @dump_vec_i32(%arg0: tensor<?xi32, #SparseVector>) {
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
  func @dump_mat(%arg0: tensor<?x?xf64, #DCSR>) {
    %d0 = arith.constant 0.0 : f64
    %c0 = arith.constant 0 : index
    %dm = sparse_tensor.convert %arg0 : tensor<?x?xf64, #DCSR> to tensor<?x?xf64>
    %0 = bufferization.to_memref %dm : memref<?x?xf64>
    %1 = vector.transfer_read %0[%c0, %c0], %d0: memref<?x?xf64>, vector<4x8xf64>
    vector.print %1 : vector<4x8xf64>
    memref.dealloc %0 : memref<?x?xf64>
    return
  }

  // Driver method to call and verify vector kernels.
  func @entry() {
    %c0 = arith.constant 0 : index

    // Setup sparse vectors.
    %v1 = arith.constant sparse<
       [ [0], [3], [11], [17], [20], [21], [28], [29], [31] ],
         [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 ]
    > : tensor<32xf64>
    %v2 = arith.constant sparse<
       [ [1], [3], [4], [10], [16], [18], [21], [28], [29], [31] ],
         [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0 ]
    > : tensor<32xf64>
    %v3 = arith.constant dense<
      [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
       0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 0., 1.]
    > : tensor<32xf64>
    %sv1 = sparse_tensor.convert %v1 : tensor<32xf64> to tensor<?xf64, #SparseVector>
    %sv2 = sparse_tensor.convert %v2 : tensor<32xf64> to tensor<?xf64, #SparseVector>
    %dv3 = tensor.cast %v3 : tensor<32xf64> to tensor<?xf64>

    // Setup sparse matrices.
    %m1 = arith.constant sparse<
       [ [0,0], [0,1], [1,7], [2,2], [2,4], [2,7], [3,0], [3,2], [3,3] ],
         [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 ]
    > : tensor<4x8xf64>
    %m2 = arith.constant sparse<
       [ [0,0], [0,7], [1,0], [1,6], [2,1], [2,7] ],
         [6.0, 5.0, 4.0, 3.0, 2.0, 1.0 ]
    > : tensor<4x8xf64>
    %sm1 = sparse_tensor.convert %m1 : tensor<4x8xf64> to tensor<?x?xf64, #DCSR>
    %sm2 = sparse_tensor.convert %m2 : tensor<4x8xf64> to tensor<?x?xf64, #DCSR>

    // Call sparse vector kernels.
    %0 = call @vector_min(%sv1, %sv2)
       : (tensor<?xf64, #SparseVector>,
          tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector>
    %1 = call @vector_mul(%sv1, %dv3)
      : (tensor<?xf64, #SparseVector>,
         tensor<?xf64>) -> tensor<?xf64, #SparseVector>
    %2 = call @vector_setdiff(%sv1, %sv2)
       : (tensor<?xf64, #SparseVector>,
          tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector>
    %3 = call @vector_index(%sv1)
       : (tensor<?xf64, #SparseVector>) -> tensor<?xi32, #SparseVector>

    // Call sparse matrix kernels.
    %5 = call @matrix_intersect(%sm1, %sm2)
      : (tensor<?x?xf64, #DCSR>, tensor<?x?xf64, #DCSR>) -> tensor<?x?xf64, #DCSR>

    //
    // Verify the results.
    //
    // CHECK:      ( 1, 2, 3, 4, 5, 6, 7, 8, 9, -1, -1, -1, -1, -1, -1, -1 )
    // CHECK-NEXT: ( 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 4, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 7, 8, 0, 9 )
    // CHECK-NEXT: ( 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, -1, -1, -1, -1, -1, -1 )
    // CHECK-NEXT: ( 0, 11, 0, 12, 13, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 15, 0, 16, 0, 0, 17, 0, 0, 0, 0, 0, 0, 18, 19, 0, 20 )
    // CHECK-NEXT: ( 1, 11, 2, 13, 14, 3, 15, 4, 16, 5, 6, 7, 8, 9, -1, -1 )
    // CHECK-NEXT: ( 1, 11, 0, 2, 13, 0, 0, 0, 0, 0, 14, 3, 0, 0, 0, 0, 15, 4, 16, 0, 5, 6, 0, 0, 0, 0, 0, 0, 7, 8, 0, 9 )
    // CHECK-NEXT: ( 0, 6, 3, 28, 0, 6, 56, 72, 9, -1, -1, -1, -1, -1, -1, -1 )
    // CHECK-NEXT: ( 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 28, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 56, 72, 0, 9 )
    // CHECK-NEXT: ( 1, 3, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 )
    // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 4, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 3, 11, 17, 20, 21, 28, 29, 31, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 )
    // CHECK-NEXT: ( 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 17, 0, 0, 20, 21, 0, 0, 0, 0, 0, 0, 28, 29, 0, 31 )
    // CHECK-NEXT: ( ( 7, 0, 0, 0, 0, 0, 0, -5 ), ( -4, 0, 0, 0, 0, 0, -3, 0 ), ( 0, -2, 0, 0, 0, 0, 0, 7 ), ( 0, 0, 0, 0, 0, 0, 0, 0 ) )
    //
    call @dump_vec(%sv1) : (tensor<?xf64, #SparseVector>) -> ()
    call @dump_vec(%sv2) : (tensor<?xf64, #SparseVector>) -> ()
    call @dump_vec(%0) : (tensor<?xf64, #SparseVector>) -> ()
    call @dump_vec(%1) : (tensor<?xf64, #SparseVector>) -> ()
    call @dump_vec(%2) : (tensor<?xf64, #SparseVector>) -> ()
    call @dump_vec_i32(%3) : (tensor<?xi32, #SparseVector>) -> ()
    call @dump_mat(%5) : (tensor<?x?xf64, #DCSR>) -> ()

    // Release the resources.
    sparse_tensor.release %sv1 : tensor<?xf64, #SparseVector>
    sparse_tensor.release %sv2 : tensor<?xf64, #SparseVector>
    sparse_tensor.release %sm1 : tensor<?x?xf64, #DCSR>
    sparse_tensor.release %sm2 : tensor<?x?xf64, #DCSR>
    sparse_tensor.release %0 : tensor<?xf64, #SparseVector>
    sparse_tensor.release %1 : tensor<?xf64, #SparseVector>
    sparse_tensor.release %2 : tensor<?xf64, #SparseVector>
    sparse_tensor.release %3 : tensor<?xi32, #SparseVector>
    sparse_tensor.release %5 : tensor<?x?xf64, #DCSR>
    return
  }
}