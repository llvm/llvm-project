//--------------------------------------------------------------------------------------------------
// WHEN CREATING A NEW TEST, PLEASE JUST COPY & PASTE WITHOUT EDITS.
//
// Set-up that's shared across all tests in this directory. In principle, this
// config could be moved to lit.local.cfg. However, there are downstream users that
//  do not use these LIT config files. Hence why this is kept inline.
//
// DEFINE: %{sparsifier_opts} = enable-runtime-library=true
// DEFINE: %{sparsifier_opts_sve} = enable-arm-sve=true %{sparsifier_opts}
// DEFINE: %{compile} = mlir-opt %s --sparsifier="%{sparsifier_opts}"
// DEFINE: %{compile_sve} = mlir-opt %s --sparsifier="%{sparsifier_opts_sve}"
// DEFINE: %{run_libs} = -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils
// DEFINE: %{run_opts} = -e entry -entry-point-result=void
// DEFINE: %{run} = mlir-cpu-runner %{run_opts} %{run_libs}
// DEFINE: %{run_sve} = %mcr_aarch64_cmd --march=aarch64 --mattr="+sve" %{run_opts} %{run_libs}
//
// DEFINE: %{env} =
//--------------------------------------------------------------------------------------------------

// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false enable-buffer-initialization=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false enable-buffer-initialization=true vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with  VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#SparseVector = #sparse_tensor.encoding<{map = (d0) -> (d0 : compressed)}>
#DCSR = #sparse_tensor.encoding<{map = (d0, d1) -> (d0 : compressed, d1 : compressed)}>

//
// Traits for tensor operations.
//
#trait_vec = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a (in)
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"]
}
#trait_mat = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A (in)
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"]
}

module {
  // Invert the structure of a sparse vector. Present values become missing.
  // Missing values are filled with 1 (i32). Output is sparse.
  func.func @vector_complement_sparse(%arga: tensor<?xf64, #SparseVector>) -> tensor<?xi32, #SparseVector> {
    %c = arith.constant 0 : index
    %ci1 = arith.constant 1 : i32
    %d = tensor.dim %arga, %c : tensor<?xf64, #SparseVector>
    %xv = tensor.empty(%d) : tensor<?xi32, #SparseVector>
    %0 = linalg.generic #trait_vec
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

  // Invert the structure of a sparse vector, where missing values are
  // filled with 1. For a dense output, the sparsifier initializes
  // the buffer to all zero at all other places.
  func.func @vector_complement_dense(%arga: tensor<?xf64, #SparseVector>) -> tensor<?xi32> {
    %c = arith.constant 0 : index
    %d = tensor.dim %arga, %c : tensor<?xf64, #SparseVector>
    %xv = tensor.empty(%d) : tensor<?xi32>
    %0 = linalg.generic #trait_vec
       ins(%arga: tensor<?xf64, #SparseVector>)
        outs(%xv: tensor<?xi32>) {
        ^bb(%a: f64, %x: i32):
          %1 = sparse_tensor.unary %a : f64 to i32
            present={}
            absent={
              %ci1 = arith.constant 1 : i32
              sparse_tensor.yield %ci1 : i32
            }
          linalg.yield %1 : i32
    } -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }

  // Negate existing values. Fill missing ones with +1.
  func.func @vector_negation(%arga: tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector> {
    %c = arith.constant 0 : index
    %cf1 = arith.constant 1.0 : f64
    %d = tensor.dim %arga, %c : tensor<?xf64, #SparseVector>
    %xv = tensor.empty(%d) : tensor<?xf64, #SparseVector>
    %0 = linalg.generic #trait_vec
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

  // Performs B[i] = i * A[i].
  func.func @vector_magnify(%arga: tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector> {
    %c = arith.constant 0 : index
    %d = tensor.dim %arga, %c : tensor<?xf64, #SparseVector>
    %xv = tensor.empty(%d) : tensor<?xf64, #SparseVector>
    %0 = linalg.generic #trait_vec
       ins(%arga: tensor<?xf64, #SparseVector>)
        outs(%xv: tensor<?xf64, #SparseVector>) {
        ^bb(%a: f64, %x: f64):
          %idx = linalg.index 0 : index
          %1 = sparse_tensor.unary %a : f64 to f64
            present={
              ^bb0(%x0: f64):
                %tmp = arith.index_cast %idx : index to i64
                %idxf = arith.uitofp %tmp : i64 to f64
                %ret = arith.mulf %x0, %idxf : f64
                sparse_tensor.yield %ret : f64
            }
            absent={}
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
    %xv = tensor.empty(%d0, %d1) : tensor<?x?xf64, #DCSR>
    %0 = linalg.generic #trait_mat
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

  // Slices matrix and only keep the value of the lower-right corner of the original
  // matrix (i.e., A[2/d0 ..][2/d1 ..]), and set other values to 99.
  func.func @matrix_slice(%argx: tensor<?x?xf64, #DCSR>) -> tensor<?x?xf64, #DCSR> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = tensor.dim %argx, %c0 : tensor<?x?xf64, #DCSR>
    %d1 = tensor.dim %argx, %c1 : tensor<?x?xf64, #DCSR>
    %xv = tensor.empty(%d0, %d1) : tensor<?x?xf64, #DCSR>
    %0 = linalg.generic #trait_mat
       ins(%argx: tensor<?x?xf64, #DCSR>)
        outs(%xv: tensor<?x?xf64, #DCSR>) {
        ^bb(%a: f64, %x: f64):
          %row = linalg.index 0 : index
          %col = linalg.index 1 : index
          %1 = sparse_tensor.unary %a: f64 to f64
            present={
              ^bb0(%x0: f64):
                %v = arith.constant 99.0 : f64
                %two = arith.constant 2 : index
                %r = arith.muli %two, %row : index
                %c = arith.muli %two, %col : index
                %cmp1 = arith.cmpi "ult", %r, %d0 : index
                %tmp = arith.select %cmp1, %v, %x0 : f64
                %cmp2 = arith.cmpi "ult", %c, %d1 : index
                %result = arith.select %cmp2, %v, %tmp : f64
                sparse_tensor.yield %result : f64
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
    %d0 = arith.constant 0.0 : f64
    %0 = sparse_tensor.values %arg0 : tensor<?xf64, #SparseVector> to memref<?xf64>
    %1 = vector.transfer_read %0[%c0], %d0: memref<?xf64>, vector<32xf64>
    vector.print %1 : vector<32xf64>
    // Dump the dense vector to verify structure is correct.
    %dv = sparse_tensor.convert %arg0 : tensor<?xf64, #SparseVector> to tensor<?xf64>
    %3 = vector.transfer_read %dv[%c0], %d0: tensor<?xf64>, vector<32xf64>
    vector.print %3 : vector<32xf64>
    bufferization.dealloc_tensor %dv : tensor<?xf64>
    return
  }

  // Dumps a sparse vector of type i32.
  func.func @dump_vec_i32(%arg0: tensor<?xi32, #SparseVector>) {
    // Dump the values array to verify only sparse contents are stored.
    %c0 = arith.constant 0 : index
    %d0 = arith.constant 0 : i32
    %0 = sparse_tensor.values %arg0 : tensor<?xi32, #SparseVector> to memref<?xi32>
    %1 = vector.transfer_read %0[%c0], %d0: memref<?xi32>, vector<24xi32>
    vector.print %1 : vector<24xi32>
    // Dump the dense vector to verify structure is correct.
    %dv = sparse_tensor.convert %arg0 : tensor<?xi32, #SparseVector> to tensor<?xi32>
    %3 = vector.transfer_read %dv[%c0], %d0: tensor<?xi32>, vector<32xi32>
    vector.print %3 : vector<32xi32>
    bufferization.dealloc_tensor %dv : tensor<?xi32>
    return
  }

  // Dump a sparse matrix.
  func.func @dump_mat(%arg0: tensor<?x?xf64, #DCSR>) {
    %c0 = arith.constant 0 : index
    %d0 = arith.constant 0.0 : f64
    %0 = sparse_tensor.values %arg0 : tensor<?x?xf64, #DCSR> to memref<?xf64>
    %1 = vector.transfer_read %0[%c0], %d0: memref<?xf64>, vector<16xf64>
    vector.print %1 : vector<16xf64>
    %dm = sparse_tensor.convert %arg0 : tensor<?x?xf64, #DCSR> to tensor<?x?xf64>
    %3 = vector.transfer_read %dm[%c0, %c0], %d0: tensor<?x?xf64>, vector<4x8xf64>
    vector.print %3 : vector<4x8xf64>
    bufferization.dealloc_tensor %dm : tensor<?x?xf64>
    return
  }

  // Driver method to call and verify vector kernels.
  func.func @entry() {
    %cmu = arith.constant -99 : i32
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
    %0 = call @vector_complement_sparse(%sv1)
       : (tensor<?xf64, #SparseVector>) -> tensor<?xi32, #SparseVector>
    %1 = call @vector_negation(%sv1)
       : (tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector>
    %2 = call @vector_magnify(%sv1)
       : (tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector>

    // Call sparse matrix kernels.
    %3 = call @matrix_clip(%sm1)
      : (tensor<?x?xf64, #DCSR>) -> tensor<?x?xf64, #DCSR>
    %4 = call @matrix_slice(%sm1)
      : (tensor<?x?xf64, #DCSR>) -> tensor<?x?xf64, #DCSR>

    // Call kernel with dense output.
    %5 = call @vector_complement_dense(%sv1) : (tensor<?xf64, #SparseVector>) -> tensor<?xi32>

    //
    // Verify the results.
    //
    // CHECK:      ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 4, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 7, 8, 0, 9 )
    // CHECK-NEXT: ( 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0 )
    // CHECK-NEXT: ( 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0 )
    // CHECK-NEXT: ( -1, 1, 1, -2, 1, 1, 1, 1, 1, 1, 1, -3, 1, 1, 1, 1, 1, -4, 1, 1, -5, -6, 1, 1, 1, 1, 1, 1, -7, -8, 1, -9 )
    // CHECK-NEXT: ( -1, 1, 1, -2, 1, 1, 1, 1, 1, 1, 1, -3, 1, 1, 1, 1, 1, -4, 1, 1, -5, -6, 1, 1, 1, 1, 1, 1, -7, -8, 1, -9 )
    // CHECK-NEXT: ( 0, 6, 33, 68, 100, 126, 196, 232, 279, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 0, 0, 68, 0, 0, 100, 126, 0, 0, 0, 0, 0, 0, 196, 232, 0, 279 )
    // CHECK-NEXT: ( 3, 3, 3, 4, 5, 6, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( ( 3, 3, 0, 0, 0, 0, 0, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 3 ), ( 0, 0, 4, 0, 5, 0, 0, 6 ), ( 7, 0, 7, 7, 0, 0, 0, 0 ) )
    // CHECK-NEXT: ( 99, 99, 99, 99, 5, 6, 99, 99, 99, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( ( 99, 99, 0, 0, 0, 0, 0, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 99 ), ( 0, 0, 99, 0, 5, 0, 0, 6 ), ( 99, 0, 99, 99, 0, 0, 0, 0 ) )
    // CHECK-NEXT: ( 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0 )
    //
    call @dump_vec_f64(%sv1) : (tensor<?xf64, #SparseVector>) -> ()
    call @dump_vec_i32(%0) : (tensor<?xi32, #SparseVector>) -> ()
    call @dump_vec_f64(%1) : (tensor<?xf64, #SparseVector>) -> ()
    call @dump_vec_f64(%2) : (tensor<?xf64, #SparseVector>) -> ()
    call @dump_mat(%3) : (tensor<?x?xf64, #DCSR>) -> ()
    call @dump_mat(%4) : (tensor<?x?xf64, #DCSR>) -> ()
    %v = vector.transfer_read %5[%c0], %cmu: tensor<?xi32>, vector<32xi32>
    vector.print %v : vector<32xi32>

    // Release the resources.
    bufferization.dealloc_tensor %sv1 : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %sm1 : tensor<?x?xf64, #DCSR>
    bufferization.dealloc_tensor %0 : tensor<?xi32, #SparseVector>
    bufferization.dealloc_tensor %1 : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %2 : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %3 : tensor<?x?xf64, #DCSR>
    bufferization.dealloc_tensor %4 : tensor<?x?xf64, #DCSR>
    bufferization.dealloc_tensor %5 : tensor<?xi32>
    return
  }
}
