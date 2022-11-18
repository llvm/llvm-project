// DEFINE: %{option} = enable-runtime-library=true
// DEFINE: %{command} = mlir-opt %s --sparse-compiler=%{option} | \
// DEFINE: mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// DEFINE: FileCheck %s
//
// RUN: %{command}
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{option} = "enable-runtime-library=false enable-buffer-initialization=true"
// RUN: %{command}

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>
#CSR = #sparse_tensor.encoding<{dimLevelType = ["dense", "compressed"]}>
#CSC = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>
}>

//
// Traits for tensor operations.
//
#trait_matmul = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,k)>, // A
    affine_map<(i,j,k) -> (k,j)>, // B
    affine_map<(i,j,k) -> (i,j)>  // C (out)
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  doc = "C(i,j) = SUM_k A(i,k) * B(k,j)"
}

#trait_mat_reduce_rowwise = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A (in)
    affine_map<(i,j) -> (i)>   // X (out)
  ],
  iterator_types = ["parallel", "reduction"],
  doc = "X(i) = PROD_j A(i,j)"
}

#trait_mat_reduce_colwise = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A (in)
    affine_map<(i,j) -> (j)>   // X (out)
  ],
  iterator_types = ["reduction", "parallel"],
  doc = "X(j) = PROD_i A(i,j)"
}

module {
  func.func @redProdLex(%arga: tensor<?x?xf64, #CSR>) -> tensor<?xf64, #SparseVector> {
    %c0 = arith.constant 0 : index
    %cf1 = arith.constant 1.0 : f64
    %d0 = tensor.dim %arga, %c0 : tensor<?x?xf64, #CSR>
    %xv = bufferization.alloc_tensor(%d0): tensor<?xf64, #SparseVector>
    %0 = linalg.generic #trait_mat_reduce_rowwise
      ins(%arga: tensor<?x?xf64, #CSR>)
      outs(%xv: tensor<?xf64, #SparseVector>) {
        ^bb(%a: f64, %b: f64):
          %1 = sparse_tensor.reduce %a, %b, %cf1 : f64 {
              ^bb0(%x: f64, %y: f64):
                %2 = arith.mulf %x, %y : f64
                sparse_tensor.yield %2 : f64
            }
          linalg.yield %1 : f64
    } -> tensor<?xf64, #SparseVector>
    return %0 : tensor<?xf64, #SparseVector>
  }

  func.func @redProdExpand(%arga: tensor<?x?xf64, #CSC>) -> tensor<?xf64, #SparseVector> {
    %c0 = arith.constant 0 : index
    %cf1 = arith.constant 1.0 : f64
    %d0 = tensor.dim %arga, %c0 : tensor<?x?xf64, #CSC>
    %xv = bufferization.alloc_tensor(%d0): tensor<?xf64, #SparseVector>
    %0 = linalg.generic #trait_mat_reduce_rowwise
      ins(%arga: tensor<?x?xf64, #CSC>)
      outs(%xv: tensor<?xf64, #SparseVector>) {
        ^bb(%a: f64, %b: f64):
          %1 = sparse_tensor.reduce %a, %b, %cf1 : f64 {
              ^bb0(%x: f64, %y: f64):
                %2 = arith.mulf %x, %y : f64
                sparse_tensor.yield %2 : f64
            }
          linalg.yield %1 : f64
    } -> tensor<?xf64, #SparseVector>
    return %0 : tensor<?xf64, #SparseVector>
  }

  func.func @min_plus_csrcsr(%arga: tensor<?x?xf64, #CSR>,
                             %argb: tensor<?x?xf64, #CSR>) -> tensor<?x?xf64, #CSR> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %maxf = arith.constant 1.0e999 : f64
    %d0 = tensor.dim %arga, %c0 : tensor<?x?xf64, #CSR>
    %d1 = tensor.dim %argb, %c1 : tensor<?x?xf64, #CSR>
    %xm = bufferization.alloc_tensor(%d0, %d1) : tensor<?x?xf64, #CSR>
    %0 = linalg.generic #trait_matmul
       ins(%arga, %argb: tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSR>)
        outs(%xm: tensor<?x?xf64, #CSR>) {
        ^bb(%a: f64, %b: f64, %output: f64):
          %1 = sparse_tensor.binary %a, %b : f64, f64 to f64
            overlap = {
              ^bb0(%x: f64, %y: f64):
                %3 = arith.addf %x, %y : f64
                sparse_tensor.yield %3 : f64
            }
            left={}
            right={}
          %2 = sparse_tensor.reduce %1, %output, %maxf : f64 {
              ^bb0(%x: f64, %y: f64):
                %cmp = arith.cmpf "olt", %x, %y : f64
                %3 = arith.select %cmp, %x, %y : f64
                sparse_tensor.yield %3 : f64
            }
          linalg.yield %2 : f64
    } -> tensor<?x?xf64, #CSR>
    return %0 : tensor<?x?xf64, #CSR>
  }

  func.func @min_plus_csrcsc(%arga: tensor<?x?xf64, #CSR>,
                             %argb: tensor<?x?xf64, #CSC>) -> tensor<?x?xf64, #CSR> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %maxf = arith.constant 1.0e999 : f64
    %d0 = tensor.dim %arga, %c0 : tensor<?x?xf64, #CSR>
    %d1 = tensor.dim %argb, %c1 : tensor<?x?xf64, #CSC>
    %xm = bufferization.alloc_tensor(%d0, %d1) : tensor<?x?xf64, #CSR>
    %0 = linalg.generic #trait_matmul
       ins(%arga, %argb: tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSC>)
        outs(%xm: tensor<?x?xf64, #CSR>) {
        ^bb(%a: f64, %b: f64, %output: f64):
          %1 = sparse_tensor.binary %a, %b : f64, f64 to f64
            overlap = {
              ^bb0(%x: f64, %y: f64):
                %3 = arith.addf %x, %y : f64
                sparse_tensor.yield %3 : f64
            }
            left={}
            right={}
          %2 = sparse_tensor.reduce %1, %output, %maxf : f64 {
              ^bb0(%x: f64, %y: f64):
                %cmp = arith.cmpf "olt", %x, %y : f64
                %3 = arith.select %cmp, %x, %y : f64
                sparse_tensor.yield %3 : f64
            }
          linalg.yield %2 : f64
    } -> tensor<?x?xf64, #CSR>
    return %0 : tensor<?x?xf64, #CSR>
  }

  // Dumps a sparse vector of type f64.
  func.func @dump_vec(%arg0: tensor<?xf64, #SparseVector>) {
    // Dump the values array to verify only sparse contents are stored.
    %c0 = arith.constant 0 : index
    %d0 = arith.constant 0.0 : f64
    %0 = sparse_tensor.values %arg0 : tensor<?xf64, #SparseVector> to memref<?xf64>
    %1 = vector.transfer_read %0[%c0], %d0: memref<?xf64>, vector<8xf64>
    vector.print %1 : vector<8xf64>
    // Dump the dense vector to verify structure is correct.
    %dv = sparse_tensor.convert %arg0 : tensor<?xf64, #SparseVector> to tensor<?xf64>
    %2 = vector.transfer_read %dv[%c0], %d0: tensor<?xf64>, vector<16xf64>
    vector.print %2 : vector<16xf64>
    return
  }

  // Dump a sparse matrix.
  func.func @dump_mat(%arg0: tensor<?x?xf64, #CSR>) {
    // Dump the values array to verify only sparse contents are stored.
    %c0 = arith.constant 0 : index
    %d0 = arith.constant 0.0 : f64
    %0 = sparse_tensor.values %arg0 : tensor<?x?xf64, #CSR> to memref<?xf64>
    %1 = vector.transfer_read %0[%c0], %d0: memref<?xf64>, vector<16xf64>
    vector.print %1 : vector<16xf64>
    %dm = sparse_tensor.convert %arg0 : tensor<?x?xf64, #CSR> to tensor<?x?xf64>
    %2 = vector.transfer_read %dm[%c0, %c0], %d0: tensor<?x?xf64>, vector<5x5xf64>
    vector.print %2 : vector<5x5xf64>
    return
  }

  // Driver method to call and verify vector kernels.
  func.func @entry() {
    %c0 = arith.constant 0 : index

    // Setup sparse matrices.
    %m1 = arith.constant sparse<
       [ [0,0], [0,1], [1,0], [2,2], [2,3], [2,4], [3,0], [3,2], [3,3] ],
         [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 ]
    > : tensor<4x5xf64>
    %m2 = arith.constant sparse<
       [ [0,0], [1,3], [2,0], [2,3], [3,1], [4,1] ],
         [6.0, 5.0, 4.0, 3.0, 2.0, 11.0 ]
    > : tensor<5x4xf64>
    %sm1 = sparse_tensor.convert %m1 : tensor<4x5xf64> to tensor<?x?xf64, #CSR>
    %sm2r = sparse_tensor.convert %m2 : tensor<5x4xf64> to tensor<?x?xf64, #CSR>
    %sm2c = sparse_tensor.convert %m2 : tensor<5x4xf64> to tensor<?x?xf64, #CSC>

    // Call sparse matrix kernels.
    %1 = call @redProdLex(%sm1) : (tensor<?x?xf64, #CSR>) -> tensor<?xf64, #SparseVector>
    %2 = call @redProdExpand(%sm2c) : (tensor<?x?xf64, #CSC>) -> tensor<?xf64, #SparseVector>
    %5 = call @min_plus_csrcsr(%sm1, %sm2r)
      : (tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSR>) -> tensor<?x?xf64, #CSR>
    %6 = call @min_plus_csrcsc(%sm1, %sm2c)
      : (tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSC>) -> tensor<?x?xf64, #CSR>

    //
    // Verify the results.
    //
    // CHECK:      ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( ( 1, 2, 0, 0, 0 ), ( 3, 0, 0, 0, 0 ), ( 0, 0, 4, 5, 6 ), ( 7, 0, 8, 9, 0 ), ( 0, 0, 0, 0, 0 ) )
    // CHECK-NEXT: ( 6, 5, 4, 3, 2, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( ( 6, 0, 0, 0, 0 ), ( 0, 0, 0, 5, 0 ), ( 4, 0, 0, 3, 0 ), ( 0, 2, 0, 0, 0 ), ( 0, 11, 0, 0, 0 ) )
    // CHECK-NEXT: ( 2, 3, 120, 504, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 2, 3, 120, 504, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 6, 5, 12, 2, 11, 0, 0, 0 )
    // CHECK-NEXT: ( 6, 5, 12, 2, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 7, 7, 9, 8, 7, 7, 12, 11, 11, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( ( 7, 0, 0, 7, 0 ), ( 9, 0, 0, 0, 0 ), ( 8, 7, 0, 7, 0 ), ( 12, 11, 0, 11, 0 ), ( 0, 0, 0, 0, 0 ) )
    // TODO: Update once identity values are no longer inserted for non-overlapping dot product
    // CHECK-NEXT: ( 7, inf, inf, 7, 9, inf, inf, inf, 8, 7, inf, 7, 12, 11, inf, 11 )
    // CHECK-NEXT: ( ( 7, inf, inf, 7, 0 ), ( 9, inf, inf, inf, 0 ), ( 8, 7, inf, 7, 0 ), ( 12, 11, inf, 11, 0 ), ( 0, 0, 0, 0, 0 ) )
    //
    call @dump_mat(%sm1) : (tensor<?x?xf64, #CSR>) -> ()
    call @dump_mat(%sm2r) : (tensor<?x?xf64, #CSR>) -> ()
    call @dump_vec(%1) : (tensor<?xf64, #SparseVector>) -> ()
    call @dump_vec(%2) : (tensor<?xf64, #SparseVector>) -> ()
    call @dump_mat(%5) : (tensor<?x?xf64, #CSR>) -> ()
    call @dump_mat(%6) : (tensor<?x?xf64, #CSR>) -> ()

    // Release the resources.
    bufferization.dealloc_tensor %sm1 : tensor<?x?xf64, #CSR>
    bufferization.dealloc_tensor %sm2r : tensor<?x?xf64, #CSR>
    bufferization.dealloc_tensor %sm2c : tensor<?x?xf64, #CSC>
    bufferization.dealloc_tensor %1 : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %2 : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %5 : tensor<?x?xf64, #CSR>
    bufferization.dealloc_tensor %6 : tensor<?x?xf64, #CSR>
    return
  }
}
