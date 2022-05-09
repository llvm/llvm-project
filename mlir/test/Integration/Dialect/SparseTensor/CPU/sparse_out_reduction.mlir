// RUN: mlir-opt %s --sparse-compiler | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#SparseMatrix = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ]
}>

#SparseTensor = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed", "compressed" ]
}>

#redsum = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,j,k)>, // A
    affine_map<(i,j,k) -> (i,j,k)>, // B
    affine_map<(i,j,k) -> (i,j)>    // X (out)
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  doc = "X(i,j) = SUM_k A(i,j,k) * B(i,j,k)"
}

module {
  func.func @redsum(%arga: tensor<?x?x?xi32, #SparseTensor>,
               %argb: tensor<?x?x?xi32, #SparseTensor>)
	           -> tensor<?x?xi32, #SparseMatrix> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = tensor.dim %arga, %c0 : tensor<?x?x?xi32, #SparseTensor>
    %d1 = tensor.dim %arga, %c1 : tensor<?x?x?xi32, #SparseTensor>
    %xinit = sparse_tensor.init [%d0, %d1] : tensor<?x?xi32, #SparseMatrix>
    %0 = linalg.generic #redsum
      ins(%arga, %argb: tensor<?x?x?xi32, #SparseTensor>,
                        tensor<?x?x?xi32, #SparseTensor>)
      outs(%xinit: tensor<?x?xi32, #SparseMatrix>) {
        ^bb(%a: i32, %b: i32, %x: i32):
          %0 = arith.muli %a, %b : i32
          %1 = arith.addi %x, %0 : i32
          linalg.yield %1 : i32
    } -> tensor<?x?xi32, #SparseMatrix>
    return %0 : tensor<?x?xi32, #SparseMatrix>
  }

  // Driver method to call and verify tensor kernel.
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %i0 = arith.constant -1 : i32

    // Setup very sparse 3-d tensors.
    %t1 = arith.constant sparse<
       [ [1,1,3], [2,0,0], [2,2,1], [2,2,2], [2,2,3] ], [ 1, 2, 3, 4, 5 ]
    > : tensor<3x3x4xi32>
    %t2 = arith.constant sparse<
       [ [1,0,0], [1,1,3], [2,2,1], [2,2,3] ], [ 6, 7, 8, 9 ]
    > : tensor<3x3x4xi32>
    %st1 = sparse_tensor.convert %t1
      : tensor<3x3x4xi32> to tensor<?x?x?xi32, #SparseTensor>
    %st2 = sparse_tensor.convert %t2
      : tensor<3x3x4xi32> to tensor<?x?x?xi32, #SparseTensor>

    // Call kernel.
    %0 = call @redsum(%st1, %st2)
      : (tensor<?x?x?xi32, #SparseTensor>,
         tensor<?x?x?xi32, #SparseTensor>) -> tensor<?x?xi32, #SparseMatrix>

    //
    // Verify results. Only two entries stored in result. Correct structure.
    //
    // CHECK: ( 7, 69, -1, -1 )
    // CHECK-NEXT: ( ( 0, 0, 0 ), ( 0, 7, 0 ), ( 0, 0, 69 ) )
    //
    %val = sparse_tensor.values %0
      : tensor<?x?xi32, #SparseMatrix> to memref<?xi32>
    %vv = vector.transfer_read %val[%c0], %i0: memref<?xi32>, vector<4xi32>
    vector.print %vv : vector<4xi32>
    %dm = sparse_tensor.convert %0
      : tensor<?x?xi32, #SparseMatrix> to tensor<?x?xi32>
    %db = bufferization.to_memref %dm : memref<?x?xi32>
    %vm = vector.transfer_read %db[%c0, %c0], %i0: memref<?x?xi32>, vector<3x3xi32>
    vector.print %vm : vector<3x3xi32>

    // Release the resources.
    sparse_tensor.release %st1 : tensor<?x?x?xi32, #SparseTensor>
    sparse_tensor.release %st2 : tensor<?x?x?xi32, #SparseTensor>
    sparse_tensor.release %0 : tensor<?x?xi32, #SparseMatrix>
    memref.dealloc %db : memref<?x?xi32>
    return
  }
}
