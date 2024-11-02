// RUN: mlir-opt %s --sparse-compiler | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>
#DenseVector = #sparse_tensor.encoding<{dimLevelType = ["dense"]}>

#trait_vec_op = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a (in)
    affine_map<(i) -> (i)>,  // b (in)
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"]
}

module {
  // Creates a dense vector using the minimum values from two input sparse vectors.
  // When there is no overlap, include the present value in the output.
  func.func @vector_min(%arga: tensor<?xf16, #SparseVector>,
                        %argb: tensor<?xf16, #SparseVector>) -> tensor<?xf16, #DenseVector> {
    %c = arith.constant 0 : index
    %d = tensor.dim %arga, %c : tensor<?xf16, #SparseVector>
    %xv = bufferization.alloc_tensor (%d) : tensor<?xf16, #DenseVector>
    %0 = linalg.generic #trait_vec_op
       ins(%arga, %argb: tensor<?xf16, #SparseVector>, tensor<?xf16, #SparseVector>)
        outs(%xv: tensor<?xf16, #DenseVector>) {
        ^bb(%a: f16, %b: f16, %x: f16):
          %1 = sparse_tensor.binary %a, %b : f16, f16 to f16
            overlap={
              ^bb0(%a0: f16, %b0: f16):
                %cmp = arith.cmpf "olt", %a0, %b0 : f16
                %2 = arith.select %cmp, %a0, %b0: f16
                sparse_tensor.yield %2 : f16
            }
            left=identity
            right=identity
          linalg.yield %1 : f16
    } -> tensor<?xf16, #DenseVector>
    return %0 : tensor<?xf16, #DenseVector>
  }

  // Dumps a dense vector of type f16.
  func.func @dump_vec(%arg0: tensor<?xf16, #DenseVector>) {
    // Dump the values array to verify only sparse contents are stored.
    %c0 = arith.constant 0 : index
    %d0 = arith.constant -1.0 : f16
    %0 = sparse_tensor.values %arg0 : tensor<?xf16, #DenseVector> to memref<?xf16>
    %1 = vector.transfer_read %0[%c0], %d0: memref<?xf16>, vector<32xf16>
    %f1 = arith.extf %1: vector<32xf16> to vector<32xf32>
    vector.print %f1 : vector<32xf32>
    return
  }

  // Driver method to call and verify the kernel.
  func.func @entry() {
    %c0 = arith.constant 0 : index

    // Setup sparse vectors.
    %v1 = arith.constant sparse<
       [ [0], [3], [11], [17], [20], [21], [28], [29], [31] ],
         [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 ]
    > : tensor<32xf16>
    %v2 = arith.constant sparse<
       [ [1], [3], [4], [10], [16], [18], [21], [28], [29], [31] ],
         [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0 ]
    > : tensor<32xf16>
    %sv1 = sparse_tensor.convert %v1 : tensor<32xf16> to tensor<?xf16, #SparseVector>
    %sv2 = sparse_tensor.convert %v2 : tensor<32xf16> to tensor<?xf16, #SparseVector>

    // Call the sparse vector kernel.
    %0 = call @vector_min(%sv1, %sv2)
       : (tensor<?xf16, #SparseVector>,
          tensor<?xf16, #SparseVector>) -> tensor<?xf16, #DenseVector>

    //
    // Verify the result.
    //
    // CHECK: ( 1, 11, 0, 2, 13, 0, 0, 0, 0, 0, 14, 3, 0, 0, 0, 0, 15, 4, 16, 0, 5, 6, 0, 0, 0, 0, 0, 0, 7, 8, 0, 9 )
    call @dump_vec(%0) : (tensor<?xf16, #DenseVector>) -> ()

    // Release the resources.
    bufferization.dealloc_tensor %sv1 : tensor<?xf16, #SparseVector>
    bufferization.dealloc_tensor %sv2 : tensor<?xf16, #SparseVector>
    bufferization.dealloc_tensor %0 : tensor<?xf16, #DenseVector>
    return
  }
}
