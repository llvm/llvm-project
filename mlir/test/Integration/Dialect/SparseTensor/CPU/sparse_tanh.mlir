// RUN: mlir-opt %s --sparse-compiler | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#SparseVector = #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>

#trait_op = {
  indexing_maps = [
    affine_map<(i) -> (i)>   // X (out)
  ],
  iterator_types = ["parallel"],
  doc = "X(i) = OP X(i)"
}

module {
  // Performs zero-preserving math to sparse vector.
  func @sparse_tanh(%vec: tensor<?xf64, #SparseVector>
                          {linalg.inplaceable = true})
                       -> tensor<?xf64, #SparseVector> {
    %0 = linalg.generic #trait_op
      outs(%vec: tensor<?xf64, #SparseVector>) {
        ^bb(%x: f64):
          %1 = math.tanh %x : f64
          linalg.yield %1 : f64
    } -> tensor<?xf64, #SparseVector>
    return %0 : tensor<?xf64, #SparseVector>
  }

  // Dumps a sparse vector of type f64.
  func @dump_vec_f64(%arg0: tensor<?xf64, #SparseVector>) {
    // Dump the values array to verify only sparse contents are stored.
    %c0 = arith.constant 0 : index
    %d0 = arith.constant -1.0 : f64
    %0 = sparse_tensor.values %arg0
      : tensor<?xf64, #SparseVector> to memref<?xf64>
    %1 = vector.transfer_read %0[%c0], %d0: memref<?xf64>, vector<32xf64>
    vector.print %1 : vector<32xf64>
    // Dump the dense vector to verify structure is correct.
    %dv = sparse_tensor.convert %arg0
        : tensor<?xf64, #SparseVector> to tensor<?xf64>
    %2 = bufferization.to_memref %dv : memref<?xf64>
    %3 = vector.transfer_read %2[%c0], %d0: memref<?xf64>, vector<32xf64>
    vector.print %3 : vector<32xf64>
    memref.dealloc %2 : memref<?xf64>
    return
  }

  // Driver method to call and verify vector kernels.
  func @entry() {
    // Setup sparse vector.
    %v1 = arith.constant sparse<
       [ [0], [3], [11], [17], [20], [21], [28], [29], [31] ],
         [ -1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0 ]
    > : tensor<32xf64>
    %sv1 = sparse_tensor.convert %v1
         : tensor<32xf64> to tensor<?xf64, #SparseVector>

    // Call sparse vector kernel.
    %0 = call @sparse_tanh(%sv1) : (tensor<?xf64, #SparseVector>)
                                 -> tensor<?xf64, #SparseVector>

    //
    // Verify the results (within some precision).
    //
    // CHECK: {{( -0.761[0-9]*, 0.761[0-9]*, 0.96[0-9]*, 0.99[0-9]*, 0.99[0-9]*, 0.99[0-9]*, 0.99[0-9]*, 0.99[0-9]*, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 )}}
    // CHECK-NEXT {{( -0.761[0-9]*, 0, 0, 0.761[0-9]*, 0, 0, 0, 0, 0, 0, 0, 0.96[0-9]*, 0, 0, 0, 0, 0, 0.99[0-9]*, 0, 0, 0.99[0-9]*, 0.99[0-9]*, 0, 0, 0, 0, 0, 0, 0.99[0-9]*, 0.99[0-9]*, 0, 1 )}}
    //
    call @dump_vec_f64(%sv1) : (tensor<?xf64, #SparseVector>) -> ()

    // Release the resources.
    sparse_tensor.release %sv1 : tensor<?xf64, #SparseVector>
    return
  }
}
