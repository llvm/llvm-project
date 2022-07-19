// RUN: mlir-opt %s --sparse-compiler | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#SparseVector = #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>

#trait_op = {
  indexing_maps = [
    affine_map<(i) -> (i)>, // a
    affine_map<(i) -> (i)>  // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "x(i) = OP a(i)"
}

module {
  // Performs sign operation (using semi-ring unary op)
  // with semantics that
  // > 0 : +1.0
  // < 0 : -1.0
  // +Inf: +1.0
  // -Inf: -1.0
  // +NaN: +NaN
  // -NaN: -NaN
  // +0.0: +0.0
  // -0.0: -0.0
  func.func @sparse_sign(%arg0: tensor<?xf64, #SparseVector>)
                             -> tensor<?xf64, #SparseVector> {
    %c0 = arith.constant 0 : index
    %d = tensor.dim %arg0, %c0 : tensor<?xf64, #SparseVector>
    %xin = bufferization.alloc_tensor(%d) : tensor<?xf64, #SparseVector>
    %0 = linalg.generic #trait_op
      ins(%arg0: tensor<?xf64, #SparseVector>)
      outs(%xin: tensor<?xf64, #SparseVector>) {
      ^bb0(%a: f64, %x: f64) :
        %result = sparse_tensor.unary %a : f64 to f64
          present={
            ^bb1(%s: f64):
              %z = arith.constant 0.0 : f64
              %1 = arith.cmpf one, %s, %z : f64
              %2 = arith.uitofp %1 : i1 to f64
              %3 = math.copysign %2, %s : f64
              %4 = arith.cmpf uno, %s, %s : f64
              %5 = arith.select %4, %s, %3 : f64
              sparse_tensor.yield %5 : f64
          }
          absent={}
        linalg.yield %result : f64
    } -> tensor<?xf64, #SparseVector>
    return %0 : tensor<?xf64, #SparseVector>
  }

  // Driver method to call and verify sign kernel.
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %du = arith.constant 99.99 : f64

    %pnan = arith.constant 0x7FF0000001000000 : f64
    %nnan = arith.constant 0xFFF0000001000000 : f64
    %pinf = arith.constant 0x7FF0000000000000 : f64
    %ninf = arith.constant 0xFFF0000000000000 : f64

    // Setup sparse vector.
    %v1 = arith.constant sparse<
       [ [0], [3], [5], [11], [13], [17], [18], [20], [21], [28], [29], [31] ],
         [ -1.5, 1.5, -10.2, 11.3, 1.0, -1.0,
           0x7FF0000001000000, // +NaN
           0xFFF0000001000000, // -NaN
           0x7FF0000000000000, // +Inf
           0xFFF0000000000000, // -Inf
           -0.0,               // -Zero
           0.0                 // +Zero
        ]
    > : tensor<32xf64>
    %sv1 = sparse_tensor.convert %v1
         : tensor<32xf64> to tensor<?xf64, #SparseVector>

    // Call sign kernel.
    %0 = call @sparse_sign(%sv1) : (tensor<?xf64, #SparseVector>)
                                 -> tensor<?xf64, #SparseVector>

    //
    // Verify the results.
    //
    // CHECK: ( -1, 1, -1, 1, 1, -1, nan, -nan, 1, -1, -0, 0, 99.99 )
    //
    %1 = sparse_tensor.values %0 : tensor<?xf64, #SparseVector> to memref<?xf64>
    %2 = vector.transfer_read %1[%c0], %du: memref<?xf64>, vector<13xf64>
    vector.print %2 : vector<13xf64>

    // Release the resources.
    bufferization.dealloc_tensor %sv1 : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %0 : tensor<?xf64, #SparseVector>
    return
  }
}


