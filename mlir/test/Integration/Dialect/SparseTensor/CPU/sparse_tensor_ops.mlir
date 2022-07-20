// RUN: mlir-opt %s --sparse-compiler | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#ST1 = #sparse_tensor.encoding<{dimLevelType = ["compressed", "compressed", "compressed"]}>
#ST2 = #sparse_tensor.encoding<{dimLevelType = ["compressed", "compressed", "dense"]}>

//
// Trait for 3-d tensor operation.
//
#trait_scale = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,j,k)>,  // A (in)
    affine_map<(i,j,k) -> (i,j,k)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel", "parallel"],
  doc = "X(i,j,k) = A(i,j,k) * 2.0"
}

module {
  // Scales a sparse tensor into a new sparse tensor.
  func.func @tensor_scale(%arga: tensor<?x?x?xf64, #ST1>) -> tensor<?x?x?xf64, #ST2> {
    %s = arith.constant 2.0 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %d0 = tensor.dim %arga, %c0 : tensor<?x?x?xf64, #ST1>
    %d1 = tensor.dim %arga, %c1 : tensor<?x?x?xf64, #ST1>
    %d2 = tensor.dim %arga, %c2 : tensor<?x?x?xf64, #ST1>
    %xm = bufferization.alloc_tensor(%d0, %d1, %d2) : tensor<?x?x?xf64, #ST2>
    %0 = linalg.generic #trait_scale
       ins(%arga: tensor<?x?x?xf64, #ST1>)
        outs(%xm: tensor<?x?x?xf64, #ST2>) {
        ^bb(%a: f64, %x: f64):
          %1 = arith.mulf %a, %s : f64
          linalg.yield %1 : f64
    } -> tensor<?x?x?xf64, #ST2>
    return %0 : tensor<?x?x?xf64, #ST2>
  }

  // Driver method to call and verify tensor kernel.
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %d1 = arith.constant -1.0 : f64

    // Setup sparse tensor.
    %t = arith.constant dense<
      [ [ [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0 ] ],
        [ [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ],
        [ [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
          [0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0 ],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ] ]> : tensor<3x4x8xf64>
    %st = sparse_tensor.convert %t : tensor<3x4x8xf64> to tensor<?x?x?xf64, #ST1>

    // Call sparse vector kernels.
    %0 = call @tensor_scale(%st) : (tensor<?x?x?xf64, #ST1>) -> tensor<?x?x?xf64, #ST2>

    // Sanity check on stored values.
    //
    // CHECK:      ( 1, 2, 3, 4, 5, -1, -1, -1 )
    // CHECK-NEXT: ( 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 6, 8, 0, 0, 0, 0, 10, -1, -1, -1, -1, -1, -1, -1, -1 )
    %m1 = sparse_tensor.values %st : tensor<?x?x?xf64, #ST1> to memref<?xf64>
    %m2 = sparse_tensor.values %0  : tensor<?x?x?xf64, #ST2> to memref<?xf64>
    %v1 = vector.transfer_read %m1[%c0], %d1: memref<?xf64>, vector<8xf64>
    %v2 = vector.transfer_read %m2[%c0], %d1: memref<?xf64>, vector<32xf64>
    vector.print %v1 : vector<8xf64>
    vector.print %v2 : vector<32xf64>

    // Release the resources.
    bufferization.dealloc_tensor %st : tensor<?x?x?xf64, #ST1>
    bufferization.dealloc_tensor %0  : tensor<?x?x?xf64, #ST2>
    return
  }
}
