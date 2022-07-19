// RUN: mlir-opt %s --sparse-compiler | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#DCSR = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ]
}>

#DCSC = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>
}>

#transpose_trait = {
  indexing_maps = [
    affine_map<(i,j) -> (j,i)>,  // A
    affine_map<(i,j) -> (i,j)>   // X
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(j,i)"
}

module {

  //
  // Transposing a sparse row-wise matrix into another sparse row-wise
  // matrix introduces a cycle in the iteration graph. This complication
  // can be avoided by manually inserting a conversion of the incoming
  // matrix into a sparse column-wise matrix first.
  //
  func.func @sparse_transpose(%arga: tensor<3x4xf64, #DCSR>)
                                  -> tensor<4x3xf64, #DCSR> {
    %t = sparse_tensor.convert %arga
      : tensor<3x4xf64, #DCSR> to tensor<3x4xf64, #DCSC>

    %i = bufferization.alloc_tensor() : tensor<4x3xf64, #DCSR>
    %0 = linalg.generic #transpose_trait
       ins(%t: tensor<3x4xf64, #DCSC>)
       outs(%i: tensor<4x3xf64, #DCSR>) {
       ^bb(%a: f64, %x: f64):
         linalg.yield %a : f64
    } -> tensor<4x3xf64, #DCSR>

    bufferization.dealloc_tensor %t : tensor<3x4xf64, #DCSC>

    return %0 : tensor<4x3xf64, #DCSR>
  }

  //
  // However, even better, the sparse compiler is able to insert such a
  // conversion automatically to resolve a cycle in the iteration graph!
  //
  func.func @sparse_transpose_auto(%arga: tensor<3x4xf64, #DCSR>)
                                       -> tensor<4x3xf64, #DCSR> {
    %i = bufferization.alloc_tensor() : tensor<4x3xf64, #DCSR>
    %0 = linalg.generic #transpose_trait
       ins(%arga: tensor<3x4xf64, #DCSR>)
       outs(%i: tensor<4x3xf64, #DCSR>) {
       ^bb(%a: f64, %x: f64):
         linalg.yield %a : f64
    } -> tensor<4x3xf64, #DCSR>
    return %0 : tensor<4x3xf64, #DCSR>
  }

  //
  // Main driver.
  //
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %du = arith.constant 0.0 : f64

    // Setup input sparse matrix from compressed constant.
    %d = arith.constant dense <[
       [ 1.1,  1.2,  0.0,  1.4 ],
       [ 0.0,  0.0,  0.0,  0.0 ],
       [ 3.1,  0.0,  3.3,  3.4 ]
    ]> : tensor<3x4xf64>
    %a = sparse_tensor.convert %d : tensor<3x4xf64> to tensor<3x4xf64, #DCSR>

    // Call the kernels.
    %0 = call @sparse_transpose(%a)
      : (tensor<3x4xf64, #DCSR>) -> tensor<4x3xf64, #DCSR>
    %1 = call @sparse_transpose_auto(%a)
      : (tensor<3x4xf64, #DCSR>) -> tensor<4x3xf64, #DCSR>

    //
    // Verify result.
    //
    // CHECK:      ( 1.1, 0, 3.1 )
    // CHECK-NEXT: ( 1.2, 0, 0 )
    // CHECK-NEXT: ( 0, 0, 3.3 )
    // CHECK-NEXT: ( 1.4, 0, 3.4 )
    //
    // CHECK-NEXT: ( 1.1, 0, 3.1 )
    // CHECK-NEXT: ( 1.2, 0, 0 )
    // CHECK-NEXT: ( 0, 0, 3.3 )
    // CHECK-NEXT: ( 1.4, 0, 3.4 )
    //
    %x = sparse_tensor.convert %0 : tensor<4x3xf64, #DCSR> to tensor<4x3xf64>
    scf.for %i = %c0 to %c4 step %c1 {
      %v1 = vector.transfer_read %x[%i, %c0], %du: tensor<4x3xf64>, vector<3xf64>
      vector.print %v1 : vector<3xf64>
    }
    %y = sparse_tensor.convert %1 : tensor<4x3xf64, #DCSR> to tensor<4x3xf64>
    scf.for %i = %c0 to %c4 step %c1 {
      %v2 = vector.transfer_read %y[%i, %c0], %du: tensor<4x3xf64>, vector<3xf64>
      vector.print %v2 : vector<3xf64>
    }

    // Release resources.
    bufferization.dealloc_tensor %a : tensor<3x4xf64, #DCSR>
    bufferization.dealloc_tensor %0 : tensor<4x3xf64, #DCSR>
    bufferization.dealloc_tensor %1 : tensor<4x3xf64, #DCSR>

    return
  }
}
