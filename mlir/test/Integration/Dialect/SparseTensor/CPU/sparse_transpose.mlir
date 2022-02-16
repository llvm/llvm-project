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
  // matrix would fail direct codegen, since it introduces a cycle in
  // the iteration graph. This can be avoided by converting the incoming
  // matrix into a sparse column-wise matrix first.
  //
  func @sparse_transpose(%arga: tensor<3x4xf64, #DCSR>) -> tensor<4x3xf64, #DCSR> {
    %t = sparse_tensor.convert %arga : tensor<3x4xf64, #DCSR> to tensor<3x4xf64, #DCSC>

    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %i = sparse_tensor.init [%c4, %c3] : tensor<4x3xf64, #DCSR>

    %0 = linalg.generic #transpose_trait
       ins(%t: tensor<3x4xf64, #DCSC>)
       outs(%i: tensor<4x3xf64, #DCSR>) {
       ^bb(%a: f64, %x: f64):
         linalg.yield %a : f64
     } -> tensor<4x3xf64, #DCSR>

     sparse_tensor.release %t : tensor<3x4xf64, #DCSC>

     return %0 : tensor<4x3xf64, #DCSR>
  }

  //
  // Main driver.
  //
  func @entry() {
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

    // Call the kernel.
    %0 = call @sparse_transpose(%a) : (tensor<3x4xf64, #DCSR>) -> tensor<4x3xf64, #DCSR>

    //
    // Verify result.
    //
    // CHECK:      ( 1.1, 0, 3.1 )
    // CHECK-NEXT: ( 1.2, 0, 0 )
    // CHECK-NEXT: ( 0, 0, 3.3 )
    // CHECK-NEXT: ( 1.4, 0, 3.4 )
    //
    %x = sparse_tensor.convert %0 : tensor<4x3xf64, #DCSR> to tensor<4x3xf64>
    %m = bufferization.to_memref %x : memref<4x3xf64>
    scf.for %i = %c0 to %c4 step %c1 {
      %v = vector.transfer_read %m[%i, %c0], %du: memref<4x3xf64>, vector<3xf64>
      vector.print %v : vector<3xf64>
    }

    // Release resources.
    sparse_tensor.release %a : tensor<3x4xf64, #DCSR>
    sparse_tensor.release %0 : tensor<4x3xf64, #DCSR>
    memref.dealloc %m : memref<4x3xf64>

    return
  }
}
