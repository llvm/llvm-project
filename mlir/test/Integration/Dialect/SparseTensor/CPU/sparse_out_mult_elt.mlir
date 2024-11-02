// RUN: mlir-opt %s --sparse-compiler | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#DCSR = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ]
}>

#trait_mult_elt = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A
    affine_map<(i,j) -> (i,j)>,  // B
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(i,j) * B(i,j)"
}

module {
  // Sparse kernel.
  func.func @sparse_mult_elt(
      %arga: tensor<32x16xf32, #DCSR>, %argb: tensor<32x16xf32, #DCSR>) -> tensor<32x16xf32, #DCSR> {
    %argx = bufferization.alloc_tensor() : tensor<32x16xf32, #DCSR>
    %0 = linalg.generic #trait_mult_elt
      ins(%arga, %argb: tensor<32x16xf32, #DCSR>, tensor<32x16xf32, #DCSR>)
      outs(%argx: tensor<32x16xf32, #DCSR>) {
        ^bb(%a: f32, %b: f32, %x: f32):
          %1 = arith.mulf %a, %b : f32
          linalg.yield %1 : f32
    } -> tensor<32x16xf32, #DCSR>
    return %0 : tensor<32x16xf32, #DCSR>
  }

  // Driver method to call and verify kernel.
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %f1 = arith.constant -1.0 : f32

    // Setup very sparse matrices.
    %ta = arith.constant sparse<
       [ [2,2], [15,15], [31,0], [31,14] ], [ 2.0, 3.0, -2.0, 4.0 ]
    > : tensor<32x16xf32>
    %tb = arith.constant sparse<
       [ [1,1], [2,0], [2,2], [2,15], [31,0], [31,15] ], [ 5.0, 6.0, 7.0, 8.0, -10.0, 9.0 ]
    > : tensor<32x16xf32>
    %sta = sparse_tensor.convert %ta
      : tensor<32x16xf32> to tensor<32x16xf32, #DCSR>
    %stb = sparse_tensor.convert %tb
      : tensor<32x16xf32> to tensor<32x16xf32, #DCSR>

    // Call kernel.
    %0 = call @sparse_mult_elt(%sta, %stb)
      : (tensor<32x16xf32, #DCSR>,
         tensor<32x16xf32, #DCSR>) -> tensor<32x16xf32, #DCSR>

    //
    // Verify results. Only two entries stored in result!
    //
    // CHECK: ( 14, 20, -1, -1 )
    //
    %val = sparse_tensor.values %0 : tensor<32x16xf32, #DCSR> to memref<?xf32>
    %vv = vector.transfer_read %val[%c0], %f1: memref<?xf32>, vector<4xf32>
    vector.print %vv : vector<4xf32>

    // Release the resources.
    bufferization.dealloc_tensor %sta : tensor<32x16xf32, #DCSR>
    bufferization.dealloc_tensor %stb : tensor<32x16xf32, #DCSR>
    bufferization.dealloc_tensor %0   : tensor<32x16xf32, #DCSR>
    return
  }
}
