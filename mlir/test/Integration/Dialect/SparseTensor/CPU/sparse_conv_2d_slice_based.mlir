// DEFINE: %{option} = "enable-index-reduction=true enable-runtime-library=false"
// DEFINE: %{command} = mlir-opt %s --sparse-compiler=%{option} | \
// DEFINE: mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// DEFINE: FileCheck %s
//
// RUN: %{command}

#map = affine_map<(d0, d1, d2, d3) -> (d0 + d2, d1 + d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

#DCSR = #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>

module {
  func.func @conv2d_all_sparse_CSR(%arg0: tensor<8x8xi32, #DCSR>, %arg1: tensor<3x3xi32>) -> tensor<6x6xi32, #DCSR> {
    %0 = bufferization.alloc_tensor() : tensor<6x6xi32, #DCSR>
    %1 = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    ins(%arg0, %arg1 : tensor<8x8xi32, #DCSR>, tensor<3x3xi32>)
    outs(%0 : tensor<6x6xi32, #DCSR>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %2 = arith.muli %in, %in_0 : i32
      %3 = arith.addi %out, %2 : i32
      linalg.yield %3 : i32
    } -> tensor<6x6xi32, #DCSR>
    return %1 : tensor<6x6xi32, #DCSR>
  }

  func.func @entry() {
    %c0 = arith.constant 0 : index
    %i0 = arith.constant 0 : i32

    // A typical edge detection filter.
    %filter = arith.constant dense<[
      [  1,  0, -1 ],
      [  0,  0,  0 ],
      [ -1,  0,  1 ]
    ]> : tensor<3x3xi32>

    %input = arith.constant dense<[
      [  1,  2,  3,  4,  0,  6,  7,  8 ],
      [  2,  2,  4,  4,  0,  0,  6,  8 ],
      [  2,  2,  4,  4,  0,  0,  6,  8 ],
      [  2,  2,  3,  4,  0,  0,  7,  8 ],
      [  1,  3,  3,  4,  0,  0,  6,  8 ],
      [  3,  2,  3,  4,  0,  0,  7,  8 ],
      [  1,  3,  3,  4,  3,  6,  6,  8 ],
      [  1,  3,  3,  4,  3,  0,  7,  8 ]
    ]> : tensor<8x8xi32>

    %sparse_filter_CSR = sparse_tensor.convert %filter
      : tensor<3x3xi32> to tensor<3x3xi32>

    %sparse_input_CSR = sparse_tensor.convert %input
      : tensor<8x8xi32> to tensor<8x8xi32, #DCSR>

    %3 = call @conv2d_all_sparse_CSR(%sparse_input_CSR, %sparse_filter_CSR)
         : (tensor<8x8xi32, #DCSR>,
            tensor<3x3xi32>) -> tensor<6x6xi32, #DCSR>

    %out = sparse_tensor.convert %3
      : tensor<6x6xi32, #DCSR> to tensor<6x6xi32>
    //
    // CHECK:    ( ( 0, 0, -1, -6, -1, 6 ),
    // CHECK-SAME: ( -1, 0, 1, 0, 1, 0 ),
    // CHECK-SAME: ( 0, -1, 1, 0, 0, 0 ),
    // CHECK-SAME: ( -1, 0, 0, 0, 0, 0 ),
    // CHECK-SAME: ( 0, 0, 3, 6, -3, -6 ),
    // CHECK-SAME: ( 2, -1, 3, 0, -3, 0 ) )
    //
    %v2 = vector.transfer_read %out[%c0, %c0], %i0
      : tensor<6x6xi32>, vector<6x6xi32>
    vector.print %v2 : vector<6x6xi32>

    bufferization.dealloc_tensor %sparse_input_CSR : tensor<8x8xi32, #DCSR>
    bufferization.dealloc_tensor %3 : tensor<6x6xi32, #DCSR>
    return
  }

}
