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
// REDEFINE: %{option} = enable-runtime-library=false
// RUN: %{command}

#DCSR = #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>

// An example of a 2D convolution with a sparse filter.
module {

  func.func @conv2d(%input:  tensor<8x8xi32>,
               %filter: tensor<3x3xi32, #DCSR>,
               %output: tensor<6x6xi32>) -> tensor<6x6xi32> {
    %0 = linalg.conv_2d
      ins  (%input, %filter: tensor<8x8xi32>, tensor<3x3xi32, #DCSR>)
      outs (%output: tensor<6x6xi32>) -> tensor<6x6xi32>
    return %0 : tensor<6x6xi32>
  }

  func.func @conv2d_sparse_out(%input:  tensor<8x8xi32>,
               %filter: tensor<3x3xi32, #DCSR>) -> tensor<6x6xi32, #DCSR> {
    %s = bufferization.alloc_tensor() : tensor<6x6xi32, #DCSR>
    %0 = linalg.conv_2d
      ins  (%input, %filter: tensor<8x8xi32>, tensor<3x3xi32, #DCSR>)
      outs (%s: tensor<6x6xi32, #DCSR>) -> tensor<6x6xi32, #DCSR>
    return %0 : tensor<6x6xi32, #DCSR>
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
    %sparse_filter = sparse_tensor.convert %filter
      : tensor<3x3xi32> to tensor<3x3xi32, #DCSR>

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

    // Call the kernel.
    %output = arith.constant dense<0> : tensor<6x6xi32>
    %0 = call @conv2d(%input, %sparse_filter, %output)
       : (tensor<8x8xi32>,
          tensor<3x3xi32, #DCSR>, tensor<6x6xi32>) -> tensor<6x6xi32>
    %1 = call @conv2d_sparse_out(%input, %sparse_filter)
       : (tensor<8x8xi32>,
          tensor<3x3xi32, #DCSR>) -> tensor<6x6xi32, #DCSR>

    // Verify the output.
    //
    // CHECK:    ( ( 0, 0, -1, -6, -1, 6 ),
    // CHECK-SAME: ( -1, 0, 1, 0, 1, 0 ),
    // CHECK-SAME: ( 0, -1, 1, 0, 0, 0 ),
    // CHECK-SAME: ( -1, 0, 0, 0, 0, 0 ),
    // CHECK-SAME: ( 0, 0, 3, 6, -3, -6 ),
    // CHECK-SAME: ( 2, -1, 3, 0, -3, 0 ) )
    //
    %v = vector.transfer_read %0[%c0, %c0], %i0
      : tensor<6x6xi32>, vector<6x6xi32>
    vector.print %v : vector<6x6xi32>

    //
    // Should be the same as dense output
    // CHECK:    ( ( 0, 0, -1, -6, -1, 6 ),
    // CHECK-SAME: ( -1, 0, 1, 0, 1, 0 ),
    // CHECK-SAME: ( 0, -1, 1, 0, 0, 0 ),
    // CHECK-SAME: ( -1, 0, 0, 0, 0, 0 ),
    // CHECK-SAME: ( 0, 0, 3, 6, -3, -6 ),
    // CHECK-SAME: ( 2, -1, 3, 0, -3, 0 ) )
    //
    %sparse_ret = sparse_tensor.convert %1
      : tensor<6x6xi32, #DCSR> to tensor<6x6xi32>
    %v1 = vector.transfer_read %sparse_ret[%c0, %c0], %i0
      : tensor<6x6xi32>, vector<6x6xi32>
    vector.print %v1 : vector<6x6xi32>

    // Release the resources.
    bufferization.dealloc_tensor %sparse_filter : tensor<3x3xi32, #DCSR>
    bufferization.dealloc_tensor %1 : tensor<6x6xi32, #DCSR>
    return
  }
}
