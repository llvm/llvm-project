// RUN: mlir-opt %s --sparse-compiler=enable-runtime-library=true | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

// RUN: mlir-opt %s --sparse-compiler="enable-runtime-library=false enable-buffer-initialization=true" | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#DCSR = #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>
#CSR = #sparse_tensor.encoding<{dimLevelType = ["dense", "compressed"]}>
#CSC = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>
}>

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

  func.func @conv2d_all_sparse_DCSR(%input:  tensor<8x8xi32, #DCSR>,
               %filter: tensor<3x3xi32, #DCSR>) -> tensor<6x6xi32, #DCSR> {
    %s = bufferization.alloc_tensor() : tensor<6x6xi32, #DCSR>
    %0 = linalg.conv_2d
      ins  (%input, %filter: tensor<8x8xi32, #DCSR>, tensor<3x3xi32, #DCSR>)
      outs (%s: tensor<6x6xi32, #DCSR>) -> tensor<6x6xi32, #DCSR>
    return %0 : tensor<6x6xi32, #DCSR>
  }

  func.func @conv2d_all_sparse_CSR(%input:  tensor<8x8xi32, #CSR>,
               %filter: tensor<3x3xi32, #CSR>) -> tensor<6x6xi32, #CSR> {
    %s = bufferization.alloc_tensor() : tensor<6x6xi32, #CSR>
    %0 = linalg.conv_2d
      ins  (%input, %filter: tensor<8x8xi32, #CSR>, tensor<3x3xi32, #CSR>)
      outs (%s: tensor<6x6xi32, #CSR>) -> tensor<6x6xi32, #CSR>
    return %0 : tensor<6x6xi32, #CSR>
  }

  func.func @conv2d_all_sparse_CSC(%input:  tensor<8x8xi32, #CSC>,
               %filter: tensor<3x3xi32, #CSC>) -> tensor<6x6xi32, #CSC> {
    %s = bufferization.alloc_tensor() : tensor<6x6xi32, #CSC>
    %0 = linalg.conv_2d
      ins  (%input, %filter: tensor<8x8xi32, #CSC>, tensor<3x3xi32, #CSC>)
      outs (%s: tensor<6x6xi32, #CSC>) -> tensor<6x6xi32, #CSC>
    return %0 : tensor<6x6xi32, #CSC>
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
    %sparse_filter_DCSR = sparse_tensor.convert %filter
      : tensor<3x3xi32> to tensor<3x3xi32, #DCSR>
    %sparse_filter_CSR = sparse_tensor.convert %filter
      : tensor<3x3xi32> to tensor<3x3xi32, #CSR>
    %sparse_filter_CSC = sparse_tensor.convert %filter
      : tensor<3x3xi32> to tensor<3x3xi32, #CSC>


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
    %sparse_input_DCSR = sparse_tensor.convert %input
      : tensor<8x8xi32> to tensor<8x8xi32, #DCSR>
    %sparse_input_CSR = sparse_tensor.convert %input
      : tensor<8x8xi32> to tensor<8x8xi32, #CSR>
    %sparse_input_CSC = sparse_tensor.convert %input
      : tensor<8x8xi32> to tensor<8x8xi32, #CSC>

    // Call the kernel.
    %output = arith.constant dense<0> : tensor<6x6xi32>
    %0 = call @conv2d(%input, %sparse_filter_DCSR, %output)
       : (tensor<8x8xi32>,
          tensor<3x3xi32, #DCSR>, tensor<6x6xi32>) -> tensor<6x6xi32>
    %1 = call @conv2d_sparse_out(%input, %sparse_filter_DCSR)
       : (tensor<8x8xi32>,
          tensor<3x3xi32, #DCSR>) -> tensor<6x6xi32, #DCSR>
    %2 = call @conv2d_all_sparse_DCSR(%sparse_input_DCSR, %sparse_filter_DCSR)
       : (tensor<8x8xi32, #DCSR>,
          tensor<3x3xi32, #DCSR>) -> tensor<6x6xi32, #DCSR>
    %3 = call @conv2d_all_sparse_CSR(%sparse_input_CSR, %sparse_filter_CSR)
       : (tensor<8x8xi32, #CSR>,
          tensor<3x3xi32, #CSR>) -> tensor<6x6xi32, #CSR>
    %4 = call @conv2d_all_sparse_CSC(%sparse_input_CSC, %sparse_filter_CSC)
       : (tensor<8x8xi32, #CSC>,
          tensor<3x3xi32, #CSC>) -> tensor<6x6xi32, #CSC>


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

    //
    // Should be the same as dense output
    // CHECK:    ( ( 0, 0, -1, -6, -1, 6 ),
    // CHECK-SAME: ( -1, 0, 1, 0, 1, 0 ),
    // CHECK-SAME: ( 0, -1, 1, 0, 0, 0 ),
    // CHECK-SAME: ( -1, 0, 0, 0, 0, 0 ),
    // CHECK-SAME: ( 0, 0, 3, 6, -3, -6 ),
    // CHECK-SAME: ( 2, -1, 3, 0, -3, 0 ) )
    //
    %all_sparse_DCSR = sparse_tensor.convert %2
      : tensor<6x6xi32, #DCSR> to tensor<6x6xi32>
    %v2 = vector.transfer_read %all_sparse_DCSR[%c0, %c0], %i0
      : tensor<6x6xi32>, vector<6x6xi32>
    vector.print %v2 : vector<6x6xi32>

    //
    // Should be the same as dense output
    // CHECK:    ( ( 0, 0, -1, -6, -1, 6 ),
    // CHECK-SAME: ( -1, 0, 1, 0, 1, 0 ),
    // CHECK-SAME: ( 0, -1, 1, 0, 0, 0 ),
    // CHECK-SAME: ( -1, 0, 0, 0, 0, 0 ),
    // CHECK-SAME: ( 0, 0, 3, 6, -3, -6 ),
    // CHECK-SAME: ( 2, -1, 3, 0, -3, 0 ) )
    //
    %all_sparse_CSR = sparse_tensor.convert %3
      : tensor<6x6xi32, #CSR> to tensor<6x6xi32>
    %v3 = vector.transfer_read %all_sparse_CSR[%c0, %c0], %i0
      : tensor<6x6xi32>, vector<6x6xi32>
    vector.print %v3 : vector<6x6xi32>

    //
    // Should be the same as dense output
    // CHECK:    ( ( 0, 0, -1, -6, -1, 6 ),
    // CHECK-SAME: ( -1, 0, 1, 0, 1, 0 ),
    // CHECK-SAME: ( 0, -1, 1, 0, 0, 0 ),
    // CHECK-SAME: ( -1, 0, 0, 0, 0, 0 ),
    // CHECK-SAME: ( 0, 0, 3, 6, -3, -6 ),
    // CHECK-SAME: ( 2, -1, 3, 0, -3, 0 ) )
    //
    %all_sparse_CSC = sparse_tensor.convert %4
      : tensor<6x6xi32, #CSC> to tensor<6x6xi32>
    %v4 = vector.transfer_read %all_sparse_CSC[%c0, %c0], %i0
      : tensor<6x6xi32>, vector<6x6xi32>
    vector.print %v4 : vector<6x6xi32>

    // Release the resources.
    bufferization.dealloc_tensor %sparse_filter_DCSR : tensor<3x3xi32, #DCSR>
    bufferization.dealloc_tensor %sparse_filter_CSR : tensor<3x3xi32, #CSR>
    bufferization.dealloc_tensor %sparse_filter_CSC : tensor<3x3xi32, #CSC>

    bufferization.dealloc_tensor %sparse_input_DCSR : tensor<8x8xi32, #DCSR>
    bufferization.dealloc_tensor %sparse_input_CSR : tensor<8x8xi32, #CSR>
    bufferization.dealloc_tensor %sparse_input_CSC : tensor<8x8xi32, #CSC>

    bufferization.dealloc_tensor %1 : tensor<6x6xi32, #DCSR>
    bufferization.dealloc_tensor %2 : tensor<6x6xi32, #DCSR>
    bufferization.dealloc_tensor %3 : tensor<6x6xi32, #CSR>
    bufferization.dealloc_tensor %4 : tensor<6x6xi32, #CSC>
    return
  }
}
