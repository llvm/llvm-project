// DEFINE: %{option} = "enable-runtime-library=false enable-index-reduction=true"
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler=%{option}
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_c_runner_utils | \
// DEFINE: FileCheck %s
//
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{option} = "enable-runtime-library=true enable-buffer-initialization=true enable-index-reduction=true"
// RUN: %{compile} | %{run}

#CCCC = #sparse_tensor.encoding<{ lvlTypes = [ "compressed", "compressed", "compressed", "compressed" ], posWidth = 32, crdWidth = 32 }>

func.func @pooling_nhwc_sum_CCCC(%input: tensor<1x4x4x1xf32, #CCCC>, %filter: tensor<2x2xf32>) -> tensor<1x3x3x1xf32, #CCCC> {
  %init = bufferization.alloc_tensor() : tensor<1x3x3x1xf32, #CCCC>
  %0 = linalg.pooling_nhwc_sum {dilations = dense<1> : tensor<2xi64>,
                                strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<1x4x4x1xf32, #CCCC>, tensor<2x2xf32>)
    outs (%init: tensor<1x3x3x1xf32, #CCCC>) -> tensor<1x3x3x1xf32, #CCCC>
  return %0 : tensor<1x3x3x1xf32, #CCCC>
}

func.func @pooling_nhwc_sum(%input: tensor<1x4x4x1xf32>, %filter: tensor<2x2xf32>) -> tensor<1x3x3x1xf32> {
  %init = arith.constant dense<[[ [[0.0], [0.0], [0.0]],
                                  [[0.0], [0.0], [0.0]],
                                  [[0.0], [0.0], [0.0]] ]]> : tensor<1x3x3x1xf32>
  %0 = linalg.pooling_nhwc_sum {dilations = dense<1> : tensor<2xi64>,
                                strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<1x4x4x1xf32>, tensor<2x2xf32>)
    outs (%init: tensor<1x3x3x1xf32>) -> tensor<1x3x3x1xf32>
  return %0 : tensor<1x3x3x1xf32>
}


func.func @entry() {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.00000e+00 : f32

  %filter = arith.constant dense<
     [[  1.0,  0.0],
      [  0.0,  1.0]]
  > : tensor<2x2xf32>

  %in_dense = arith.constant dense<
     [[[[1.0],  [2.0],  [1.0],  [2.0]],
       [[1.0],  [2.0],  [1.0],  [2.0]],
       [[1.0],  [2.0],  [1.0],  [2.0]],
       [[1.0],  [2.0],  [1.0],  [2.0]]]]
  > : tensor<1x4x4x1xf32>

  %in_CCCC = sparse_tensor.convert %in_dense : tensor<1x4x4x1xf32> to tensor<1x4x4x1xf32, #CCCC>

  %dense_ret = call @pooling_nhwc_sum(%in_dense, %filter) : (tensor<1x4x4x1xf32>, tensor<2x2xf32>) -> tensor<1x3x3x1xf32>
  %CCCC_ret = call @pooling_nhwc_sum_CCCC(%in_CCCC, %filter) : (tensor<1x4x4x1xf32, #CCCC>, tensor<2x2xf32>) -> tensor<1x3x3x1xf32, #CCCC>

  // CHECK: ( ( ( ( 6 ), ( 6 ), ( 6 ) ), ( ( 6 ), ( 6 ), ( 6 ) ), ( ( 6 ), ( 6 ), ( 6 ) ) ) )
  %dense_v = vector.transfer_read %dense_ret[%c0, %c0, %c0, %c0], %zero
      : tensor<1x3x3x1xf32>, vector<1x3x3x1xf32>
  vector.print %dense_v : vector<1x3x3x1xf32>

  //
  // Sparse pooling should have the same output.
  //

  // CHECK-NEXT: ( ( ( ( 6 ), ( 6 ), ( 6 ) ), ( ( 6 ), ( 6 ), ( 6 ) ), ( ( 6 ), ( 6 ), ( 6 ) ) ) )
  %s1 = sparse_tensor.convert %CCCC_ret : tensor<1x3x3x1xf32, #CCCC> to tensor<1x3x3x1xf32>
  %v1 = vector.transfer_read %s1[%c0, %c0, %c0, %c0], %zero
      : tensor<1x3x3x1xf32>, vector<1x3x3x1xf32>
  vector.print %v1 : vector<1x3x3x1xf32>

  // Releases resources.
  bufferization.dealloc_tensor %in_CCCC : tensor<1x4x4x1xf32, #CCCC>
  bufferization.dealloc_tensor %CCCC_ret : tensor<1x3x3x1xf32, #CCCC>
  bufferization.dealloc_tensor %dense_ret : tensor<1x3x3x1xf32>
  return
}
