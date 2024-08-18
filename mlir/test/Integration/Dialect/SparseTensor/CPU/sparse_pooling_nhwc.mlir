//--------------------------------------------------------------------------------------------------
// WHEN CREATING A NEW TEST, PLEASE JUST COPY & PASTE WITHOUT EDITS.
//
// Set-up that's shared across all tests in this directory. In principle, this
// config could be moved to lit.local.cfg. However, there are downstream users that
//  do not use these LIT config files. Hence why this is kept inline.
//
// DEFINE: %{sparsifier_opts} = enable-runtime-library=true
// DEFINE: %{sparsifier_opts_sve} = enable-arm-sve=true %{sparsifier_opts}
// DEFINE: %{compile} = mlir-opt %s --sparsifier="%{sparsifier_opts}"
// DEFINE: %{compile_sve} = mlir-opt %s --sparsifier="%{sparsifier_opts_sve}"
// DEFINE: %{run_libs} = -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils
// DEFINE: %{run_libs_sve} = -shared-libs=%native_mlir_runner_utils,%native_mlir_c_runner_utils
// DEFINE: %{run_opts} = -e main -entry-point-result=void
// DEFINE: %{run} = mlir-cpu-runner %{run_opts} %{run_libs}
// DEFINE: %{run_sve} = %mcr_aarch64_cmd --march=aarch64 --mattr="+sve" %{run_opts} %{run_libs_sve}
//
// DEFINE: %{env} =
//--------------------------------------------------------------------------------------------------

// REDEFINE: %{sparsifier_opts} = enable-runtime-library=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false enable-buffer-initialization=true
// RUN: %{compile} | %{run} | FileCheck %s

#CCCC = #sparse_tensor.encoding<{  map = (d0, d1, d2, d3) -> (d0 : compressed, d1 : compressed, d2 : compressed, d3 : compressed), posWidth = 32, crdWidth = 32 }>

func.func @pooling_nhwc_sum_CCCC(%input: tensor<1x4x4x1xf32, #CCCC>, %filter: tensor<2x2xf32>) -> tensor<1x3x3x1xf32, #CCCC> {
  %init = tensor.empty() : tensor<1x3x3x1xf32, #CCCC>
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


func.func @main() {
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
  // CHECK:      ---- Sparse Tensor ----
  // CHECK-NEXT: nse = 9
  // CHECK-NEXT: dim = ( 1, 3, 3, 1 )
  // CHECK-NEXT: lvl = ( 1, 3, 3, 1 )
  // CHECK-NEXT: pos[0] : ( 0, 1 )
  // CHECK-NEXT: crd[0] : ( 0 )
  // CHECK-NEXT: pos[1] : ( 0, 3 )
  // CHECK-NEXT: crd[1] : ( 0, 1, 2 )
  // CHECK-NEXT: pos[2] : ( 0, 3, 6, 9 )
  // CHECK-NEXT: crd[2] : ( 0, 1, 2, 0, 1, 2, 0, 1, 2 )
  // CHECK-NEXT: pos[3] : ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 )
  // CHECK-NEXT: crd[3] : ( 0, 0, 0, 0, 0, 0, 0, 0, 0 )
  // CHECK-NEXT: values : ( 6, 6, 6, 6, 6, 6, 6, 6, 6 )
  // CHECK-NEXT: ----
  //
  sparse_tensor.print %CCCC_ret : tensor<1x3x3x1xf32, #CCCC>

  // Releases resources.
  bufferization.dealloc_tensor %in_CCCC : tensor<1x4x4x1xf32, #CCCC>
  bufferization.dealloc_tensor %CCCC_ret : tensor<1x3x3x1xf32, #CCCC>
  bufferization.dealloc_tensor %dense_ret : tensor<1x3x3x1xf32>
  return
}
