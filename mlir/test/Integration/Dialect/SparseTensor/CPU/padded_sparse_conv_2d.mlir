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
// DEFINE: %{run} = mlir-runner %{run_opts} %{run_libs}
// DEFINE: %{run_sve} = %mcr_aarch64_cmd --march=aarch64 --mattr="+sve" %{run_opts} %{run_libs_sve}
//
// DEFINE: %{env} =
//--------------------------------------------------------------------------------------------------

// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false enable-buffer-initialization=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false enable-buffer-initialization=true vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#CDCC_NHWC = #sparse_tensor.encoding<{
  map = (d0, d1, d2, d3) -> (d0 : compressed, d1 : dense, d2 : compressed, d3 : compressed)
}>

// Creates and returns 4-D buffer of size (%s1, %s2, %s3, %s4) filled with the value %f
func.func @alloc_4d_filled_f32(%s1 : index, %s2 : index, %s3 : index, %s4 : index, %f : f32) -> tensor<?x?x?x?xf32> {
  %buf = tensor.empty(%s1, %s2, %s3, %s4) : tensor<?x?x?x?xf32>
  %ret = linalg.fill ins(%f : f32) outs(%buf : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %ret : tensor<?x?x?x?xf32>
}

func.func @conv_2d_nhwc_hwcf(%arg0: tensor<3x8x8x3xf32>, %arg1: tensor<5x5x3x1xf32>, %arg2: tensor<3x8x8x1xf32>) -> tensor<3x8x8x1xf32> {
  %cst_0 = arith.constant 0.00000e+00 : f32

  %padded = tensor.pad %arg0 low[0, 2, 2, 0] high[0, 2, 2, 0] {
   ^bb0(%arg75: index, %arg76: index, %arg77: index, %arg78: index):
     tensor.yield %cst_0 : f32
   } : tensor<3x8x8x3xf32> to tensor<3x12x12x3xf32>

  %ret = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                                     strides = dense<1> : tensor<2xi64>}
     ins (%padded, %arg1: tensor<3x12x12x3xf32>, tensor<5x5x3x1xf32>)
    outs (%arg2: tensor<3x8x8x1xf32>) -> tensor<3x8x8x1xf32>

  bufferization.dealloc_tensor %padded : tensor<3x12x12x3xf32>
  return %ret : tensor<3x8x8x1xf32>
}

func.func @conv_2d_nhwc_hwcf_CDCC_NHWC(%arg0: tensor<3x8x8x3xf32, #CDCC_NHWC>, %arg1: tensor<5x5x3x1xf32>) -> tensor<3x8x8x1xf32> {
  %cst_0 = arith.constant 0.00000e+00 : f32
  %buf = tensor.empty() : tensor<3x8x8x1xf32>
  %s = linalg.fill ins(%cst_0 : f32) outs(%buf : tensor<3x8x8x1xf32>) -> tensor<3x8x8x1xf32>

  %padded = tensor.pad %arg0 low[0, 2, 2, 0] high[0, 2, 2, 0] {
   ^bb0(%arg75: index, %arg76: index, %arg77: index, %arg78: index):
     tensor.yield %cst_0 : f32
   } : tensor<3x8x8x3xf32, #CDCC_NHWC> to tensor<3x12x12x3xf32, #CDCC_NHWC>

  %ret = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                                     strides = dense<1> : tensor<2xi64>}
     ins (%padded, %arg1: tensor<3x12x12x3xf32, #CDCC_NHWC>, tensor<5x5x3x1xf32>)
    outs (%s: tensor<3x8x8x1xf32>) -> tensor<3x8x8x1xf32>
  return %ret : tensor<3x8x8x1xf32>
}

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c8 = arith.constant 8 : index
  %f10 = arith.constant 10.00000e+00 : f32
  %val = arith.constant 2.00000e+00 : f32
  %zero = arith.constant 0.00000e+00 : f32

  %filter2D_nhwc = call @alloc_4d_filled_f32(%c5, %c5, %c3, %c1, %val) :(index, index, index, index, f32) -> (tensor<?x?x?x?xf32>)
  %in2D_tmp = call @alloc_4d_filled_f32(%c3, %c8, %c8, %c3, %val) : (index, index, index, index, f32) -> (tensor<?x?x?x?xf32>)
  %in2D_nhwc = tensor.insert %f10 into %in2D_tmp[%c0, %c0, %c3, %c0] : tensor<?x?x?x?xf32>
  %out2D_nhwc = call @alloc_4d_filled_f32(%c3, %c8, %c8, %c1, %zero) : (index, index, index, index, f32) -> (tensor<?x?x?x?xf32>)

  %static_filter = tensor.cast %filter2D_nhwc : tensor<?x?x?x?xf32> to tensor<5x5x3x1xf32>
  %static_input = tensor.cast %in2D_nhwc : tensor<?x?x?x?xf32> to tensor<3x8x8x3xf32>
  %static_output = tensor.cast %out2D_nhwc : tensor<?x?x?x?xf32> to tensor<3x8x8x1xf32>

  %dense_ret = call @conv_2d_nhwc_hwcf(%static_input, %static_filter, %static_output) : (tensor<3x8x8x3xf32>, tensor<5x5x3x1xf32>, tensor<3x8x8x1xf32>) -> (tensor<3x8x8x1xf32>)

  %in2D_nhwc_CDCC_NHWC = sparse_tensor.convert %static_input : tensor<3x8x8x3xf32> to tensor<3x8x8x3xf32, #CDCC_NHWC>
  %CDCC_NHWC_ret = call @conv_2d_nhwc_hwcf_CDCC_NHWC(%in2D_nhwc_CDCC_NHWC, %static_filter) : (tensor<3x8x8x3xf32, #CDCC_NHWC>, tensor<5x5x3x1xf32>) -> (tensor<3x8x8x1xf32>)


  // CHECK:   ( ( ( ( 108 ), ( 160 ), ( 196 ), ( 196 ), ( 196 ), ( 196 ), ( 144 ), ( 108 ) ),
  // CHECK-SAME:  ( ( 144 ), ( 208 ), ( 256 ), ( 256 ), ( 256 ), ( 256 ), ( 192 ), ( 144 ) ),
  // CHECK-SAME:  ( ( 180 ), ( 256 ), ( 316 ), ( 316 ), ( 316 ), ( 316 ), ( 240 ), ( 180 ) ),
  // CHECK-SAME:  ( ( 180 ), ( 240 ), ( 300 ), ( 300 ), ( 300 ), ( 300 ), ( 240 ), ( 180 ) ),
  // CHECK-SAME:  ( ( 180 ), ( 240 ), ( 300 ), ( 300 ), ( 300 ), ( 300 ), ( 240 ), ( 180 ) ),
  // CHECK-SAME:  ( ( 180 ), ( 240 ), ( 300 ), ( 300 ), ( 300 ), ( 300 ), ( 240 ), ( 180 ) ),
  // CHECK-SAME:  ( ( 144 ), ( 192 ), ( 240 ), ( 240 ), ( 240 ), ( 240 ), ( 192 ), ( 144 ) ),
  // CHECK-SAME:  ( ( 108 ), ( 144 ), ( 180 ), ( 180 ), ( 180 ), ( 180 ), ( 144 ), ( 108 ) ) ),
  // CHECK-SAME:( ( ( 108 ), ( 144 ), ( 180 ), ( 180 ), ( 180 ), ( 180 ), ( 144 ), ( 108 ) ),
  // CHECK-SAME:  ( ( 144 ), ( 192 ), ( 240 ), ( 240 ), ( 240 ), ( 240 ), ( 192 ), ( 144 ) ),
  // CHECK-SAME:  ( ( 180 ), ( 240 ), ( 300 ), ( 300 ), ( 300 ), ( 300 ), ( 240 ), ( 180 ) ),
  // CHECK-SAME:  ( ( 180 ), ( 240 ), ( 300 ), ( 300 ), ( 300 ), ( 300 ), ( 240 ), ( 180 ) ),
  // CHECK-SAME:  ( ( 180 ), ( 240 ), ( 300 ), ( 300 ), ( 300 ), ( 300 ), ( 240 ), ( 180 ) ),
  // CHECK-SAME:  ( ( 180 ), ( 240 ), ( 300 ), ( 300 ), ( 300 ), ( 300 ), ( 240 ), ( 180 ) ),
  // CHECK-SAME:  ( ( 144 ), ( 192 ), ( 240 ), ( 240 ), ( 240 ), ( 240 ), ( 192 ), ( 144 ) ),
  // CHECK-SAME:  ( ( 108 ), ( 144 ), ( 180 ), ( 180 ), ( 180 ), ( 180 ), ( 144 ), ( 108 ) ) ),
  // CHECK-SAME:( ( ( 108 ), ( 144 ), ( 180 ), ( 180 ), ( 180 ), ( 180 ), ( 144 ), ( 108 ) ),
  // CHECK-SAME:  ( ( 144 ), ( 192 ), ( 240 ), ( 240 ), ( 240 ), ( 240 ), ( 192 ), ( 144 ) ),
  // CHECK-SAME:  ( ( 180 ), ( 240 ), ( 300 ), ( 300 ), ( 300 ), ( 300 ), ( 240 ), ( 180 ) ),
  // CHECK-SAME:  ( ( 180 ), ( 240 ), ( 300 ), ( 300 ), ( 300 ), ( 300 ), ( 240 ), ( 180 ) ),
  // CHECK-SAME:  ( ( 180 ), ( 240 ), ( 300 ), ( 300 ), ( 300 ), ( 300 ), ( 240 ), ( 180 ) ),
  // CHECK-SAME:  ( ( 180 ), ( 240 ), ( 300 ), ( 300 ), ( 300 ), ( 300 ), ( 240 ), ( 180 ) ),
  // CHECK-SAME:  ( ( 144 ), ( 192 ), ( 240 ), ( 240 ), ( 240 ), ( 240 ), ( 192 ), ( 144 ) ),
  // CHECK-SAME:  ( ( 108 ), ( 144 ), ( 180 ), ( 180 ), ( 180 ), ( 180 ), ( 144 ), ( 108 ) ) ) )
  %dense_v = vector.transfer_read %dense_ret[%c0, %c0, %c0, %c0], %zero
      : tensor<3x8x8x1xf32>, vector<3x8x8x1xf32>
  vector.print %dense_v : vector<3x8x8x1xf32>

  // CHECK-NEXT: ( ( ( ( 108 ), ( 160 ), ( 196 ), ( 196 ), ( 196 ), ( 196 ), ( 144 ), ( 108 ) ),
  // CHECK-SAME:     ( ( 144 ), ( 208 ), ( 256 ), ( 256 ), ( 256 ), ( 256 ), ( 192 ), ( 144 ) ),
  // CHECK-SAME:     ( ( 180 ), ( 256 ), ( 316 ), ( 316 ), ( 316 ), ( 316 ), ( 240 ), ( 180 ) ),
  // CHECK-SAME:     ( ( 180 ), ( 240 ), ( 300 ), ( 300 ), ( 300 ), ( 300 ), ( 240 ), ( 180 ) ),
  // CHECK-SAME:     ( ( 180 ), ( 240 ), ( 300 ), ( 300 ), ( 300 ), ( 300 ), ( 240 ), ( 180 ) ),
  // CHECK-SAME:     ( ( 180 ), ( 240 ), ( 300 ), ( 300 ), ( 300 ), ( 300 ), ( 240 ), ( 180 ) ),
  // CHECK-SAME:     ( ( 144 ), ( 192 ), ( 240 ), ( 240 ), ( 240 ), ( 240 ), ( 192 ), ( 144 ) ),
  // CHECK-SAME:     ( ( 108 ), ( 144 ), ( 180 ), ( 180 ), ( 180 ), ( 180 ), ( 144 ), ( 108 ) ) ),
  // CHECK-SAME:   ( ( ( 108 ), ( 144 ), ( 180 ), ( 180 ), ( 180 ), ( 180 ), ( 144 ), ( 108 ) ),
  // CHECK-SAME:     ( ( 144 ), ( 192 ), ( 240 ), ( 240 ), ( 240 ), ( 240 ), ( 192 ), ( 144 ) ),
  // CHECK-SAME:     ( ( 180 ), ( 240 ), ( 300 ), ( 300 ), ( 300 ), ( 300 ), ( 240 ), ( 180 ) ),
  // CHECK-SAME:     ( ( 180 ), ( 240 ), ( 300 ), ( 300 ), ( 300 ), ( 300 ), ( 240 ), ( 180 ) ),
  // CHECK-SAME:     ( ( 180 ), ( 240 ), ( 300 ), ( 300 ), ( 300 ), ( 300 ), ( 240 ), ( 180 ) ),
  // CHECK-SAME:     ( ( 180 ), ( 240 ), ( 300 ), ( 300 ), ( 300 ), ( 300 ), ( 240 ), ( 180 ) ),
  // CHECK-SAME:     ( ( 144 ), ( 192 ), ( 240 ), ( 240 ), ( 240 ), ( 240 ), ( 192 ), ( 144 ) ),
  // CHECK-SAME:     ( ( 108 ), ( 144 ), ( 180 ), ( 180 ), ( 180 ), ( 180 ), ( 144 ), ( 108 ) ) ),
  // CHECK-SAME:   ( ( ( 108 ), ( 144 ), ( 180 ), ( 180 ), ( 180 ), ( 180 ), ( 144 ), ( 108 ) ),
  // CHECK-SAME:     ( ( 144 ), ( 192 ), ( 240 ), ( 240 ), ( 240 ), ( 240 ), ( 192 ), ( 144 ) ),
  // CHECK-SAME:     ( ( 180 ), ( 240 ), ( 300 ), ( 300 ), ( 300 ), ( 300 ), ( 240 ), ( 180 ) ),
  // CHECK-SAME:     ( ( 180 ), ( 240 ), ( 300 ), ( 300 ), ( 300 ), ( 300 ), ( 240 ), ( 180 ) ),
  // CHECK-SAME:     ( ( 180 ), ( 240 ), ( 300 ), ( 300 ), ( 300 ), ( 300 ), ( 240 ), ( 180 ) ),
  // CHECK-SAME:     ( ( 180 ), ( 240 ), ( 300 ), ( 300 ), ( 300 ), ( 300 ), ( 240 ), ( 180 ) ),
  // CHECK-SAME:     ( ( 144 ), ( 192 ), ( 240 ), ( 240 ), ( 240 ), ( 240 ), ( 192 ), ( 144 ) ),
  // CHECK-SAME:     ( ( 108 ), ( 144 ), ( 180 ), ( 180 ), ( 180 ), ( 180 ), ( 144 ), ( 108 ) ) ) )
  %CDCC_NHWC_v = vector.transfer_read %CDCC_NHWC_ret[%c0, %c0, %c0, %c0], %zero
      : tensor<3x8x8x1xf32>, vector<3x8x8x1xf32>
  vector.print %CDCC_NHWC_v : vector<3x8x8x1xf32>

  bufferization.dealloc_tensor %static_filter : tensor<5x5x3x1xf32>
  bufferization.dealloc_tensor %static_input  : tensor<3x8x8x3xf32>
  bufferization.dealloc_tensor %static_output : tensor<3x8x8x1xf32>

  bufferization.dealloc_tensor %CDCC_NHWC_ret : tensor<3x8x8x1xf32>

  bufferization.dealloc_tensor %in2D_nhwc_CDCC_NHWC : tensor<3x8x8x3xf32, #CDCC_NHWC>

  return
}
