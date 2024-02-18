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
// DEFINE: %{run_opts} = -e entry -entry-point-result=void
// DEFINE: %{run} = mlir-cpu-runner %{run_opts} %{run_libs}
// DEFINE: %{run_sve} = %mcr_aarch64_cmd --march=aarch64 --mattr="+sve" %{run_opts} %{run_libs}
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


// TODO: we can only support dense output for nchw input because 'c' is a reduction loop


#CDCD = #sparse_tensor.encoding<{
  map = (d0, d1, d2, d3) -> (d0 : compressed, d1 : dense, d2 : compressed, d3 : dense)
}>


#CCCC = #sparse_tensor.encoding<{
  map = (d0, d1, d2, d3) -> (d0 : compressed, d1 : compressed, d2 : compressed, d3 : compressed)
}>

// Creates and returns 4-D buffer of size (%s1, %s2, %s3, %s4) filled with the value %f
func.func @alloc_4d_filled_f32(%s1 : index, %s2 : index, %s3 : index, %s4 : index, %f : f32) -> tensor<?x?x?x?xf32> {
  %buf = tensor.empty(%s1, %s2, %s3, %s4) : tensor<?x?x?x?xf32>
  %ret = linalg.fill ins(%f : f32) outs(%buf : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %ret : tensor<?x?x?x?xf32>
}

func.func @conv_2d_nchw_fchw(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>, %arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %ret = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>,
                                     strides = dense<1> : tensor<2xi64>}
     ins (%arg0, %arg1: tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
    outs (%arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %ret : tensor<?x?x?x?xf32>
}

func.func @conv_2d_nchw_fchw_CDCD(%arg0: tensor<?x?x?x?xf32, #CDCD>, %arg1: tensor<?x?x?x?xf32>, %arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %ret = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>,
                                     strides = dense<1> : tensor<2xi64>}
     ins (%arg0, %arg1: tensor<?x?x?x?xf32, #CDCD>, tensor<?x?x?x?xf32>)
    outs (%arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %ret : tensor<?x?x?x?xf32>
}

func.func @conv_2d_nchw_fchw_CCCC(%arg0: tensor<?x?x?x?xf32, #CCCC>, %arg1: tensor<?x?x?x?xf32>, %arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %ret = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>,
                                     strides = dense<1> : tensor<2xi64>}
     ins (%arg0, %arg1: tensor<?x?x?x?xf32, #CCCC>, tensor<?x?x?x?xf32>)
    outs (%arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %ret : tensor<?x?x?x?xf32>
}

func.func @conv_2d_nchw_fchw_CCCC_CCCC(%arg0: tensor<?x?x?x?xf32, #CCCC>, %arg1: tensor<?x?x?x?xf32, #CCCC>, %arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %ret = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>,
                                     strides = dense<1> : tensor<2xi64>}
     ins (%arg0, %arg1: tensor<?x?x?x?xf32, #CCCC>, tensor<?x?x?x?xf32, #CCCC>)
    outs (%arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %ret : tensor<?x?x?x?xf32>
}

func.func @entry() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c6 = arith.constant 6 : index
  %c8 = arith.constant 8 : index
  %f10 = arith.constant 10.00000e+00 : f32
  %val = arith.constant 2.00000e+00 : f32
  %zero = arith.constant 0.00000e+00 : f32

  %filter2D_nhwc = call @alloc_4d_filled_f32(%c1, %c3, %c3, %c3, %val) :(index, index, index, index, f32) -> (tensor<?x?x?x?xf32>)
  %in2D_tmp = call @alloc_4d_filled_f32(%c3, %c3, %c8, %c8, %val) : (index, index, index, index, f32) -> (tensor<?x?x?x?xf32>)
  %in2D_nhwc = tensor.insert %f10 into %in2D_tmp[%c0, %c0, %c0, %c3] : tensor<?x?x?x?xf32>
  %out2D_nhwc = call @alloc_4d_filled_f32(%c3, %c1, %c6, %c6, %zero) : (index, index, index, index, f32) -> (tensor<?x?x?x?xf32>)
  %out2D_nhwc_CCCD = call @alloc_4d_filled_f32(%c3, %c1, %c6, %c6, %zero) : (index, index, index, index, f32) -> (tensor<?x?x?x?xf32>)
  %out2D_nhwc_CCCC = call @alloc_4d_filled_f32(%c3, %c1, %c6, %c6, %zero) : (index, index, index, index, f32) -> (tensor<?x?x?x?xf32>)

  %in2D_nhwc_CCCD = sparse_tensor.convert %in2D_nhwc
    : tensor<?x?x?x?xf32> to tensor<?x?x?x?xf32, #CDCD>
  %in2D_nhwc_CCCC = sparse_tensor.convert %in2D_nhwc
    : tensor<?x?x?x?xf32> to tensor<?x?x?x?xf32, #CCCC>

  %filter2D_nhwc_CCCC = sparse_tensor.convert %filter2D_nhwc
    : tensor<?x?x?x?xf32> to tensor<?x?x?x?xf32, #CCCC>

  %dense_ret = call @conv_2d_nchw_fchw(%in2D_nhwc, %filter2D_nhwc, %out2D_nhwc) : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> (tensor<?x?x?x?xf32>)
  %CCCC_ret = call @conv_2d_nchw_fchw_CDCD(%in2D_nhwc_CCCD, %filter2D_nhwc, %out2D_nhwc_CCCD) : (tensor<?x?x?x?xf32, #CDCD>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> (tensor<?x?x?x?xf32>)
  %CDCD_ret = call @conv_2d_nchw_fchw_CCCC(%in2D_nhwc_CCCC, %filter2D_nhwc, %out2D_nhwc_CCCC) : (tensor<?x?x?x?xf32, #CCCC>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> (tensor<?x?x?x?xf32>)
  %dual_CCCC_ret = call @conv_2d_nchw_fchw_CCCC_CCCC(%in2D_nhwc_CCCC, %filter2D_nhwc_CCCC, %out2D_nhwc) : (tensor<?x?x?x?xf32, #CCCC>, tensor<?x?x?x?xf32, #CCCC>, tensor<?x?x?x?xf32>) -> (tensor<?x?x?x?xf32>)


  // CHECK:     ( ( ( ( 108, 124, 124, 124, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ) ) ),
  // CHECK-SAME:  ( ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ) ) ),
  // CHECK-SAME:  ( ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ) ) ) )
  %dense_v = vector.transfer_read %dense_ret[%c0, %c0, %c0, %c0], %zero
      : tensor<?x?x?x?xf32>, vector<3x1x6x6xf32>
  vector.print %dense_v : vector<3x1x6x6xf32>

  // CHECK:     ( ( ( ( 108, 124, 124, 124, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ) ) ),
  // CHECK-SAME:  ( ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ) ) ),
  // CHECK-SAME:  ( ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ) ) ) )
  %v1 = vector.transfer_read %CCCC_ret[%c0, %c0, %c0, %c0], %zero
      : tensor<?x?x?x?xf32>, vector<3x1x6x6xf32>
  vector.print %v1 : vector<3x1x6x6xf32>

  // CHECK:     ( ( ( ( 108, 124, 124, 124, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ) ) ),
  // CHECK-SAME:  ( ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ) ) ),
  // CHECK-SAME:  ( ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ) ) ) )
  %v2 = vector.transfer_read %CDCD_ret[%c0, %c0, %c0, %c0], %zero
      : tensor<?x?x?x?xf32>, vector<3x1x6x6xf32>
  vector.print %v2 : vector<3x1x6x6xf32>

  // CHECK:     ( ( ( ( 108, 124, 124, 124, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ) ) ),
  // CHECK-SAME:  ( ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ) ) ),
  // CHECK-SAME:  ( ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:      ( 108, 108, 108, 108, 108, 108 ) ) ) )
  %v3 = vector.transfer_read %dual_CCCC_ret[%c0, %c0, %c0, %c0], %zero
      : tensor<?x?x?x?xf32>, vector<3x1x6x6xf32>
  vector.print %v3 : vector<3x1x6x6xf32>

  // Free the resources
  bufferization.dealloc_tensor %in2D_nhwc : tensor<?x?x?x?xf32>
  bufferization.dealloc_tensor %filter2D_nhwc : tensor<?x?x?x?xf32>
  bufferization.dealloc_tensor %out2D_nhwc : tensor<?x?x?x?xf32>
  bufferization.dealloc_tensor %out2D_nhwc_CCCD : tensor<?x?x?x?xf32>
  bufferization.dealloc_tensor %out2D_nhwc_CCCC : tensor<?x?x?x?xf32>
  bufferization.dealloc_tensor %dense_ret :tensor<?x?x?x?xf32>

  bufferization.dealloc_tensor %in2D_nhwc_CCCC : tensor<?x?x?x?xf32, #CCCC>
  bufferization.dealloc_tensor %in2D_nhwc_CCCD : tensor<?x?x?x?xf32, #CDCD>
  bufferization.dealloc_tensor %filter2D_nhwc_CCCC : tensor<?x?x?x?xf32, #CCCC>
  return
}
