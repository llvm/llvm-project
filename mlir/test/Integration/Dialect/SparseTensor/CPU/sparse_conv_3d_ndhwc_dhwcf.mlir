//--------------------------------------------------------------------------------------------------
// WHEN CREATING A NEW TEST, PLEASE JUST COPY & PASTE WITHOUT EDITS.
//
// Set-up that's shared across all tests in this directory. In principle, this
// config could be moved to lit.local.cfg. However, there are downstream users that
//  do not use these LIT config files. Hence why this is kept inline.
//
// DEFINE: %{sparse_compiler_opts} = enable-runtime-library=true
// DEFINE: %{sparse_compiler_opts_sve} = enable-arm-sve=true %{sparse_compiler_opts}
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler="%{sparse_compiler_opts}"
// DEFINE: %{compile_sve} = mlir-opt %s --sparse-compiler="%{sparse_compiler_opts_sve}"
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
// REDEFINE: %{sparse_compiler_opts} = enable-runtime-library=false enable-buffer-initialization=true enable-index-reduction=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparse_compiler_opts} = enable-runtime-library=false enable-buffer-initialization=true vl=2 reassociate-fp-reductions=true enable-index-optimizations=true enable-index-reduction=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#CCCCC = #sparse_tensor.encoding<{
  map = (d0, d1, d2, d3, d4) -> (d0 : compressed, d1 : compressed, d2 : compressed, d3 : compressed, d4 : compressed)
}>

#CDCDC = #sparse_tensor.encoding<{
  map = (d0, d1, d2, d3, d4) -> (d0 : compressed, d1 : dense, d2 : compressed, d3 : dense, d4 : compressed)
}>

// Creates and returns 5-D buffer of size (%s1, %s2, %s3, %s4, %s5) filled with the value %f
func.func @alloc_5d_filled_f32(%s1 : index, %s2 : index, %s3 : index, %s4 : index, %s5 : index, %f : f32) -> tensor<?x?x?x?x?xf32> {
  %buf = tensor.empty(%s1, %s2, %s3, %s4, %s5) : tensor<?x?x?x?x?xf32>
  %ret = linalg.fill ins(%f : f32) outs(%buf : tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %ret : tensor<?x?x?x?x?xf32>
}

func.func @conv_3d_ndhwc_dhwcf(%arg0: tensor<?x?x?x?x?xf32>,
                               %arg1: tensor<?x?x?x?x?xf32>,
                               %arg2: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
  %ret = linalg.conv_3d_ndhwc_dhwcf {dilations = dense<1> : tensor<3xi64>,
                                       strides = dense<1> : tensor<3xi64>}
     ins (%arg0, %arg1: tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>)
    outs (%arg2: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %ret : tensor<?x?x?x?x?xf32>
}

func.func @conv_3d_ndhwc_dhwcf_CCCCC(%arg0: tensor<?x?x?x?x?xf32, #CCCCC>,
                                     %arg1: tensor<?x?x?x?x?xf32>)
                                     -> tensor<?x?x?x?x?xf32, #CCCCC> {
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %s = tensor.empty(%c1, %c6, %c6, %c6, %c1)
    : tensor<?x?x?x?x?xf32, #CCCCC>
  %ret = linalg.conv_3d_ndhwc_dhwcf {dilations = dense<1> : tensor<3xi64>,
                                       strides = dense<1> : tensor<3xi64>}
     ins (%arg0, %arg1: tensor<?x?x?x?x?xf32, #CCCCC>, tensor<?x?x?x?x?xf32>)
    outs (%s: tensor<?x?x?x?x?xf32, #CCCCC>) -> tensor<?x?x?x?x?xf32, #CCCCC>
  return %ret : tensor<?x?x?x?x?xf32, #CCCCC>
}

func.func @conv_3d_ndhwc_dhwcf_CDCDC(%arg0: tensor<?x?x?x?x?xf32, #CDCDC>,
                                     %arg1: tensor<?x?x?x?x?xf32>)
                                     -> tensor<?x?x?x?x?xf32, #CDCDC> {
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %s = tensor.empty(%c1, %c6, %c6, %c6, %c1)
    : tensor<?x?x?x?x?xf32, #CDCDC>
  %ret = linalg.conv_3d_ndhwc_dhwcf {dilations = dense<1> : tensor<3xi64>,
                                       strides = dense<1> : tensor<3xi64>}
     ins (%arg0, %arg1: tensor<?x?x?x?x?xf32, #CDCDC>, tensor<?x?x?x?x?xf32>)
    outs (%s: tensor<?x?x?x?x?xf32, #CDCDC>) -> tensor<?x?x?x?x?xf32, #CDCDC>
  return %ret : tensor<?x?x?x?x?xf32, #CDCDC>
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

  %in3D_tmp = call @alloc_5d_filled_f32(%c1, %c8, %c8, %c8, %c1, %val) : (index, index, index, index, index, f32) -> (tensor<?x?x?x?x?xf32>)
  %in3D_ndhwc = tensor.insert %f10 into %in3D_tmp[%c0, %c0, %c0, %c3, %c0] : tensor<?x?x?x?x?xf32>

  %filter3D_ndhwc = call @alloc_5d_filled_f32(%c3, %c3, %c3, %c1, %c1, %val) : (index, index, index, index, index, f32) -> (tensor<?x?x?x?x?xf32>)
  %out3D_ndhwc = call @alloc_5d_filled_f32(%c1, %c6, %c6, %c6, %c1, %zero) : (index, index, index, index, index, f32) -> (tensor<?x?x?x?x?xf32>)

  %in3D_ndhwc_CCCCC = sparse_tensor.convert %in3D_ndhwc
    : tensor<?x?x?x?x?xf32> to tensor<?x?x?x?x?xf32, #CCCCC>
  %in3D_ndhwc_CDCDC = sparse_tensor.convert %in3D_ndhwc
    : tensor<?x?x?x?x?xf32> to tensor<?x?x?x?x?xf32, #CDCDC>

  //      CHECK:( ( ( ( ( 108 ), ( 124 ), ( 124 ), ( 124 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ) ),
  // CHECK-SAME:    ( ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ) ),
  // CHECK-SAME:    ( ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ) ),
  // CHECK-SAME:    ( ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ) ),
  // CHECK-SAME:    ( ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ) ),
  // CHECK-SAME:    ( ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ) ) ) )
  %dense_ret = call @conv_3d_ndhwc_dhwcf(%in3D_ndhwc, %filter3D_ndhwc, %out3D_ndhwc)
      : (tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>) -> (tensor<?x?x?x?x?xf32>)
  %dense_v = vector.transfer_read %dense_ret[%c0, %c0, %c0, %c0, %c0], %zero
      : tensor<?x?x?x?x?xf32>, vector<1x6x6x6x1xf32>
  vector.print %dense_v : vector<1x6x6x6x1xf32>

  %CCCCC_ret = call @conv_3d_ndhwc_dhwcf_CCCCC(%in3D_ndhwc_CCCCC, %filter3D_ndhwc)
      : (tensor<?x?x?x?x?xf32, #CCCCC>,
         tensor<?x?x?x?x?xf32>) -> (tensor<?x?x?x?x?xf32, #CCCCC>)

  // CHECK-NEXT:( ( ( ( ( 108 ), ( 124 ), ( 124 ), ( 124 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ) ),
  // CHECK-SAME:    ( ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ) ),
  // CHECK-SAME:    ( ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ) ),
  // CHECK-SAME:    ( ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ) ),
  // CHECK-SAME:    ( ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ) ),
  // CHECK-SAME:    ( ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ) ) ) )
  %1 = sparse_tensor.convert %CCCCC_ret
    : tensor<?x?x?x?x?xf32, #CCCCC> to tensor<?x?x?x?x?xf32>
  %v1 = vector.transfer_read %1[%c0, %c0, %c0, %c0, %c0], %zero
      : tensor<?x?x?x?x?xf32>, vector<1x6x6x6x1xf32>
  vector.print %v1 : vector<1x6x6x6x1xf32>

  %CDCDC_ret = call @conv_3d_ndhwc_dhwcf_CDCDC(%in3D_ndhwc_CDCDC, %filter3D_ndhwc)
      : (tensor<?x?x?x?x?xf32, #CDCDC>,
         tensor<?x?x?x?x?xf32>) -> (tensor<?x?x?x?x?xf32, #CDCDC>)

  // CHECK-NEXT:( ( ( ( ( 108 ), ( 124 ), ( 124 ), ( 124 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ) ),
  // CHECK-SAME:    ( ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ) ),
  // CHECK-SAME:    ( ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ) ),
  // CHECK-SAME:    ( ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ) ),
  // CHECK-SAME:    ( ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ) ),
  // CHECK-SAME:    ( ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ) ) ) )
  %2 = sparse_tensor.convert %CDCDC_ret
    : tensor<?x?x?x?x?xf32, #CDCDC> to tensor<?x?x?x?x?xf32>
  %v2 = vector.transfer_read %dense_ret[%c0, %c0, %c0, %c0, %c0], %zero
      : tensor<?x?x?x?x?xf32>, vector<1x6x6x6x1xf32>
  vector.print %v2 : vector<1x6x6x6x1xf32>

  // Free the resources
  bufferization.dealloc_tensor %in3D_ndhwc : tensor<?x?x?x?x?xf32>
  bufferization.dealloc_tensor %filter3D_ndhwc : tensor<?x?x?x?x?xf32>
  bufferization.dealloc_tensor %out3D_ndhwc : tensor<?x?x?x?x?xf32>

  bufferization.dealloc_tensor %in3D_ndhwc_CDCDC : tensor<?x?x?x?x?xf32, #CDCDC>
  bufferization.dealloc_tensor %in3D_ndhwc_CCCCC : tensor<?x?x?x?x?xf32, #CCCCC>

  bufferization.dealloc_tensor %CCCCC_ret : tensor<?x?x?x?x?xf32, #CCCCC>
  bufferization.dealloc_tensor %CDCDC_ret : tensor<?x?x?x?x?xf32, #CDCDC>

  return
}
