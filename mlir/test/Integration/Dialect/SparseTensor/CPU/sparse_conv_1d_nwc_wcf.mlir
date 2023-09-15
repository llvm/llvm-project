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

#CCC = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed", "compressed", "compressed" ] }>

#CDC = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed", "dense", "compressed" ]
  // FIXME: Still inadmissible might need investigation
  // dimToLvl = affine_map<(i,j,k) -> (j,k,i)>
}>

// Creates and returns 3-D buffer of size (%s1, %s2, %s3) filled with the value %f
func.func @alloc_3d_filled_f32(%s1 : index, %s2 : index, %s3 : index, %f : f32) -> tensor<?x?x?xf32> {
  %buf = tensor.empty(%s1, %s2, %s3) : tensor<?x?x?xf32>
  %ret = linalg.fill ins(%f : f32) outs(%buf : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %ret : tensor<?x?x?xf32>
}

func.func @conv_1d_nwc_wcf(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %ret = linalg.conv_1d_nwc_wcf {dilations = dense<1> : tensor<1xi64>,
                                   strides = dense<1> : tensor<1xi64>}
     ins (%arg0, %arg1: tensor<?x?x?xf32>, tensor<?x?x?xf32>)
    outs (%arg2: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %ret : tensor<?x?x?xf32>
}

func.func @conv_1d_nwc_wcf_CCC(%arg0: tensor<?x?x?xf32, #CCC>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32, #CCC> {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c6 = arith.constant 6 : index
  %s = tensor.empty(%c3, %c6, %c1) : tensor<?x?x?xf32, #CCC>
  %ret = linalg.conv_1d_nwc_wcf {dilations = dense<1> : tensor<1xi64>,
                                   strides = dense<1> : tensor<1xi64>}
     ins (%arg0, %arg1: tensor<?x?x?xf32, #CCC>, tensor<?x?x?xf32>)
    outs (%s: tensor<?x?x?xf32, #CCC>) -> tensor<?x?x?xf32, #CCC>
  return %ret : tensor<?x?x?xf32, #CCC>
}

func.func @conv_1d_nwc_wcf_CDC(%arg0: tensor<?x?x?xf32, #CDC>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32, #CDC> {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c6 = arith.constant 6 : index
  %s = tensor.empty(%c3, %c6, %c1) : tensor<?x?x?xf32, #CDC>
  %ret = linalg.conv_1d_nwc_wcf {dilations = dense<1> : tensor<1xi64>,
                                   strides = dense<1> : tensor<1xi64>}
     ins (%arg0, %arg1: tensor<?x?x?xf32, #CDC>, tensor<?x?x?xf32>)
    outs (%s: tensor<?x?x?xf32, #CDC>) -> tensor<?x?x?xf32, #CDC>
  return %ret : tensor<?x?x?xf32, #CDC>
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

  %in1D_tmp = call @alloc_3d_filled_f32(%c3, %c8, %c1, %val) : (index, index, index, f32) -> (tensor<?x?x?xf32>)
  %in1D_nwc = tensor.insert %f10 into %in1D_tmp[%c0, %c3, %c0] : tensor<?x?x?xf32>

  %filter1D_nwc = call @alloc_3d_filled_f32(%c3, %c1, %c1, %val) : (index, index, index, f32) -> (tensor<?x?x?xf32>)
  %out1D_nwc = call @alloc_3d_filled_f32(%c3, %c6, %c1, %zero) : (index, index, index, f32) -> (tensor<?x?x?xf32>)

  %in1D_nwc_CCC = sparse_tensor.convert %in1D_nwc
    : tensor<?x?x?xf32> to tensor<?x?x?xf32, #CCC>
  %in1D_nwc_CDC = sparse_tensor.convert %in1D_nwc
    : tensor<?x?x?xf32> to tensor<?x?x?xf32, #CDC>

  %dense_ret = call @conv_1d_nwc_wcf(%in1D_nwc, %filter1D_nwc, %out1D_nwc) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>)
  %CCC_ret = call @conv_1d_nwc_wcf_CCC(%in1D_nwc_CCC, %filter1D_nwc) : (tensor<?x?x?xf32, #CCC>, tensor<?x?x?xf32>) -> (tensor<?x?x?xf32, #CCC>)
  %CDC_ret = call @conv_1d_nwc_wcf_CDC(%in1D_nwc_CDC, %filter1D_nwc) : (tensor<?x?x?xf32, #CDC>, tensor<?x?x?xf32>) -> (tensor<?x?x?xf32, #CDC>)

  //      CHECK: ( ( ( 12 ), ( 28 ), ( 28 ), ( 28 ), ( 12 ), ( 12 ) ),
  // CHECK-SAME:   ( ( 12 ), ( 12 ), ( 12 ), ( 12 ), ( 12 ), ( 12 ) ),
  // CHECK-SAME:   ( ( 12 ), ( 12 ), ( 12 ), ( 12 ), ( 12 ), ( 12 ) ) )
  %dense_v = vector.transfer_read %dense_ret[%c0, %c0, %c0], %zero
      : tensor<?x?x?xf32>, vector<3x6x1xf32>
  vector.print %dense_v : vector<3x6x1xf32>

  //      CHECK: ( ( ( 12 ), ( 28 ), ( 28 ), ( 28 ), ( 12 ), ( 12 ) ),
  // CHECK-SAME:   ( ( 12 ), ( 12 ), ( 12 ), ( 12 ), ( 12 ), ( 12 ) ),
  // CHECK-SAME:   ( ( 12 ), ( 12 ), ( 12 ), ( 12 ), ( 12 ), ( 12 ) ) )
  %1 = sparse_tensor.convert %CCC_ret
    : tensor<?x?x?xf32, #CCC> to tensor<?x?x?xf32>
  %v1 = vector.transfer_read %1[%c0, %c0, %c0], %zero
      : tensor<?x?x?xf32>, vector<3x6x1xf32>
  vector.print %v1 : vector<3x6x1xf32>

  //      CHECK: ( ( ( 12 ), ( 28 ), ( 28 ), ( 28 ), ( 12 ), ( 12 ) ),
  // CHECK-SAME:   ( ( 12 ), ( 12 ), ( 12 ), ( 12 ), ( 12 ), ( 12 ) ),
  // CHECK-SAME:   ( ( 12 ), ( 12 ), ( 12 ), ( 12 ), ( 12 ), ( 12 ) ) )
  %2 = sparse_tensor.convert %CDC_ret
    : tensor<?x?x?xf32, #CDC> to tensor<?x?x?xf32>
  %v2 = vector.transfer_read %2[%c0, %c0, %c0], %zero
      : tensor<?x?x?xf32>, vector<3x6x1xf32>
  vector.print %v2 : vector<3x6x1xf32>

  // Free the resources
  bufferization.dealloc_tensor %in1D_nwc : tensor<?x?x?xf32>
  bufferization.dealloc_tensor %filter1D_nwc : tensor<?x?x?xf32>
  bufferization.dealloc_tensor %out1D_nwc : tensor<?x?x?xf32>

  bufferization.dealloc_tensor %in1D_nwc_CDC : tensor<?x?x?xf32, #CDC>
  bufferization.dealloc_tensor %in1D_nwc_CCC : tensor<?x?x?xf32, #CCC>

  bufferization.dealloc_tensor %CCC_ret : tensor<?x?x?xf32, #CCC>
  bufferization.dealloc_tensor %CDC_ret : tensor<?x?x?xf32, #CDC>

  return
}
