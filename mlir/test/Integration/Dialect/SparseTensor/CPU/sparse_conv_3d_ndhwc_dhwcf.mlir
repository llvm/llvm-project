// RUN: mlir-opt %s --sparse-compiler=enable-runtime-library=true | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

// RUN: mlir-opt %s --sparse-compiler="enable-runtime-library=false enable-buffer-initialization=true" | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#CCCCC = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed", "compressed", "compressed", "compressed" ]
}>

#CDCDC = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "dense", "compressed", "dense", "compressed"]
}>

// Creates and returns 5-D buffer of size (%s1, %s2, %s3, %s4, %s5) filled with the value %f
func.func @alloc_5d_filled_f32(%s1 : index, %s2 : index, %s3 : index, %s4 : index, %s5 : index, %f : f32) -> tensor<?x?x?x?x?xf32> {
  %buf = bufferization.alloc_tensor(%s1, %s2, %s3, %s4, %s5) : tensor<?x?x?x?x?xf32>
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
                                     %arg1: tensor<?x?x?x?x?xf32, #CCCCC>)
                                     -> tensor<?x?x?x?x?xf32, #CCCCC> {
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %s = bufferization.alloc_tensor(%c1, %c6, %c6, %c6, %c1)
    : tensor<?x?x?x?x?xf32, #CCCCC>
  %ret = linalg.conv_3d_ndhwc_dhwcf {dilations = dense<1> : tensor<3xi64>,
                                       strides = dense<1> : tensor<3xi64>}
     ins (%arg0, %arg1: tensor<?x?x?x?x?xf32, #CCCCC>, tensor<?x?x?x?x?xf32, #CCCCC>)
    outs (%s: tensor<?x?x?x?x?xf32, #CCCCC>) -> tensor<?x?x?x?x?xf32, #CCCCC>
  return %ret : tensor<?x?x?x?x?xf32, #CCCCC>
}

func.func @conv_3d_ndhwc_dhwcf_CDCDC(%arg0: tensor<?x?x?x?x?xf32, #CDCDC>,
                                     %arg1: tensor<?x?x?x?x?xf32, #CDCDC>)
                                     -> tensor<?x?x?x?x?xf32, #CDCDC> {
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %s = bufferization.alloc_tensor(%c1, %c6, %c6, %c6, %c1)
    : tensor<?x?x?x?x?xf32, #CDCDC>
  %ret = linalg.conv_3d_ndhwc_dhwcf {dilations = dense<1> : tensor<3xi64>,
                                       strides = dense<1> : tensor<3xi64>}
     ins (%arg0, %arg1: tensor<?x?x?x?x?xf32, #CDCDC>, tensor<?x?x?x?x?xf32, #CDCDC>)
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
  %filter3D_ndhwc_CCCCC = sparse_tensor.convert %filter3D_ndhwc
    : tensor<?x?x?x?x?xf32> to tensor<?x?x?x?x?xf32, #CCCCC>

  %in3D_ndhwc_CDCDC = sparse_tensor.convert %in3D_ndhwc
    : tensor<?x?x?x?x?xf32> to tensor<?x?x?x?x?xf32, #CDCDC>
  %filter3D_ndhwc_CDCDC = sparse_tensor.convert %filter3D_ndhwc
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

  %CCCCC_ret = call @conv_3d_ndhwc_dhwcf_CCCCC(%in3D_ndhwc_CCCCC, %filter3D_ndhwc_CCCCC)
      : (tensor<?x?x?x?x?xf32, #CCCCC>,
         tensor<?x?x?x?x?xf32, #CCCCC>) -> (tensor<?x?x?x?x?xf32, #CCCCC>)

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

  %CDCDC_ret = call @conv_3d_ndhwc_dhwcf_CDCDC(%in3D_ndhwc_CDCDC, %filter3D_ndhwc_CDCDC)
      : (tensor<?x?x?x?x?xf32, #CDCDC>,
         tensor<?x?x?x?x?xf32, #CDCDC>) -> (tensor<?x?x?x?x?xf32, #CDCDC>)

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
  bufferization.dealloc_tensor %filter3D_ndhwc_CDCDC : tensor<?x?x?x?x?xf32, #CDCDC>
  bufferization.dealloc_tensor %in3D_ndhwc_CCCCC : tensor<?x?x?x?x?xf32, #CCCCC>
  bufferization.dealloc_tensor %filter3D_ndhwc_CCCCC : tensor<?x?x?x?x?xf32, #CCCCC>

  bufferization.dealloc_tensor %CCCCC_ret : tensor<?x?x?x?x?xf32, #CCCCC>
  bufferization.dealloc_tensor %CDCDC_ret : tensor<?x?x?x?x?xf32, #CDCDC>

  return
}
