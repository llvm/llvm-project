// RUN: mlir-opt %s -split-input-file -verify-diagnostics -tosa-attach-target="level=none profiles=pro_int,pro_fp extensions=int16,int4,bf16,fp8e4m3,fp8e5m2,fft,variable,controlflow,dynamic" -tosa-validate="strict-op-spec-alignment validate-function-signature=false"

func.func @test_matmul_rank3_lhs_broadcast_dynamic_rhs(%arg0: tensor<1x4x7xf32>, %arg1: tensor<?x7x5xf32>) -> tensor<2x4x5xf32> {
  %azp0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %bzp0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  // expected-error@+1 {{'tosa.matmul' op MATMUL ranks other than 3 or batch broadcasting require TOSA specification version 1.1.draft}}
  %0 = tosa.matmul %arg0, %arg1, %azp0, %bzp0 : (tensor<1x4x7xf32>, tensor<?x7x5xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x4x5xf32>
  return %0 : tensor<2x4x5xf32>
}

// -----

func.func @test_matmul_rank3_rhs_broadcast_dynamic_lhs(%arg0: tensor<?x4x7xf32>, %arg1: tensor<1x7x5xf32>) -> tensor<2x4x5xf32> {
  %azp0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %bzp0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  // expected-error@+1 {{'tosa.matmul' op MATMUL ranks other than 3 or batch broadcasting require TOSA specification version 1.1.draft}}
  %0 = tosa.matmul %arg0, %arg1, %azp0, %bzp0 : (tensor<?x4x7xf32>, tensor<1x7x5xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x4x5xf32>
  return %0 : tensor<2x4x5xf32>
}

// -----

// CHECK-LABEL: test_resize_large_image_size
func.func @test_resize_large_image_size(%arg0: tensor<1x16384x16384x8xf32>) -> tensor<1x32767x32767x8xf32> {
  %scale = tosa.const_shape values(dense<[2, 1, 2, 1]> : tensor<4xindex>) : () -> !tosa.shape<4>
  %offset = tosa.const_shape values(dense<0> : tensor<2xindex>) : () -> !tosa.shape<2>
  %border = tosa.const_shape values(dense<0> : tensor<2xindex>) : () -> !tosa.shape<2>
  // expected-error@+1 {{'tosa.resize' op expect input/output height/width dims to be < 16384, got [OH, OW, IH, IW] = 32767, 32767, 16384, 16384}}
  %1 = tosa.resize %arg0, %scale, %offset, %border mode<BILINEAR> : (tensor<1x16384x16384x8xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x32767x32767x8xf32>
  return %1 : tensor<1x32767x32767x8xf32>
}

// -----

// CHECK-LABEL: test_resize_invalid_offset_x
func.func @test_resize_invalid_offset_x(%arg0: tensor<1x8x8x8xf32>) -> tensor<?x?x?x?xf32> {
  %scale = tosa.const_shape values(dense<[1, 1, 1, 1]> : tensor<4xindex>) : () -> !tosa.shape<4>
  %offset = tosa.const_shape values(dense<[0, -2]> : tensor<2xindex>) : () -> !tosa.shape<2>
  %border = tosa.const_shape values(dense<0> : tensor<2xindex>) : () -> !tosa.shape<2>
  // expected-error@+1 {{'tosa.resize' op expect offsetX / scaleXNumerator to be in range [-1, 16), got -2/1}}
  %1 = tosa.resize %arg0, %scale, %offset, %border mode<BILINEAR> : (tensor<1x8x8x8xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: test_resize_invalid_border_x
func.func @test_resize_invalid_boarder_x(%arg0: tensor<1x8x8x8xf32>) -> tensor<?x?x?x?xf32> {
  %scale = tosa.const_shape values(dense<[1, 1, 1, 1]> : tensor<4xindex>) : () -> !tosa.shape<4>
  %offset = tosa.const_shape values(dense<0> : tensor<2xindex>) : () -> !tosa.shape<2>
  %border = tosa.const_shape values(dense<[0, 2]> : tensor<2xindex>) : () -> !tosa.shape<2>
  // expected-error@+1 {{'tosa.resize' op expect borderX / scaleXNumerator to be in range [-16, 1), got 2/1}}
  %1 = tosa.resize %arg0, %scale, %offset, %border mode<BILINEAR> : (tensor<1x8x8x8xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: test_resize_invalid_offset_y
func.func @test_resize_invalid_offset_y(%arg0: tensor<1x8x8x8xf32>) -> tensor<?x?x?x?xf32> {
  %scale = tosa.const_shape values(dense<[1, 1, 1, 1]> : tensor<4xindex>) : () -> !tosa.shape<4>
  %offset = tosa.const_shape values(dense<[17, 0]> : tensor<2xindex>) : () -> !tosa.shape<2>
  %border = tosa.const_shape values(dense<0> : tensor<2xindex>) : () -> !tosa.shape<2>
  // expected-error@+1 {{'tosa.resize' op expect offsetY / scaleYNumerator to be in range [-1, 16), got 17/1}}
  %1 = tosa.resize %arg0, %scale, %offset, %border mode<BILINEAR> : (tensor<1x8x8x8xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: test_resize_invalid_border_y
func.func @test_resize_invalid_boarder_y(%arg0: tensor<1x8x8x8xf32>) -> tensor<?x?x?x?xf32> {
  %scale = tosa.const_shape values(dense<[1, 1, 1, 1]> : tensor<4xindex>) : () -> !tosa.shape<4>
  %offset = tosa.const_shape values(dense<0> : tensor<2xindex>) : () -> !tosa.shape<2>
  %border = tosa.const_shape values(dense<[-17, 0]> : tensor<2xindex>) : () -> !tosa.shape<2>
  // expected-error@+1 {{'tosa.resize' op expect borderY / scaleYNumerator to be in range [-16, 1), got -17/1}}
  %1 = tosa.resize %arg0, %scale, %offset, %border mode<BILINEAR> : (tensor<1x8x8x8xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: test_resize_large_image_size
func.func @test_resize_large_image_size(%arg0: tensor<1x16384x16384x8xf32>) -> tensor<1x32767x32767x8xf32> {
  %scale = tosa.const_shape values(dense<[2, 1, 2, 1]> : tensor<4xindex>) : () -> !tosa.shape<4>
  %offset = tosa.const_shape values(dense<0> : tensor<2xindex>) : () -> !tosa.shape<2>
  %border = tosa.const_shape values(dense<0> : tensor<2xindex>) : () -> !tosa.shape<2>
  // expected-error@+1 {{'tosa.resize' op expect input/output height/width dims to be < 16384, got [OH, OW, IH, IW] = 32767, 32767, 16384, 16384}}
  %1 = tosa.resize %arg0, %scale, %offset, %border mode<BILINEAR> : (tensor<1x16384x16384x8xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x32767x32767x8xf32>
  return %1 : tensor<1x32767x32767x8xf32>
}
