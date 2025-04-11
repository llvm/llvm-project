// RUN: mlir-opt %s -split-input-file -verify-diagnostics --tosa-validate="level=none profile=pro_int,pro_fp extension=int16,int4,bf16,fp8e4m3,fp8e5m2,fft,variable,controlflow,dynamic strict-op-spec-alignment"

// -----

// CHECK-LABEL: test_resize_large_image_size
func.func @test_resize_large_image_size(%arg0: tensor<1x16384x16384x8xf32>) -> tensor<1x32767x32767x8xf32> {
  %scale = tosa.const_shape { values = dense<[2, 1, 2, 1]> : tensor<4xindex> } : () -> !tosa.shape<4>
  %offset = tosa.const_shape { values = dense<0> : tensor<2xindex> } : () -> !tosa.shape<2>
  %border = tosa.const_shape { values = dense<0> : tensor<2xindex> } : () -> !tosa.shape<2>
  // expected-error@+1 {{'tosa.resize' op expect input/output height/width dims to be < 16384, got [OH, OW, IH, IW] = 32767, 32767, 16384, 16384}}
  %1 = tosa.resize %arg0, %scale, %offset, %border { mode = "BILINEAR" } : (tensor<1x16384x16384x8xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x32767x32767x8xf32>
  return %1 : tensor<1x32767x32767x8xf32>
}

// -----

// CHECK-LABEL: test_resize_invalid_scale_numerator
func.func @test_resize_invalid_scale_numerator(%arg0: tensor<1x9x9x8xf32>) -> tensor<?x?x?x?xf32> {
  %scale = tosa.const_shape { values = dense<[2049, 8, 1, 2]> : tensor<4xindex> } : () -> !tosa.shape<4>
  %offset = tosa.const_shape { values = dense<0> : tensor<2xindex> } : () -> !tosa.shape<2>
  %border = tosa.const_shape { values = dense<0> : tensor<2xindex> } : () -> !tosa.shape<2>
  // expected-error@+1 {{'tosa.resize' op expect all scale numerator values to be <= (1 << 11), got scale_y_n=2049, scale_x_n=1}}
  %1 = tosa.resize %arg0, %scale, %offset, %border { mode = "BILINEAR" } : (tensor<1x9x9x8xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: test_resize_invalid_downscale
func.func @test_resize_invalid_downscale(%arg0: tensor<1x37x37x8xf32>) -> tensor<?x?x?x?xf32> {
  %scale = tosa.const_shape { values = dense<[1, 18, 1, 18]> : tensor<4xindex> } : () -> !tosa.shape<4>
  %offset = tosa.const_shape { values = dense<0> : tensor<2xindex> } : () -> !tosa.shape<2>
  %border = tosa.const_shape { values = dense<0> : tensor<2xindex> } : () -> !tosa.shape<2>
  // expected-error@+1 {{'tosa.resize' op expect a downscale ratio larger than 1/16, got y=1/18, x=1/18}}
  %1 = tosa.resize %arg0, %scale, %offset, %border { mode = "BILINEAR" } : (tensor<1x37x37x8xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: test_resize_invalid_offset_y
func.func @test_resize_invalid_offset_y(%arg0: tensor<1x8x8x8xf32>) -> tensor<?x?x?x?xf32> {
  %scale = tosa.const_shape { values = dense<[1, 1, 1, 1]> : tensor<4xindex> } : () -> !tosa.shape<4>
  %offset = tosa.const_shape { values = dense<[17, 0]> : tensor<2xindex> } : () -> !tosa.shape<2>
  %border = tosa.const_shape { values = dense<0> : tensor<2xindex> } : () -> !tosa.shape<2>
  // expected-error@+1 {{'tosa.resize' op expect offsetY / scaleYNumerator to be in range [-1, 16), got 17/1}}
  %1 = tosa.resize %arg0, %scale, %offset, %border { mode = "BILINEAR" } : (tensor<1x8x8x8xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: test_resize_invalid_offset_x
func.func @test_resize_invalid_offset_x(%arg0: tensor<1x8x8x8xf32>) -> tensor<?x?x?x?xf32> {
  %scale = tosa.const_shape { values = dense<[1, 1, 1, 1]> : tensor<4xindex> } : () -> !tosa.shape<4>
  %offset = tosa.const_shape { values = dense<[0, -2]> : tensor<2xindex> } : () -> !tosa.shape<2>
  %border = tosa.const_shape { values = dense<0> : tensor<2xindex> } : () -> !tosa.shape<2>
  // expected-error@+1 {{'tosa.resize' op expect offsetX / scaleXNumerator to be in range [-1, 16), got -2/1}}
  %1 = tosa.resize %arg0, %scale, %offset, %border { mode = "BILINEAR" } : (tensor<1x8x8x8xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: test_resize_invalid_border_y
func.func @test_resize_invalid_boarder_y(%arg0: tensor<1x8x8x8xf32>) -> tensor<?x?x?x?xf32> {
  %scale = tosa.const_shape { values = dense<[1, 1, 1, 1]> : tensor<4xindex> } : () -> !tosa.shape<4>
  %offset = tosa.const_shape { values = dense<0> : tensor<2xindex> } : () -> !tosa.shape<2>
  %border = tosa.const_shape { values = dense<[-17, 0]> : tensor<2xindex> } : () -> !tosa.shape<2>
  // expected-error@+1 {{'tosa.resize' op expect borderY / scaleYNumerator to be in range [-16, 1), got -17/1}}
  %1 = tosa.resize %arg0, %scale, %offset, %border { mode = "BILINEAR" } : (tensor<1x8x8x8xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: test_resize_invalid_border_x
func.func @test_resize_invalid_boarder_x(%arg0: tensor<1x8x8x8xf32>) -> tensor<?x?x?x?xf32> {
  %scale = tosa.const_shape { values = dense<[1, 1, 1, 1]> : tensor<4xindex> } : () -> !tosa.shape<4>
  %offset = tosa.const_shape { values = dense<0> : tensor<2xindex> } : () -> !tosa.shape<2>
  %border = tosa.const_shape { values = dense<[0, 2]> : tensor<2xindex> } : () -> !tosa.shape<2>
  // expected-error@+1 {{'tosa.resize' op expect borderX / scaleXNumerator to be in range [-16, 1), got 2/1}}
  %1 = tosa.resize %arg0, %scale, %offset, %border { mode = "BILINEAR" } : (tensor<1x8x8x8xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: test_mul_negative_shift
func.func @test_mul_negative_shift(%arg0: tensor<1x8x8x8xi32>, %arg1: tensor<1x8x8x8xi32>) -> tensor<1x8x8x8xi32> {
  %shift = "tosa.const" () { values = dense<-1> : tensor<1xi8> } : () -> tensor<1xi8>
  // expected-error@+1 {{'tosa.mul' op requires 0 <= shift && shift <= 63, but got: -1}}
  %mul = tosa.mul %arg0, %arg1, %shift : (tensor<1x8x8x8xi32>, tensor<1x8x8x8xi32>, tensor<1xi8>) -> tensor<1x8x8x8xi32>
  return %mul : tensor<1x8x8x8xi32>
}

// -----

// CHECK-LABEL: test_mul_too_big_shift
func.func @test_mul_too_big_shift(%arg0: tensor<1x8x8x8xi32>, %arg1: tensor<1x8x8x8xi32>) -> tensor<1x8x8x8xi32> {
  %shift = "tosa.const" () { values = dense<64> : tensor<1xi8> } : () -> tensor<1xi8>
  // expected-error@+1 {{'tosa.mul' op requires 0 <= shift && shift <= 63, but got: 64}}
  %mul = tosa.mul %arg0, %arg1, %shift : (tensor<1x8x8x8xi32>, tensor<1x8x8x8xi32>, tensor<1xi8>) -> tensor<1x8x8x8xi32>
  return %mul : tensor<1x8x8x8xi32>
}

// -----

// CHECK-LABEL: test_mul_non_zero_shift
func.func @test_mul_non_zero_shift(%arg0: tensor<1x8x8x8xi16>, %arg1: tensor<1x8x8x8xi16>) -> tensor<1x8x8x8xi32> {
  %shift = "tosa.const" () { values = dense<1> : tensor<1xi8> } : () -> tensor<1xi8>
  // expected-error@+1 {{'tosa.mul' op requires shift = 0 for all input data types that are not int32_t, but got: 1}}
  %mul = tosa.mul %arg0, %arg1, %shift : (tensor<1x8x8x8xi16>, tensor<1x8x8x8xi16>, tensor<1xi8>) -> tensor<1x8x8x8xi32>
  return %mul : tensor<1x8x8x8xi32>
}
