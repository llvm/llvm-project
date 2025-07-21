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

// -----
// CHECK-LABEL: test_i16_table_size
func.func @test_i16_table_size(%arg0: tensor<2x64xi16>, %arg1: tensor<256xi16>) -> tensor<2x64xi32> {
  // expected-error@+1 {{'tosa.table' op requires table size of 513, got 256}}
    %0 = tosa.table %arg0, %arg1 : (tensor<2x64xi16>, tensor<256xi16>) -> tensor<2x64xi32>
    return %0 : tensor<2x64xi32>
}

// -----
// CHECK-LABEL: test_i8_table_size
func.func @test_i8_table_size(%arg0: tensor<2x64xi8>, %arg1: tensor<513xi8>) -> tensor<2x64xi8> {
  // expected-error@+1 {{'tosa.table' op requires table size of 256, got 513}}
    %0 = tosa.table %arg0, %arg1 : (tensor<2x64xi8>, tensor<513xi8>) -> tensor<2x64xi8>
    return %0 : tensor<2x64xi8>
}

// -----
// CHECK-LABEL: test_error_scale32_with_i48
func.func @test_error_scale32_with_i48(%arg0: tensor<1xi48>) -> tensor<1xi8> {
  %multiplier = "tosa.const"() {values = dense<19689> : tensor<1xi32> } : () -> tensor<1xi32>
  %shift = "tosa.const"() {values = dense<30> : tensor<1xi8> } : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<0> : tensor<1xi48>} : () -> tensor<1xi48>
  %output_zp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  // expected-error@+1 {{'tosa.rescale' op scale32 is not allowed with 48-bit input}}
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {scale32 = true, rounding_mode = "SINGLE_ROUND", per_channel = false, input_unsigned = false, output_unsigned = false} : (tensor<1xi48>, tensor<1xi32>, tensor<1xi8>, tensor<1xi48>, tensor<1xi8>) -> tensor<1xi8>
  return %0 : tensor<1xi8>
}

// -----
// CHECK-LABEL: test_error_input_output_unsigned
func.func @test_error_input_output_unsigned(%arg0: tensor<1xi8>) -> tensor<1xi16> {
  %multiplier = "tosa.const"() {values = dense<19689> : tensor<1xi16> } : () -> tensor<1xi16>
  %shift = "tosa.const"() {values = dense<30> : tensor<1xi8> } : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  %output_zp = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
  // expected-error@+1 {{'tosa.rescale' op input and output cannot be both unsigned}}
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {scale32 = false, rounding_mode = "SINGLE_ROUND", per_channel = false, input_unsigned = true, output_unsigned = true} : (tensor<1xi8>, tensor<1xi16>, tensor<1xi8>, tensor<1xi8>, tensor<1xi16>) -> tensor<1xi16>
  return %0 : tensor<1xi16>
}

// -----
// CHECK-LABEL: test_error_i32_output_unsigned_input
func.func @test_error_i32_output_unsigned_input(%arg0: tensor<1xi8>) -> tensor<1xi32> {
  %multiplier = "tosa.const"() {values = dense<19689> : tensor<1xi16> } : () -> tensor<1xi16>
  %shift = "tosa.const"() {values = dense<30> : tensor<1xi8> } : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  %output_zp = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  // expected-error@+1 {{'tosa.rescale' op i32 output type is not allowed with unsigned input}}
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {scale32 = false, rounding_mode = "SINGLE_ROUND", per_channel = false, input_unsigned = true, output_unsigned = false} : (tensor<1xi8>, tensor<1xi16>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1xi32>
  return %0 : tensor<1xi32>
}

// -----
// CHECK-LABEL: test_error_i32_input_unsigned_output
func.func @test_error_i32_input_unsigned_output(%arg0: tensor<1xi32>) -> tensor<1xi8> {
  %multiplier = "tosa.const"() {values = dense<19689> : tensor<1xi16> } : () -> tensor<1xi16>
  %shift = "tosa.const"() {values = dense<30> : tensor<1xi8> } : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %output_zp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  // expected-error@+1 {{'tosa.rescale' op i32 input type is not allowed with unsigned output}}
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {scale32 = false, rounding_mode = "SINGLE_ROUND", per_channel = false, input_unsigned = false, output_unsigned = true} : (tensor<1xi32>, tensor<1xi16>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1xi8>
  return %0 : tensor<1xi8>
}

// -----
// CHECK-LABEL: test_error_i48_input_unsigned_output
func.func @test_error_i48_input_unsigned_output(%arg0: tensor<1xi48>) -> tensor<1xi8> {
  %multiplier = "tosa.const"() {values = dense<19689> : tensor<1xi16> } : () -> tensor<1xi16>
  %shift = "tosa.const"() {values = dense<30> : tensor<1xi8> } : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<0> : tensor<1xi48>} : () -> tensor<1xi48>
  %output_zp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  // expected-error@+1 {{'tosa.rescale' op i48 input type is not allowed with unsigned output}}
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {scale32 = false, rounding_mode = "SINGLE_ROUND", per_channel = false, input_unsigned = false, output_unsigned = true} : (tensor<1xi48>, tensor<1xi16>, tensor<1xi8>, tensor<1xi48>, tensor<1xi8>) -> tensor<1xi8>
  return %0 : tensor<1xi8>
}

// -----
// CHECK-LABEL: test_error_i48_unsigned_input
func.func @test_error_i48_input_unsigned_output(%arg0: tensor<1xi48>) -> tensor<1xi8> {
  %multiplier = "tosa.const"() {values = dense<19689> : tensor<1xi16> } : () -> tensor<1xi16>
  %shift = "tosa.const"() {values = dense<30> : tensor<1xi8> } : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<0> : tensor<1xi48>} : () -> tensor<1xi48>
  %output_zp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  // expected-error@+1 {{'tosa.rescale' op i48 input type cannot be unsigned}}
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {scale32 = false, rounding_mode = "SINGLE_ROUND", per_channel = false, input_unsigned = true, output_unsigned = false} : (tensor<1xi48>, tensor<1xi16>, tensor<1xi8>, tensor<1xi48>, tensor<1xi8>) -> tensor<1xi8>
  return %0 : tensor<1xi8>
}

// -----
// CHECK-LABEL: test_error_i32_unsigned_input
func.func @test_error_i32_input_unsigned_output(%arg0: tensor<1xi32>) -> tensor<1xi8> {
  %multiplier = "tosa.const"() {values = dense<19689> : tensor<1xi16> } : () -> tensor<1xi16>
  %shift = "tosa.const"() {values = dense<30> : tensor<1xi8> } : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %output_zp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  // expected-error@+1 {{'tosa.rescale' op i32 input type cannot be unsigned}}
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {scale32 = false, rounding_mode = "SINGLE_ROUND", per_channel = false, input_unsigned = true, output_unsigned = false} : (tensor<1xi32>, tensor<1xi16>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1xi8>
  return %0 : tensor<1xi8>
}

// -----
// CHECK-LABEL: test_error_i32_unsigned_output
func.func @test_error_i32_unsigned_output(%arg0: tensor<1xi8>) -> tensor<1xi32> {
  %multiplier = "tosa.const"() {values = dense<19689> : tensor<1xi16> } : () -> tensor<1xi16>
  %shift = "tosa.const"() {values = dense<30> : tensor<1xi8> } : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  %output_zp = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  // expected-error@+1 {{'tosa.rescale' op i32 output type cannot be unsigned}}
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {scale32 = false, rounding_mode = "SINGLE_ROUND", per_channel = false, input_unsigned = false, output_unsigned = true} : (tensor<1xi8>, tensor<1xi16>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1xi32>
  return %0 : tensor<1xi32>
}

// -----

func.func @test_cond_if_then_not_isolated_from_above(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i1>) -> tensor<f32> {
    // expected-error@+1 {{'tosa.cond_if' op is not conformant to the TOSA specification. It requires the 'then' region is isolated from above.}}
    %0 = "tosa.cond_if"(%arg2, %arg1) ({
      ^bb0(%arg3: tensor<f32>):
        tosa.yield %arg1 : tensor<f32>
      },  {
      ^bb0(%arg3: tensor<f32>):
        tosa.yield %arg3 : tensor<f32>
      }) : (tensor<i1>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
}

// -----

func.func @test_cond_if_else_not_isolated_from_above(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i1>) -> tensor<f32> {
  // expected-error@+1 {{'tosa.cond_if' op is not conformant to the TOSA specification. It requires the 'else' region is isolated from above.}}
  %0 = "tosa.cond_if"(%arg2, %arg0, %arg1) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      tosa.yield %arg3 : tensor<f32>
    },  {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %add = tosa.add %arg0, %arg4 : (tensor<f32>, tensor<f32>) -> tensor<f32>
      tosa.yield %add : tensor<f32>
    }) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @test_cond_if_simplified_form_not_isolated_from_above(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i1>) -> tensor<f32> {
  // expected-error@+1 {{'tosa.cond_if' op is not conformant to the TOSA specification. It requires the 'then' region is isolated from above.}}
  %0 = tosa.cond_if %arg2 -> (tensor<f32>) {
    tosa.yield %arg0 : tensor<f32>
  } else {
    tosa.yield %arg1 : tensor<f32>
  }
  return %0 : tensor<f32>
}

// -----

// Check isolated cond_if's are valid
func.func @test_cond_if_isolated_from_above(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i1>) -> tensor<f32> {
  %0 = "tosa.cond_if"(%arg2, %arg0, %arg1) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      tosa.yield %arg3 : tensor<f32>
    },  {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      tosa.yield %arg4 : tensor<f32>
    }) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}
