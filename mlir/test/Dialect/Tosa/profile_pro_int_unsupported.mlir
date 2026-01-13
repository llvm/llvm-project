//--------------------------------------------------------------------------------------------------
// Enable all supported extensions to focus the verification of expected profile requirement errors.
//--------------------------------------------------------------------------------------------------

// RUN: mlir-opt %s -split-input-file -verify-diagnostics -tosa-attach-target="profiles=pro_fp extensions=int16,int4,bf16,fp8e4m3,fp8e5m2,fft,variable,controlflow,dynamic,doubleround,inexactround" -tosa-validate="strict-op-spec-alignment"

// -----
func.func @test_const_i1() -> tensor<3x11x11x3xi1> {
  %0 = "tosa.const"() {values = dense<0> : tensor<3x11x11x3xi1>} : () -> tensor<3x11x11x3xi1>
  return %0 : tensor<3x11x11x3xi1>
}

// -----
func.func @test_const_i32() -> tensor<3x11x11x3xi32> {
  %0 = "tosa.const"() {values = dense<0> : tensor<3x11x11x3xi32>} : () -> tensor<3x11x11x3xi32>
  return %0 : tensor<3x11x11x3xi32>
}

// -----
func.func @test_argmax(%arg0: tensor<14x19xi8>) -> tensor<14xi32> {
  // expected-error@+1 {{'tosa.argmax' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.argmax %arg0 {axis = 1 : i32} : (tensor<14x19xi8>) -> tensor<14xi32>
  return %0 : tensor<14xi32>
}

// -----
func.func @test_avg_pool2d(%arg0: tensor<1x7x7x9xi8>, %arg1: tensor<1xi8>, %arg2: tensor<1xi8>) -> tensor<1x7x7x9xi8> {
  // expected-error@+1 {{'tosa.avg_pool2d' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.avg_pool2d %arg0, %arg1, %arg2 {acc_type = i32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x9xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x7x7x9xi8>
  return %0 : tensor<1x7x7x9xi8>
}

// -----
func.func @test_conv2d(%arg0: tensor<1x4x4x4xi8>, %arg1: tensor<8x1x1x4xi8>, %arg2: tensor<8xi32>, %arg3: tensor<1xi8>) -> tensor<1x4x4x8xi32> {
  // expected-error@+1 {{'tosa.conv2d' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.conv2d %arg0, %arg1, %arg2, %arg3, %arg3 {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, local_bound = true} : (tensor<1x4x4x4xi8>, tensor<8x1x1x4xi8>, tensor<8xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x4x4x8xi32>
  return %0 : tensor<1x4x4x8xi32>
}

// -----
func.func @test_conv3d(%arg0: tensor<1x4x8x21x17xi8>, %arg1: tensor<34x1x1x1x17xi8>, %arg2: tensor<34xi32>, %arg3: tensor<1xi8>, %arg4: tensor<1xi8>) -> tensor<1x4x8x21x34xi32> {
  // expected-error@+1 {{'tosa.conv3d' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.conv3d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = i32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 1, 1, 1>} : (tensor<1x4x8x21x17xi8>, tensor<34x1x1x1x17xi8>, tensor<34xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x4x8x21x34xi32>
  return %0 : tensor<1x4x8x21x34xi32>
}

// -----
func.func @test_depthwise_conv2d(%arg0: tensor<1x4x4x4xi8>, %arg1: tensor<1x1x4x2xi8>, %arg2: tensor<8xi32>, %arg3: tensor<1xi8>, %arg4: tensor<1xi8>) -> tensor<1x4x4x8xi32> {
  // expected-error@+1 {{'tosa.depthwise_conv2d' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x4x4x4xi8>, tensor<1x1x4x2xi8>, tensor<8xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x4x4x8xi32>
  return %0 : tensor<1x4x4x8xi32>
}

// -----
func.func @test_matmul(%arg0: tensor<1x14x19xi8>, %arg1: tensor<1x19x28xi8>) -> tensor<1x14x28xi32> {
  %azp0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %bzp0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  // expected-error@+1 {{'tosa.matmul' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.matmul %arg0, %arg1, %azp0, %bzp0 : (tensor<1x14x19xi8>, tensor<1x19x28xi8>, tensor<1xi8>, tensor<1xi8>)  -> tensor<1x14x28xi32>
  return %0 : tensor<1x14x28xi32>
}

// -----
func.func @test_max_pool2d(%arg0: tensor<1x32x32x8xi8>) -> tensor<1x32x32x8xi8> {
  // expected-error@+1 {{'tosa.max_pool2d' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xi8>) -> tensor<1x32x32x8xi8>
  return %0 : tensor<1x32x32x8xi8>
}

// -----
func.func @test_transpose_conv2d(%arg0: tensor<1x32x32x8xi8>, %arg1: tensor<16x1x1x8xi8>, %arg2: tensor<16xi32>, %arg3: tensor<1xi8>, %arg4: tensor<1xi8>) -> tensor<1x32x32x16xi32> {
  // expected-error@+1 {{'tosa.transpose_conv2d' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = i32, out_pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xi8>, tensor<16x1x1x8xi8>, tensor<16xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x32x32x16xi32>
  return %0 : tensor<1x32x32x16xi32>
}

// -----
func.func @test_clamp(%arg0: tensor<13x21x3xi8>) -> tensor<13x21x3xi8> {
  // expected-error@+1 {{'tosa.clamp' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.clamp %arg0 {min_val = 0 : i8, max_val = 1: i8} : (tensor<13x21x3xi8>) -> tensor<13x21x3xi8>
  return %0 : tensor<13x21x3xi8>
}

// -----
func.func @test_arithmetic_right_shift(%arg0: tensor<13x21x1xi32>, %arg1: tensor<13x21x3xi32>) -> tensor<13x21x3xi32> {
  // expected-error@+1 {{'tosa.arithmetic_right_shift' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = false} : (tensor<13x21x1xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
func.func @test_bitwise_and(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x21x1xi32>) -> tensor<13x21x3xi32> {
  // expected-error@+1 {{'tosa.bitwise_and' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<13x21x3xi32>, tensor<13x21x1xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
func.func @test_max(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x21x1xi32>) -> tensor<13x21x3xi32> {
  // expected-error@+1 {{'tosa.maximum' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.maximum %arg0, %arg1 : (tensor<13x21x3xi32>, tensor<13x21x1xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
func.func @test_mul(%arg0: tensor<13x21x3xi16>, %arg1: tensor<13x1x3xi16>, %shift: tensor<1xi8>) -> tensor<13x21x3xi32> {
  // expected-error@+1 {{'tosa.mul' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.mul %arg0, %arg1, %shift : (tensor<13x21x3xi16>, tensor<13x1x3xi16>, tensor<1xi8>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
func.func @test_table(%arg0 : tensor<4x5xi8>, %arg1 : tensor<513xi8>) -> () {
  // expected-error@+1 {{'tosa.table' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.table %arg0, %arg1 : (tensor<4x5xi8>, tensor<513xi8>) -> tensor<?x?xi8>
  return
}

// -----
func.func @test_abs(%arg0: tensor<13x21x3xi32>) -> tensor<13x21x3xi32> {
  // expected-error@+1 {{'tosa.abs' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.abs %arg0 : (tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
func.func @test_bitwise_not(%arg0: tensor<13x21x1xi32>) -> tensor<13x21x1xi32> {
  // expected-error@+1 {{'tosa.bitwise_not' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.bitwise_not %arg0 : (tensor<13x21x1xi32>) -> tensor<13x21x1xi32>
  return %0 : tensor<13x21x1xi32>
}

// -----
func.func @test_clz(%arg0: tensor<13x21x3xi32>) -> tensor<13x21x3xi32> {
  // expected-error@+1 {{'tosa.clz' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.clz %arg0 : (tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
func.func @test_negate(%arg0: tensor<13x21x3xi16>, %arg1: tensor<1xi16>, %arg2: tensor<1xi16>) -> tensor<13x21x3xi16> {
  // expected-error@+1 {{'tosa.negate' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.negate %arg0, %arg1, %arg2 : (tensor<13x21x3xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<13x21x3xi16>
  return %0 : tensor<13x21x3xi16>
}

// -----
func.func @test_select(%arg0: tensor<1x1x1xi1>, %arg1: tensor<13x21x3xi8>, %arg2: tensor<13x21x3xi8>) -> tensor<13x21x3xi8> {
  // expected-error@+1 {{'tosa.select' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.select %arg0, %arg1, %arg2 : (tensor<1x1x1xi1>, tensor<13x21x3xi8>, tensor<13x21x3xi8>) -> tensor<13x21x3xi8>
  return %0 : tensor<13x21x3xi8>
}

// -----
func.func @test_equal(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x1x3xi32>) -> tensor<13x21x3xi1> {
  // expected-error@+1 {{'tosa.equal' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.equal %arg0, %arg1 : (tensor<13x21x3xi32>, tensor<13x1x3xi32>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----
func.func @test_greater(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x1x3xi32>) -> tensor<13x21x3xi1> {
  // expected-error@+1 {{'tosa.greater' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.greater %arg0, %arg1 : (tensor<13x21x3xi32>, tensor<13x1x3xi32>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----
func.func @test_reduce_max(%arg0: tensor<13x21x3xi8>) -> tensor<1x21x3xi8> {
  // expected-error@+1 {{'tosa.reduce_max' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<13x21x3xi8>) -> tensor<1x21x3xi8>
  return %0 : tensor<1x21x3xi8>
}

// -----
func.func @test_reduce_sum(%arg0: tensor<13x21x3xi32>) -> tensor<1x21x3xi32> {
  // expected-error@+1 {{'tosa.reduce_sum' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<13x21x3xi32>) -> tensor<1x21x3xi32>
  return %0 : tensor<1x21x3xi32>
}

// -----
func.func @test_concat(%arg0: tensor<13x21x3xi16>, %arg1: tensor<13x21x3xi16>) -> tensor<26x21x3xi16> {
  // expected-error@+1 {{'tosa.concat' op illegal: requires [pro_int] to work with but not enabled in target}}
  %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<13x21x3xi16>, tensor<13x21x3xi16>) -> tensor<26x21x3xi16>
  return %0 : tensor<26x21x3xi16>
}

// -----
func.func @test_pad(%arg0: tensor<13x21x3xi8>) -> tensor<13x21x3xi8> {
  %padding = tosa.const_shape {values = dense<0> : tensor<6xindex>} : () -> !tosa.shape<6>
  %pad_const = "tosa.const"() {values = dense<1> : tensor<1xi8>} : () -> tensor<1xi8>
  // expected-error@+1 {{'tosa.pad' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.pad %arg0, %padding, %pad_const : (tensor<13x21x3xi8>, !tosa.shape<6>, tensor<1xi8>) -> tensor<13x21x3xi8>
  return %0 : tensor<13x21x3xi8>
}

// -----
func.func @test_reshape(%arg0: tensor<13x21x3xi8>) -> tensor<1x819xi8> {
  %1 = tosa.const_shape {values = dense<[1, 819]> : tensor<2xindex>} : () -> !tosa.shape<2>
  // expected-error@+1 {{'tosa.reshape' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.reshape %arg0, %1 : (tensor<13x21x3xi8>, !tosa.shape<2>) -> tensor<1x819xi8>
  return %0 : tensor<1x819xi8>
}

// -----
func.func @test_reverse(%arg0: tensor<13x21x3xi8>) -> tensor<13x21x3xi8> {
  // expected-error@+1 {{'tosa.reverse' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<13x21x3xi8>) -> tensor<13x21x3xi8>
  return %0 : tensor<13x21x3xi8>
}

// -----
func.func @test_slice(%arg0: tensor<13x21x3xi8>) -> tensor<4x11x1xi8> {
  %0 = tosa.const_shape {values = dense<[4, 11, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %1 = tosa.const_shape {values = dense<[6, 8, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
  // expected-error@+1 {{'tosa.slice' op illegal: requires [pro_int] but not enabled in target}}
  %2 = tosa.slice %arg0, %0, %1 : (tensor<13x21x3xi8>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<4x11x1xi8>
  return %2 : tensor<4x11x1xi8>
}

// -----
func.func @test_tile(%arg0: tensor<13x21x3xi8>) -> tensor<39x21x6xi8> {
  %cst = tosa.const_shape { values = dense<[3, 1, 2]> : tensor<3xindex> } : () -> !tosa.shape<3>
  // expected-error@+1 {{'tosa.tile' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.tile %arg0, %cst: (tensor<13x21x3xi8>, !tosa.shape<3>) -> tensor<39x21x6xi8>
  return %0 : tensor<39x21x6xi8>
}

// -----
func.func @test_transpose(%arg0: tensor<13x21x3xi8>, %arg1: tensor<3xi32>) -> tensor<3x13x21xi8> {
  // expected-error@+1 {{'tosa.transpose' op illegal: requires [pro_int] but not enabled in target}}
  %1 = tosa.transpose %arg0 {perms = array<i32: 2, 0, 1>}: (tensor<13x21x3xi8>) -> tensor<3x13x21xi8>
  return %1 : tensor<3x13x21xi8>
}

// -----
func.func @test_gather(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x26xi32>) -> tensor<13x26x3xi32> {
  // expected-error@+1 {{'tosa.gather' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.gather %arg0, %arg1 : (tensor<13x21x3xi32>, tensor<13x26xi32>) -> tensor<13x26x3xi32>
  return %0 : tensor<13x26x3xi32>
}

// -----
func.func @test_scatter(%arg0: tensor<13x27x3xi32>, %arg1: tensor<13x26xi32>, %arg2: tensor<13x26x3xi32>) -> tensor<13x27x3xi32> {
  // expected-error@+1 {{'tosa.scatter' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.scatter %arg0, %arg1, %arg2 : (tensor<13x27x3xi32>, tensor<13x26xi32>, tensor<13x26x3xi32>) -> tensor<13x27x3xi32>
  return %0 : tensor<13x27x3xi32>
}

// -----
func.func @test_resize(%arg0: tensor<1x32x32x8xi8>) -> tensor<1x64x64x8xi32> {
  %scale = tosa.const_shape { values = dense<[4, 2, 4, 2]> : tensor<4xindex> } : () -> !tosa.shape<4>
  %offset = tosa.const_shape { values = dense<[-1, -1]> : tensor<2xindex> } : () -> !tosa.shape<2>
  %border = tosa.const_shape { values = dense<[1, 1]> : tensor<2xindex> } : () -> !tosa.shape<2>
  // expected-error@+1 {{'tosa.resize' op illegal: requires [pro_int] but not enabled in target}}
  %1 = tosa.resize %arg0, %scale, %offset, %border { mode = BILINEAR } : (tensor<1x32x32x8xi8>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x64x64x8xi32>
  return %1 : tensor<1x64x64x8xi32>
}

// -----
func.func @test_cast_i1_i8(%arg0: tensor<13x21x3xi1>) -> tensor<13x21x3xi8> {
  // expected-error@+1 {{'tosa.cast' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.cast %arg0 : (tensor<13x21x3xi1>) -> tensor<13x21x3xi8>
  return %0 : tensor<13x21x3xi8>
}

// -----
func.func @test_cast_i8_i32(%arg0: tensor<13x21x3xi8>) -> tensor<13x21x3xi32> {
  // expected-error@+1 {{'tosa.cast' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.cast %arg0 : (tensor<13x21x3xi8>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
func.func @test_cast_i16_i8(%arg0: tensor<13x21x3xi16>) -> tensor<13x21x3xi8> {
  // expected-error@+1 {{'tosa.cast' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.cast %arg0 : (tensor<13x21x3xi16>) -> tensor<13x21x3xi8>
  return %0 : tensor<13x21x3xi8>
}

// -----
func.func @test_cast_i32_i16(%arg0: tensor<13x21x3xi32>) -> tensor<13x21x3xi16> {
  // expected-error@+1 {{'tosa.cast' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.cast %arg0 : (tensor<13x21x3xi32>) -> tensor<13x21x3xi16>
  return %0 : tensor<13x21x3xi16>
}

// -----
func.func @test_rescale(%arg0: tensor<13x21x3xi8>) -> tensor<13x21x3xi32> {
  %multiplier = "tosa.const"() {values = dense<1073741824> : tensor<1xi32>} : () -> tensor<1xi32>
  %shift = "tosa.const"() {values = dense<30> : tensor<1xi8>} : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<127> : tensor<1xi8>} : () -> tensor<1xi8>
  %output_zp = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  // expected-error@+1 {{'tosa.rescale' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {rounding_mode = SINGLE_ROUND, scale32 = true, per_channel = false, input_unsigned = false, output_unsigned = false} : (tensor<13x21x3xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}
