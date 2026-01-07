//--------------------------------------------------------------------------------------------------
// Enable all supported extensions to focus the verification of expected profile requirement errors.
//--------------------------------------------------------------------------------------------------

// RUN: mlir-opt %s -split-input-file -verify-diagnostics -tosa-attach-target="extensions=int16,int4,bf16,fp8e4m3,fp8e5m2,fft,variable,controlflow,dynamic,doubleround,inexactround" -tosa-validate="strict-op-spec-alignment"

// -----
func.func @test_add_i32(%arg0: tensor<13x21x1xi32>, %arg1: tensor<13x21x3xi32>) -> tensor<13x21x3xi32> {
  // expected-error@+1 {{'tosa.add' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.add %arg0, %arg1 : (tensor<13x21x1xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
func.func @test_intdiv(%arg0: tensor<13x21x1xi32>, %arg1: tensor<13x21x3xi32>) -> tensor<13x21x3xi32> {
  // expected-error@+1 {{'tosa.intdiv' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.intdiv %arg0, %arg1 : (tensor<13x21x1xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
func.func @test_logical_and(%arg0: tensor<13x21x3xi1>, %arg1: tensor<13x21x1xi1>) -> tensor<13x21x3xi1> {
  // expected-error@+1 {{'tosa.logical_and' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.logical_and %arg0, %arg1 : (tensor<13x21x3xi1>, tensor<13x21x1xi1>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----
func.func @test_logical_left_shift(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x21x1xi32>) -> tensor<13x21x3xi32> {
  // expected-error@+1 {{'tosa.logical_left_shift' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<13x21x3xi32>, tensor<13x21x1xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
func.func @test_mul(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x1x3xi32>, %shift: tensor<1xi8>) -> tensor<13x21x3xi32> {
  // expected-error@+1 {{'tosa.mul' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.mul %arg0, %arg1, %shift : (tensor<13x21x3xi32>, tensor<13x1x3xi32>, tensor<1xi8>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
func.func @test_sub(%arg0: tensor<1x21x3xi32>, %arg1: tensor<13x21x3xi32>) -> tensor<13x21x3xi32> {
  // expected-error@+1 {{'tosa.sub' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.sub %arg0, %arg1 : (tensor<1x21x3xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
func.func @test_logical_not(%arg0: tensor<1x21x3xi1>) -> tensor<1x21x3xi1> {
  // expected-error@+1 {{'tosa.logical_not' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.logical_not %arg0 : (tensor<1x21x3xi1>) -> tensor<1x21x3xi1>
  return %0 : tensor<1x21x3xi1>
}

// -----
func.func @test_select(%arg0: tensor<1x1x1xi1>, %arg1: tensor<13x21x3xi1>, %arg2: tensor<13x21x3xi1>) -> tensor<13x21x3xi1> {
  // expected-error@+1 {{'tosa.select' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.select %arg0, %arg1, %arg2 : (tensor<1x1x1xi1>, tensor<13x21x3xi1>, tensor<13x21x3xi1>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----
func.func @test_table(%arg0 : tensor<4x5xi8>, %arg1 : tensor<513xi8>) -> () {
  // expected-error@+1 {{'tosa.table' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.table %arg0, %arg1 : (tensor<4x5xi8>, tensor<513xi8>) -> tensor<?x?xi8>
  return
}

// -----
func.func @test_conv2d(%arg0: tensor<1x4x4x4xf32>, %arg1: tensor<8x1x1x4xf32>, %arg2: tensor<8xf32>, %arg3: tensor<1xf32>) -> tensor<1x4x4x8xf32> {
  // expected-error@+1 {{'tosa.conv2d' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.conv2d %arg0, %arg1, %arg2, %arg3, %arg3 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, local_bound = true} : (tensor<1x4x4x4xf32>, tensor<8x1x1x4xf32>, tensor<8xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x4x4x8xf32>
  return %0 : tensor<1x4x4x8xf32>
}

// -----
func.func @test_avg_pool2d(%arg0: tensor<1x7x7x9xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> tensor<1x7x7x9xf32> {
  // expected-error@+1 {{'tosa.avg_pool2d' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.avg_pool2d %arg0, %arg1, %arg2 {acc_type = f32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x9xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x7x7x9xf32>
  return %0 : tensor<1x7x7x9xf32>
}

// -----
func.func @test_matmul(%arg0: tensor<1x14x19xf32>, %arg1: tensor<1x19x28xf32>, %arg2: tensor<1xf32>) -> tensor<1x14x28xf32> {
  // expected-error@+1 {{'tosa.matmul' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.matmul %arg0, %arg1, %arg2, %arg2: (tensor<1x14x19xf32>, tensor<1x19x28xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x14x28xf32>
  return %0 : tensor<1x14x28xf32>
}

// -----
func.func @test_sigmoid(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // expected-error@+1 {{'tosa.sigmoid' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.sigmoid %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
func.func @test_transpose_conv2d(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x1x1x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.transpose_conv2d' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: 1, 32, 32, 16>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xf32>, tensor<16x1x1x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----
func.func @test_add(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // expected-error@+1 {{'tosa.add' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.add %arg0, %arg1 : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
func.func @test_add_i32(%arg0: tensor<13x21x1xi32>, %arg1: tensor<13x21x3xi32>) -> tensor<13x21x3xi32> {
  // expected-error@+1 {{'tosa.add' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.add %arg0, %arg1 : (tensor<13x21x1xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
func.func @test_intdiv(%arg0: tensor<13x21x1xi32>, %arg1: tensor<13x21x3xi32>) -> tensor<13x21x3xi32> {
  // expected-error@+1 {{'tosa.intdiv' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.intdiv %arg0, %arg1 : (tensor<13x21x1xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
func.func @test_logical_and(%arg0: tensor<13x21x3xi1>, %arg1: tensor<13x21x1xi1>) -> tensor<13x21x3xi1> {
  // expected-error@+1 {{'tosa.logical_and' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.logical_and %arg0, %arg1 : (tensor<13x21x3xi1>, tensor<13x21x1xi1>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----
func.func @test_logical_left_shift(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x21x1xi32>) -> tensor<13x21x3xi32> {
  // expected-error@+1 {{'tosa.logical_left_shift' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<13x21x3xi32>, tensor<13x21x1xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
func.func @test_mul(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x1x3xi32>, %shift: tensor<1xi8>) -> tensor<13x21x3xi32> {
  // expected-error@+1 {{'tosa.mul' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.mul %arg0, %arg1, %shift : (tensor<13x21x3xi32>, tensor<13x1x3xi32>, tensor<1xi8>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
func.func @test_sub(%arg0: tensor<1x21x3xi32>, %arg1: tensor<13x21x3xi32>) -> tensor<13x21x3xi32> {
  // expected-error@+1 {{'tosa.sub' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.sub %arg0, %arg1 : (tensor<1x21x3xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
func.func @test_logical_not(%arg0: tensor<1x21x3xi1>) -> tensor<1x21x3xi1> {
  // expected-error@+1 {{'tosa.logical_not' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.logical_not %arg0 : (tensor<1x21x3xi1>) -> tensor<1x21x3xi1>
  return %0 : tensor<1x21x3xi1>
}

// -----
func.func @test_select(%arg0: tensor<1x1x1xi1>, %arg1: tensor<13x21x3xi1>, %arg2: tensor<13x21x3xi1>) -> tensor<13x21x3xi1> {
  // expected-error@+1 {{'tosa.select' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.select %arg0, %arg1, %arg2 : (tensor<1x1x1xi1>, tensor<13x21x3xi1>, tensor<13x21x3xi1>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}
// -----
func.func @test_reduce_all(%arg0: tensor<13x21x3xi1>) -> tensor<1x21x3xi1> {
  // expected-error@+1 {{'tosa.reduce_all' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.reduce_all %arg0 {axis = 0 : i32} : (tensor<13x21x3xi1>) -> tensor<1x21x3xi1>
  return %0 : tensor<1x21x3xi1>
}

// -----
func.func @test_reduce_any(%arg0: tensor<13x21x3xi1>) -> tensor<1x21x3xi1> {
  // expected-error@+1 {{'tosa.reduce_any' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.reduce_any %arg0 {axis = 0 : i32} : (tensor<13x21x3xi1>) -> tensor<1x21x3xi1>
  return %0 : tensor<1x21x3xi1>
}

// -----
func.func @test_concat(%arg0: tensor<13x21x3xi1>, %arg1: tensor<13x21x3xi1>) -> tensor<26x21x3xi1> {
  // expected-error@+1 {{'tosa.concat' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<13x21x3xi1>, tensor<13x21x3xi1>) -> tensor<26x21x3xi1>
  return %0 : tensor<26x21x3xi1>
}

// -----
func.func @test_pad(%arg0: tensor<13x21x3xi1>) -> tensor<13x21x3xi1> {
  // expected-error@+1 {{'tosa.const_shape' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %padding = tosa.const_shape {values = dense<0> : tensor<6xindex>} : () -> !tosa.shape<6>
  // expected-error@+1 {{'tosa.const' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %pad_const = "tosa.const"() {values = dense<1> : tensor<1xi1>} : () -> tensor<1xi1>
  // expected-error@+1 {{'tosa.pad' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.pad %arg0, %padding, %pad_const : (tensor<13x21x3xi1>, !tosa.shape<6>, tensor<1xi1>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----
func.func @test_reshape(%arg0: tensor<13x21x3xi1>) -> tensor<1x819xi1> {
  // expected-error@+1 {{'tosa.const_shape' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %1 = tosa.const_shape {values = dense<[1, 819]> : tensor<2xindex>} : () -> !tosa.shape<2>
  // expected-error@+1 {{'tosa.reshape' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.reshape %arg0, %1 : (tensor<13x21x3xi1>, !tosa.shape<2>) -> tensor<1x819xi1>
  return %0 : tensor<1x819xi1>
}

// -----
func.func @test_reverse(%arg0: tensor<13x21x3xi1>) -> tensor<13x21x3xi1> {
  // expected-error@+1 {{'tosa.reverse' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<13x21x3xi1>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----
func.func @test_slice(%arg0: tensor<13x21x3xi1>) -> tensor<4x11x1xi1> {
  // expected-error@+1 {{'tosa.const_shape' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.const_shape {values = dense<[4, 11, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
  // expected-error@+1 {{'tosa.const_shape' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %1 = tosa.const_shape {values = dense<[6, 8, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
  // expected-error@+1 {{'tosa.slice' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %2 = tosa.slice %arg0, %0, %1 : (tensor<13x21x3xi1>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<4x11x1xi1>
  return %2 : tensor<4x11x1xi1>
}

// -----
func.func @test_tile(%arg0: tensor<13x21x3xi1>) -> tensor<39x21x6xi1> {
  // expected-error@+1 {{'tosa.const_shape' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %cst = tosa.const_shape { values = dense<[3, 1, 2]> : tensor<3xindex> } : () -> !tosa.shape<3>
  // expected-error@+1 {{'tosa.tile' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.tile %arg0, %cst: (tensor<13x21x3xi1>, !tosa.shape<3>) -> tensor<39x21x6xi1>
  return %0 : tensor<39x21x6xi1>
}

// -----
func.func @test_transpose(%arg0: tensor<13x21x3xi1>) -> tensor<3x13x21xi1> {
  // expected-error@+1 {{'tosa.transpose' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %1 = tosa.transpose %arg0 {perms = array<i32: 2, 0, 1>} : (tensor<13x21x3xi1>) -> tensor<3x13x21xi1>
  return %1 : tensor<3x13x21xi1>
}

// -----
func.func @test_concat(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<26x21x3xf32> {
  // expected-error@+1 {{'tosa.concat' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<26x21x3xf32>
  return %0 : tensor<26x21x3xf32>
}

// -----
func.func @test_concat(%arg0: tensor<13x21x3xi1>, %arg1: tensor<13x21x3xi1>) -> tensor<26x21x3xi1> {
  // expected-error@+1 {{'tosa.concat' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<13x21x3xi1>, tensor<13x21x3xi1>) -> tensor<26x21x3xi1>
  return %0 : tensor<26x21x3xi1>
}

// -----
func.func @test_pad(%arg0: tensor<13x21x3xi1>) -> tensor<13x21x3xi1> {
  // expected-error@+1 {{'tosa.const_shape' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %padding = tosa.const_shape {values = dense<0> : tensor<6xindex>} : () -> !tosa.shape<6>
  // expected-error@+1 {{'tosa.const' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %pad_const = "tosa.const"() {values = dense<1> : tensor<1xi1>} : () -> tensor<1xi1>
  // expected-error@+1 {{'tosa.pad' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.pad %arg0, %padding, %pad_const : (tensor<13x21x3xi1>, !tosa.shape<6>, tensor<1xi1>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----
func.func @test_reshape(%arg0: tensor<13x21x3xi1>) -> tensor<1x819xi1> {
  // expected-error@+1 {{'tosa.const_shape' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %1 = tosa.const_shape {values = dense<[1, 819]> : tensor<2xindex>} : () -> !tosa.shape<2>
  // expected-error@+1 {{'tosa.reshape' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.reshape %arg0, %1 : (tensor<13x21x3xi1>, !tosa.shape<2>) -> tensor<1x819xi1>
  return %0 : tensor<1x819xi1>
}

// -----
func.func @test_reverse(%arg0: tensor<13x21x3xi1>) -> tensor<13x21x3xi1> {
  // expected-error@+1 {{'tosa.reverse' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<13x21x3xi1>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----
func.func @test_slice(%arg0: tensor<13x21x3xi1>) -> tensor<4x11x1xi1> {
  // expected-error@+1 {{'tosa.const_shape' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.const_shape {values = dense<[4, 11, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
  // expected-error@+1 {{'tosa.const_shape' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %1 = tosa.const_shape {values = dense<[6, 8, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
  // expected-error@+1 {{'tosa.slice' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %2 = tosa.slice %arg0, %0, %1 : (tensor<13x21x3xi1>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<4x11x1xi1>
  return %2 : tensor<4x11x1xi1>
}

// -----
func.func @test_tile(%arg0: tensor<13x21x3xi1>) -> tensor<39x21x6xi1> {
  // expected-error@+1 {{'tosa.const_shape' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %cst = tosa.const_shape { values = dense<[3, 1, 2]> : tensor<3xindex> } : () -> !tosa.shape<3>
  // expected-error@+1 {{'tosa.tile' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.tile %arg0, %cst: (tensor<13x21x3xi1>, !tosa.shape<3>) -> tensor<39x21x6xi1>
  return %0 : tensor<39x21x6xi1>
}

// -----
func.func @test_transpose(%arg0: tensor<13x21x3xi1>) -> tensor<3x13x21xi1> {
  // expected-error@+1 {{'tosa.transpose' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %1 = tosa.transpose %arg0 {perms = array<i32: 2, 0, 1>} : (tensor<13x21x3xi1>) -> tensor<3x13x21xi1>
  return %1 : tensor<3x13x21xi1>
}
// -----
func.func @test_cast_i32_f32(%arg0: tensor<13x21x3xi32>) -> tensor<13x21x3xf32> {
  // expected-error@+1 {{'tosa.cast' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.cast %arg0 : (tensor<13x21x3xi32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
func.func @test_custom(%arg0: tensor<10xi32>) -> tensor<10xi32> {
  // expected-error@+1 {{'tosa.custom' op illegal: requires any of [pro_int, pro_fp] but not enabled in target}}
  %0 = tosa.custom %arg0 {operator_name="custom_test", domain_name="tosa.mlir_test", implementation_attrs="" } : (tensor<10xi32>) -> (tensor<10xi32>)
  return %0 : tensor<10xi32>
}

