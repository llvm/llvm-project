//--------------------------------------------------------------------------------------------------
// Enable all supported profiles to focus the verification of expected level errors.
//--------------------------------------------------------------------------------------------------

// RUN: mlir-opt %s -split-input-file -verify-diagnostics --tosa-validate="profile=bi,mi,mt"


func.func @test_argmax(%arg0: tensor<1x1x1x1x29x29x4xf32>) -> tensor<1x1x1x1x29x4xi32> {
  // expected-error@+1 {{'tosa.argmax' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = "tosa.argmax"(%arg0) {axis = 4 : i32} : (tensor<1x1x1x1x29x29x4xf32>) -> tensor<1x1x1x1x29x4xi32>
  return %0 : tensor<1x1x1x1x29x4xi32>
}

// -----

func.func @test_reduce_all(%arg0: tensor<1x1x1x1x13x21x3xi1>) -> tensor<1x1x1x1x1x21x3xi1> {
  // expected-error@+1 {{'tosa.reduce_all' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = "tosa.reduce_all"(%arg0) {axis = 4 : i32} : (tensor<1x1x1x1x13x21x3xi1>) -> tensor<1x1x1x1x1x21x3xi1>
  return %0 : tensor<1x1x1x1x1x21x3xi1>
}

// -----

func.func @test_reduce_any(%arg0: tensor<1x1x1x1x13x21x3xi1>) -> tensor<1x1x1x1x13x21x3xi1> {
  // expected-error@+1 {{'tosa.reduce_any' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = "tosa.reduce_any"(%arg0) {axis = 0 : i32} : (tensor<1x1x1x1x13x21x3xi1>) -> tensor<1x1x1x1x13x21x3xi1>
  return %0 : tensor<1x1x1x1x13x21x3xi1>
}

// -----

func.func @test_reduce_max(%arg0: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.reduce_max' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = "tosa.reduce_max"(%arg0) {axis = 0 : i32} : (tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_reduce_min(%arg0: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.reduce_min' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = "tosa.reduce_min"(%arg0) {axis = 0 : i32} : (tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_reduce_prod(%arg0: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.reduce_prod' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = "tosa.reduce_prod"(%arg0) {axis = 0 : i32} : (tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_reduce_sum(%arg0: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.reduce_sum' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = "tosa.reduce_sum"(%arg0) {axis = 0 : i32} : (tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_concat(%arg0: tensor<1x1x1x13x21x3x8xf32>, %arg1: tensor<1x1x1x13x21x3x8xf32>) -> tensor<1x1x1x26x21x3x8xf32> {
  // expected-error@+1 {{'tosa.concat' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = "tosa.concat"(%arg0, %arg1) {axis = 3 : i32} : (tensor<1x1x1x13x21x3x8xf32>, tensor<1x1x1x13x21x3x8xf32>) -> tensor<1x1x1x26x21x3x8xf32>
  return %0 : tensor<1x1x1x26x21x3x8xf32>
}

// -----

func.func @test_reshape(%arg0: tensor<13x21x3xf32>) -> tensor<1x1x1x1x1x1x819xf32> {
  %1 = tosa.const_shape {value = dense<[1, 1, 1, 1, 1, 1, 819]> : tensor<7xindex>} : () -> !tosa.shape<7>
  // expected-error@+1 {{'tosa.reshape' op failed level check: result rank(shape) <= MAX_RANK}}
  %0 = "tosa.reshape"(%arg0, %1) : (tensor<13x21x3xf32>, !tosa.shape<7>) -> tensor<1x1x1x1x1x1x819xf32>
  return %0 : tensor<1x1x1x1x1x1x819xf32>
}

// -----

func.func @test_reverse(%arg0: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.reverse' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = "tosa.reverse"(%arg0) {axis = 0 : i32} : (tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----
// CHECK-LABEL: slice
func.func @test_slice(%arg0: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x4x11x1xf32> {
  %0 = tosa.const_shape {value = dense<[0, 0, 0, 0, 6, 8, 0]> : tensor<7xindex>} : () -> !tosa.shape<7>
  %1 = tosa.const_shape {value = dense<[1, 1, 1, 1, 4, 11, 1]> : tensor<7xindex>} : () -> !tosa.shape<7>
  // expected-error@+1 {{'tosa.slice' op failed level check: operand rank(shape) <= MAX_RANK}}
  %2= tosa.slice %arg0, %0, %1 : (tensor<1x1x1x1x13x21x3xf32>, !tosa.shape<7>, !tosa.shape<7>) -> tensor<1x1x1x1x4x11x1xf32>
  return %2 : tensor<1x1x1x1x4x11x1xf32>
}

// -----
// CHECK-LABEL: tile
func.func @test_tile(%arg0: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x39x21x6xf32> {
  %cst = tosa.const_shape { value = dense<[1, 1, 1, 1, 3, 1, 2]> : tensor<7xindex> } : () -> !tosa.shape<7>
  // expected-error@+1 {{'tosa.tile' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.tile %arg0, %cst : (tensor<1x1x1x1x13x21x3xf32>, !tosa.shape<7>) -> tensor<1x1x1x1x39x21x6xf32>
  return %0 : tensor<1x1x1x1x39x21x6xf32>
}

// -----

func.func @test_transpose(%arg0: tensor<13x21x3x1x1x1x1xf32>) -> tensor<3x13x21x1x1x1x1xf32> {
  %0 = "tosa.const"() {value = dense<[2, 0, 1, 3, 4, 5, 6]> : tensor<7xi32>} : () -> tensor<7xi32>
  // expected-error@+1 {{'tosa.transpose' op failed level check: operand rank(shape) <= MAX_RANK}}
  %1 = "tosa.transpose"(%arg0, %0) : (tensor<13x21x3x1x1x1x1xf32>, tensor<7xi32>) -> tensor<3x13x21x1x1x1x1xf32>
  return %1 : tensor<3x13x21x1x1x1x1xf32>
}

// -----

func.func @test_const(%arg0 : tensor<1x1xi32>) -> tensor<1x1x1x1x1x1x1xi32> {
  // expected-error@+1 {{'tosa.const' op failed level check: result rank(shape) <= MAX_RA}}
  %0 = "tosa.const"() {value = dense<0> : tensor<1x1x1x1x1x1x1xi32>} : () -> tensor<1x1x1x1x1x1x1xi32>
  return %0: tensor<1x1x1x1x1x1x1xi32>
}

// -----

func.func @test_const_i2(%arg0 : tensor<1xi2>) {
  // expected-error@+1 {{'tosa.const' op is not profile-aligned: element type 'i2' is not legal}}
  %0 = "tosa.const"() {value = dense<0> : tensor<1xi2>} : () -> tensor<1xi2>
  return
}

// -----

func.func @test_const_ui32(%arg0 : tensor<1xui32>) {
  // expected-error@+1 {{'tosa.const' op is not profile-aligned: element type 'ui32' is not legal}}
  %0 = "tosa.const"() {value = dense<0> : tensor<1xui32>} : () -> tensor<1xui32>
  return
}

// -----

func.func @test_const_f64(%arg0 : tensor<1xf64>) {
  // expected-error@+1 {{'tosa.const' op is not profile-aligned: element type 'f64' is not legal}}
  %0 = "tosa.const"() {value = dense<0.0> : tensor<1xf64>} : () -> tensor<1xf64>
  return
}

// -----

func.func @test_const_ui8(%arg0 : tensor<1xui8>) {
  // expected-error@+1 {{'tosa.const' op is not profile-aligned: element type 'ui8' is not legal}}
  %0 = "tosa.const"() {value = dense<0> : tensor<1xui8>} : () -> tensor<1xui8>
  return
}

// -----

func.func @test_avgpool2d_kernel_y(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  // expected-error@+1 {{'tosa.avg_pool2d' op failed level check: kernel <= MAX_KERNEL}}
  %0 = "tosa.avg_pool2d"(%arg0) {kernel = array<i64: 8193, 1>, pad = array<i64: 4, 4, 4, 4>, stride = array<i64: 1, 1>, acc_type = f32} :
         (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----

func.func @test_avgpool2d_kernel_x(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  // expected-error@+1 {{'tosa.avg_pool2d' op failed level check: kernel <= MAX_KERNEL}}
  %0 = "tosa.avg_pool2d"(%arg0) {kernel = array<i64: 1, 8193>, pad = array<i64: 4, 4, 4, 4>, stride = array<i64: 1, 1>, acc_type = f32} :
         (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----

func.func @test_avgpool2d_stride_y(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  // expected-error@+1 {{'tosa.avg_pool2d' op failed level check: stride <= MAX_STRIDE}}
  %0 = "tosa.avg_pool2d"(%arg0) {kernel = array<i64: 1, 1>, pad = array<i64: 4, 4, 4, 4>, stride = array<i64: 8193, 1>, acc_type = f32} :
         (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----

func.func @test_avgpool2d_stride_x(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  // expected-error@+1 {{'tosa.avg_pool2d' op failed level check: stride <= MAX_STRIDE}}
  %0 = "tosa.avg_pool2d"(%arg0) {kernel = array<i64: 1, 1>, pad = array<i64: 4, 4, 4, 4>, stride = array<i64: 1, 8193>, acc_type = f32} :
         (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}


// -----

func.func @test_avgpool2d_pad_top(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  // expected-error@+1 {{'tosa.avg_pool2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = "tosa.avg_pool2d"(%arg0) {kernel = array<i64: 1, 1>, pad = array<i64: 8193, 4, 4, 4>, stride = array<i64: 1, 1>, acc_type = f32} :
         (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----

func.func @test_avgpool2d_pad_bottom(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  // expected-error@+1 {{'tosa.avg_pool2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = "tosa.avg_pool2d"(%arg0) {kernel = array<i64: 1, 1>, pad = array<i64: 4, 8193, 4, 4>, stride = array<i64: 1, 1>, acc_type = f32} :
         (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----

func.func @test_avgpool2d_pad_left(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  // expected-error@+1 {{'tosa.avg_pool2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = "tosa.avg_pool2d"(%arg0) {kernel = array<i64: 1, 1>, pad = array<i64: 4, 4, 8193, 4>, stride = array<i64: 1, 1>, acc_type = f32} :
         (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----

func.func @test_avgpool2d_pad_right(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  // expected-error@+1 {{'tosa.avg_pool2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = "tosa.avg_pool2d"(%arg0) {kernel = array<i64: 1, 1>, pad = array<i64: 4, 4, 4, 8193>, stride = array<i64: 1, 1>, acc_type = f32} :
         (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----

func.func @test_conv2d_dilation_y(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x2x2x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv2d' op failed level check: dilation_y * KH <= MAX_KERNEL}}
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {acc_type = f32, dilation = array<i64: 4097, 1>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} :
            (tensor<1x32x32x8xf32>, tensor<16x2x2x8xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

func.func @test_conv2d_dilation_x(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x2x2x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv2d' op failed level check: dilation_x * KW <= MAX_KERNEL}}
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {acc_type = f32, dilation = array<i64: 1, 4097>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} :
            (tensor<1x32x32x8xf32>, tensor<16x2x2x8xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

func.func @test_conv2d_pad_top(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x2x2x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 8193, 1, 0, 1>, stride = array<i64: 1, 1>} :
            (tensor<1x32x32x8xf32>, tensor<16x2x2x8xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

func.func @test_conv2d_pad_bottom(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x2x2x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 8193, 0, 1>, stride = array<i64: 1, 1>} :
            (tensor<1x32x32x8xf32>, tensor<16x2x2x8xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

func.func @test_conv2d_pad_left(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x2x2x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 8193, 1>, stride = array<i64: 1, 1>} :
            (tensor<1x32x32x8xf32>, tensor<16x2x2x8xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

func.func @test_conv2d_pad_right(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x2x2x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 8193>, stride = array<i64: 1, 1>} :
            (tensor<1x32x32x8xf32>, tensor<16x2x2x8xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

func.func @test_conv2d_stride_y(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x2x2x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv2d' op failed level check: stride <= MAX_STRIDE}}
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 8193, 1>} :
            (tensor<1x32x32x8xf32>, tensor<16x2x2x8xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

func.func @test_conv2d_stride_x(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x2x2x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv2d' op failed level check: stride <= MAX_STRIDE}}
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 8193>} :
            (tensor<1x32x32x8xf32>, tensor<16x2x2x8xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

func.func @test_conv3d_dilation_d(%arg0: tensor<1x1x32x32x8xf32>, %arg1: tensor<16x2x2x2x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv3d' op failed level check: dilation_d * KD <= MAX_KERNEL}}
  %0 = "tosa.conv3d"(%arg0, %arg1, %arg2) {acc_type = f32, dilation = array<i64: 4097, 1, 1>, pad = array<i64: 0, 1, 0, 1, 0, 1>, stride = array<i64: 1, 1, 1>} :
            (tensor<1x1x32x32x8xf32>, tensor<16x2x2x2x8xf32>, tensor<16xf32>) -> tensor<1x1x32x32x16xf32>
  return %0 : tensor<1x1x32x32x16xf32>
}

// -----

func.func @test_conv3d_dilation_y(%arg0: tensor<1x1x32x32x8xf32>, %arg1: tensor<16x2x2x2x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv3d' op failed level check: dilation_y * KH <= MAX_KERNEL}}
  %0 = "tosa.conv3d"(%arg0, %arg1, %arg2) {acc_type = f32, dilation = array<i64: 1, 4097, 1>, pad = array<i64: 0, 1, 0, 1, 0, 1>, stride = array<i64: 1, 1, 1>} :
            (tensor<1x1x32x32x8xf32>, tensor<16x2x2x2x8xf32>, tensor<16xf32>) -> tensor<1x1x32x32x16xf32>
  return %0 : tensor<1x1x32x32x16xf32>
}

// -----

func.func @test_conv3d_dilation_x(%arg0: tensor<1x1x32x32x8xf32>, %arg1: tensor<16x2x2x2x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv3d' op failed level check: dilation_x * KW <= MAX_KERNEL}}
  %0 = "tosa.conv3d"(%arg0, %arg1, %arg2) {acc_type = f32, dilation = array<i64: 1, 1, 4097>, pad = array<i64: 0, 1, 0, 1, 0, 1>, stride = array<i64: 1, 1, 1>} :
            (tensor<1x1x32x32x8xf32>, tensor<16x2x2x2x8xf32>, tensor<16xf32>) -> tensor<1x1x32x32x16xf32>
  return %0 : tensor<1x1x32x32x16xf32>
}

// -----

func.func @test_conv3d_pad_d0(%arg0: tensor<1x1x32x32x8xf32>, %arg1: tensor<16x2x2x2x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv3d' op failed level check: pad <= MAX_KERNEL}}
  %0 = "tosa.conv3d"(%arg0, %arg1, %arg2) {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 8193, 1, 0, 1, 0, 1>, stride = array<i64: 1, 1, 1>} :
            (tensor<1x1x32x32x8xf32>, tensor<16x2x2x2x8xf32>, tensor<16xf32>) -> tensor<1x1x32x32x16xf32>
  return %0 : tensor<1x1x32x32x16xf32>
}

// -----

func.func @test_conv3d_pad_d1(%arg0: tensor<1x1x32x32x8xf32>, %arg1: tensor<16x2x2x2x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv3d' op failed level check: pad <= MAX_KERNEL}}
  %0 = "tosa.conv3d"(%arg0, %arg1, %arg2) {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 1, 8193, 0, 1, 0, 1>, stride = array<i64: 1, 1, 1>} :
            (tensor<1x1x32x32x8xf32>, tensor<16x2x2x2x8xf32>, tensor<16xf32>) -> tensor<1x1x32x32x16xf32>
  return %0 : tensor<1x1x32x32x16xf32>
}

// -----

func.func @test_conv3d_pad_top(%arg0: tensor<1x1x32x32x8xf32>, %arg1: tensor<16x2x2x2x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv3d' op failed level check: pad <= MAX_KERNEL}}
  %0 = "tosa.conv3d"(%arg0, %arg1, %arg2) {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 1, 8193, 1, 0, 1>, stride = array<i64: 1, 1, 1>} :
            (tensor<1x1x32x32x8xf32>, tensor<16x2x2x2x8xf32>, tensor<16xf32>) -> tensor<1x1x32x32x16xf32>
  return %0 : tensor<1x1x32x32x16xf32>
}

// -----

func.func @test_conv3d_pad_bottom(%arg0: tensor<1x1x32x32x8xf32>, %arg1: tensor<16x2x2x2x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv3d' op failed level check: pad <= MAX_KERNEL}}
  %0 = "tosa.conv3d"(%arg0, %arg1, %arg2) {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 1, 0, 8193, 0, 1>, stride = array<i64: 1, 1, 1>} :
            (tensor<1x1x32x32x8xf32>, tensor<16x2x2x2x8xf32>, tensor<16xf32>) -> tensor<1x1x32x32x16xf32>
  return %0 : tensor<1x1x32x32x16xf32>
}

// -----

func.func @test_conv3d_pad_left(%arg0: tensor<1x1x32x32x8xf32>, %arg1: tensor<16x2x2x2x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv3d' op failed level check: pad <= MAX_KERNEL}}
  %0 = "tosa.conv3d"(%arg0, %arg1, %arg2) {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 1, 0, 1, 8193, 1>, stride = array<i64: 1, 1, 1>} :
            (tensor<1x1x32x32x8xf32>, tensor<16x2x2x2x8xf32>, tensor<16xf32>) -> tensor<1x1x32x32x16xf32>
  return %0 : tensor<1x1x32x32x16xf32>
}

// -----

func.func @test_conv3d_pad_right(%arg0: tensor<1x1x32x32x8xf32>, %arg1: tensor<16x2x2x2x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv3d' op failed level check: pad <= MAX_KERNEL}}
  %0 = "tosa.conv3d"(%arg0, %arg1, %arg2) {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 1, 0, 1, 0, 8193>, stride = array<i64: 1, 1, 1>} :
            (tensor<1x1x32x32x8xf32>, tensor<16x2x2x2x8xf32>, tensor<16xf32>) -> tensor<1x1x32x32x16xf32>
  return %0 : tensor<1x1x32x32x16xf32>
}

// -----

func.func @test_conv3d_stride_d(%arg0: tensor<1x1x32x32x8xf32>, %arg1: tensor<16x2x2x2x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv3d' op failed level check: stride <= MAX_STRIDE}}
  %0 = "tosa.conv3d"(%arg0, %arg1, %arg2) {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 1, 0, 1, 0, 1>, stride = array<i64: 8193, 1, 1>} :
            (tensor<1x1x32x32x8xf32>, tensor<16x2x2x2x8xf32>, tensor<16xf32>) -> tensor<1x1x32x32x16xf32>
  return %0 : tensor<1x1x32x32x16xf32>
}

// -----

func.func @test_conv3d_stride_y(%arg0: tensor<1x1x32x32x8xf32>, %arg1: tensor<16x2x2x2x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv3d' op failed level check: stride <= MAX_STRIDE}}
  %0 = "tosa.conv3d"(%arg0, %arg1, %arg2) {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 1, 0, 1, 0, 1>, stride = array<i64: 1, 8193, 1>} :
            (tensor<1x1x32x32x8xf32>, tensor<16x2x2x2x8xf32>, tensor<16xf32>) -> tensor<1x1x32x32x16xf32>
  return %0 : tensor<1x1x32x32x16xf32>
}

// -----

func.func @test_conv3d_stride_x(%arg0: tensor<1x1x32x32x8xf32>, %arg1: tensor<16x2x2x2x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv3d' op failed level check: stride <= MAX_STRIDE}}
  %0 = "tosa.conv3d"(%arg0, %arg1, %arg2) {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 1, 0, 1, 0, 1>, stride = array<i64: 1, 1, 8193>} :
            (tensor<1x1x32x32x8xf32>, tensor<16x2x2x2x8xf32>, tensor<16xf32>) -> tensor<1x1x32x32x16xf32>
  return %0 : tensor<1x1x32x32x16xf32>
}

// -----

func.func @test_depthwise_conv2d_dilation_y(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<2x2x8x8xf32>, %arg2: tensor<64xf32>) -> tensor<1x32x32x64xf32> {
  // expected-error@+1 {{'tosa.depthwise_conv2d' op failed level check: dilation_y * KH <= MAX_KERNEL}}
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {acc_type = f32, dilation = array<i64: 4097, 1>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} :
            (tensor<1x32x32x8xf32>, tensor<2x2x8x8xf32>, tensor<64xf32>) -> tensor<1x32x32x64xf32>
  return %0 : tensor<1x32x32x64xf32>
}

// -----

func.func @test_depthwise_conv2d_dilation_x(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<2x2x8x8xf32>, %arg2: tensor<64xf32>) -> tensor<1x32x32x64xf32> {
  // expected-error@+1 {{'tosa.depthwise_conv2d' op failed level check: dilation_x * KW <= MAX_KERNEL}}
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {acc_type = f32, dilation = array<i64: 1, 4097>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} :
            (tensor<1x32x32x8xf32>, tensor<2x2x8x8xf32>, tensor<64xf32>) -> tensor<1x32x32x64xf32>
  return %0 : tensor<1x32x32x64xf32>
}

// -----

func.func @test_depthwise_conv2d_pad_top(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<2x2x8x8xf32>, %arg2: tensor<64xf32>) -> tensor<1x32x32x64xf32> {
  // expected-error@+1 {{'tosa.depthwise_conv2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 8193, 1, 0, 1>, stride = array<i64: 1, 1>} :
            (tensor<1x32x32x8xf32>, tensor<2x2x8x8xf32>, tensor<64xf32>) -> tensor<1x32x32x64xf32>
  return %0 : tensor<1x32x32x64xf32>
}

// -----

func.func @test_depthwise_conv2d_pad_bottom(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<2x2x8x8xf32>, %arg2: tensor<64xf32>) -> tensor<1x32x32x64xf32> {
  // expected-error@+1 {{'tosa.depthwise_conv2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 8193, 0, 1>, stride = array<i64: 1, 1>} :
            (tensor<1x32x32x8xf32>, tensor<2x2x8x8xf32>, tensor<64xf32>) -> tensor<1x32x32x64xf32>
  return %0 : tensor<1x32x32x64xf32>
}

// -----

func.func @test_depthwise_conv2d_pad_left(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<2x2x8x8xf32>, %arg2: tensor<64xf32>) -> tensor<1x32x32x64xf32> {
  // expected-error@+1 {{'tosa.depthwise_conv2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 8193, 1>, stride = array<i64: 1, 1>} :
            (tensor<1x32x32x8xf32>, tensor<2x2x8x8xf32>, tensor<64xf32>) -> tensor<1x32x32x64xf32>
  return %0 : tensor<1x32x32x64xf32>
}

// -----

func.func @test_depthwise_conv2d_pad_right(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<2x2x8x8xf32>, %arg2: tensor<64xf32>) -> tensor<1x32x32x64xf32> {
  // expected-error@+1 {{'tosa.depthwise_conv2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 8193>, stride = array<i64: 1, 1>} :
            (tensor<1x32x32x8xf32>, tensor<2x2x8x8xf32>, tensor<64xf32>) -> tensor<1x32x32x64xf32>
  return %0 : tensor<1x32x32x64xf32>
}

// -----

func.func @test_depthwise_conv2d_stride_y(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<2x2x8x8xf32>, %arg2: tensor<64xf32>) -> tensor<1x32x32x64xf32> {
  // expected-error@+1 {{'tosa.depthwise_conv2d' op failed level check: stride <= MAX_STRIDE}}
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 8193, 1>} :
            (tensor<1x32x32x8xf32>, tensor<2x2x8x8xf32>, tensor<64xf32>) -> tensor<1x32x32x64xf32>
  return %0 : tensor<1x32x32x64xf32>
}

// -----

func.func @test_depthwise_conv2d_stride_x(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<2x2x8x8xf32>, %arg2: tensor<64xf32>) -> tensor<1x32x32x64xf32> {
  // expected-error@+1 {{'tosa.depthwise_conv2d' op failed level check: stride <= MAX_STRIDE}}
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 8193>} :
            (tensor<1x32x32x8xf32>, tensor<2x2x8x8xf32>, tensor<64xf32>) -> tensor<1x32x32x64xf32>
  return %0 : tensor<1x32x32x64xf32>
}

// -----

func.func @test_fft2d_real_h(%arg0: tensor<32x8193x32xf32>, %arg1: tensor<32x32x32xf32>) -> (tensor<32x32x32xf32>, tensor<32x32x32xf32>) {
  // expected-error@+1 {{'tosa.fft2d' op failed level check: H <= MAX_KERNEL}}
  %0, %1 = "tosa.fft2d"(%arg0, %arg1) { inverse = false } :
            (tensor<32x8193x32xf32>, tensor<32x32x32xf32>) -> (tensor<32x32x32xf32>, tensor<32x32x32xf32>)
  return %0, %1 : tensor<32x32x32xf32>, tensor<32x32x32xf32>
}

// -----

func.func @test_fft2d_real_w(%arg0: tensor<32x32x8193xf32>, %arg1: tensor<32x32x32xf32>) -> (tensor<32x32x32xf32>, tensor<32x32x32xf32>) {
  // expected-error@+1 {{'tosa.fft2d' op failed level check: W <= MAX_KERNEL}}
  %0, %1 = "tosa.fft2d"(%arg0, %arg1) { inverse = false } :
            (tensor<32x32x8193xf32>, tensor<32x32x32xf32>) -> (tensor<32x32x32xf32>, tensor<32x32x32xf32>)
  return %0, %1 : tensor<32x32x32xf32>, tensor<32x32x32xf32>
}

// -----

func.func @test_fft2d_imag_h(%arg0: tensor<32x32x32xf32>, %arg1: tensor<32x8193x32xf32>) -> (tensor<32x32x32xf32>, tensor<32x32x32xf32>) {
  // expected-error@+1 {{'tosa.fft2d' op failed level check: H <= MAX_KERNEL}}
  %0, %1 = "tosa.fft2d"(%arg0, %arg1) { inverse = false } :
            (tensor<32x32x32xf32>, tensor<32x8193x32xf32>) -> (tensor<32x32x32xf32>, tensor<32x32x32xf32>)
  return %0, %1 : tensor<32x32x32xf32>, tensor<32x32x32xf32>
}

// -----

func.func @test_fft2d_imag_w(%arg0: tensor<32x32x32xf32>, %arg1: tensor<32x32x8193xf32>) -> (tensor<32x32x32xf32>, tensor<32x32x32xf32>) {
  // expected-error@+1 {{'tosa.fft2d' op failed level check: W <= MAX_KERNEL}}
  %0, %1 = "tosa.fft2d"(%arg0, %arg1) { inverse = false } :
            (tensor<32x32x32xf32>, tensor<32x32x8193xf32>) -> (tensor<32x32x32xf32>, tensor<32x32x32xf32>)
  return %0, %1 : tensor<32x32x32xf32>, tensor<32x32x32xf32>
}

// -----

func.func @test_maxpool2d_stride_y(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  // expected-error@+1 {{'tosa.max_pool2d' op failed level check: stride <= MAX_STRIDE}}
  %0 = "tosa.max_pool2d"(%arg0) {kernel = array<i64: 1, 1>, pad = array<i64: 4, 4, 4, 4>, stride = array<i64: 8193, 1>} :
         (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----

func.func @test_maxpool2d_kernel_x(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  // expected-error@+1 {{'tosa.max_pool2d' op failed level check: kernel <= MAX_KERNEL}}
  %0 = "tosa.max_pool2d"(%arg0) {kernel = array<i64: 1, 8193>, pad = array<i64: 4, 4, 4, 4>, stride = array<i64: 1, 1>} :
         (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----

func.func @test_maxpool2d_stride_y(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  // expected-error@+1 {{'tosa.max_pool2d' op failed level check: stride <= MAX_STRIDE}}
  %0 = "tosa.max_pool2d"(%arg0) {kernel = array<i64: 1, 1>, pad = array<i64: 4, 4, 4, 4>, stride = array<i64: 8193, 1>} :
         (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----

func.func @test_maxpool2d_stride_x(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  // expected-error@+1 {{'tosa.max_pool2d' op failed level check: stride <= MAX_STRIDE}}
  %0 = "tosa.max_pool2d"(%arg0) {kernel = array<i64: 1, 1>, pad = array<i64: 4, 4, 4, 4>, stride = array<i64: 1, 8193>} :
         (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}


// -----

func.func @test_maxpool2d_pad_top(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  // expected-error@+1 {{'tosa.max_pool2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = "tosa.max_pool2d"(%arg0) {kernel = array<i64: 1, 1>, pad = array<i64: 8193, 4, 4, 4>, stride = array<i64: 1, 1>} :
         (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----

func.func @test_maxpool2d_pad_bottom(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  // expected-error@+1 {{'tosa.max_pool2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = "tosa.max_pool2d"(%arg0) {kernel = array<i64: 1, 1>, pad = array<i64: 4, 8193, 4, 4>, stride = array<i64: 1, 1>} :
         (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----

func.func @test_maxpool2d_pad_left(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  // expected-error@+1 {{'tosa.max_pool2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = "tosa.max_pool2d"(%arg0) {kernel = array<i64: 1, 1>, pad = array<i64: 4, 4, 8193, 4>, stride = array<i64: 1, 1>} :
         (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----

func.func @test_maxpool2d_pad_right(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  // expected-error@+1 {{'tosa.max_pool2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = "tosa.max_pool2d"(%arg0) {kernel = array<i64: 1, 1>, pad = array<i64: 4, 4, 4, 8193>, stride = array<i64: 1, 1>} :
         (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----

func.func @test_rfft2d_input_h(%arg0: tensor<13x8193x16xf32>) -> (tensor<13x8x9xf32>, tensor<13x8x9xf32>) {
  // expected-error@+1 {{'tosa.rfft2d' op failed level check: H <= MAX_KERNEL}}
  %0, %1 = "tosa.rfft2d"(%arg0) {} : (tensor<13x8193x16xf32>) -> (tensor<13x8x9xf32>, tensor<13x8x9xf32>)
  return %0, %1 : tensor<13x8x9xf32>, tensor<13x8x9xf32>
}

// -----

func.func @test_rfft2d_input_w(%arg0: tensor<13x8x8193xf32>) -> (tensor<13x8x9xf32>, tensor<13x8x9xf32>) {
  // expected-error@+1 {{'tosa.rfft2d' op failed level check: W <= MAX_KERNEL}}
  %0, %1 = "tosa.rfft2d"(%arg0) {} : (tensor<13x8x8193xf32>) -> (tensor<13x8x9xf32>, tensor<13x8x9xf32>)
  return %0, %1 : tensor<13x8x9xf32>, tensor<13x8x9xf32>
}

// -----

func.func @test_transpose_conv2d_weight_h(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x8193x1x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.transpose_conv2d' op failed level check: KH <= MAX_KERNEL}}
  %0 = "tosa.transpose_conv2d"(%arg0, %arg1, %arg2) {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: 1, 32, 32, 16>, stride = array<i64: 1, 1>} :
              (tensor<1x32x32x8xf32>, tensor<16x8193x1x8xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

func.func @test_transpose_conv2d_weight_w(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x1x8193x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.transpose_conv2d' op failed level check: KW <= MAX_KERNEL}}
  %0 = "tosa.transpose_conv2d"(%arg0, %arg1, %arg2) {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: 1, 32, 32, 16>, stride = array<i64: 1, 1>} :
              (tensor<1x32x32x8xf32>, tensor<16x1x8193x8xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

func.func @test_transpose_conv2d_pad_top(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x1x1x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.transpose_conv2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = "tosa.transpose_conv2d"(%arg0, %arg1, %arg2) {acc_type = f32, out_pad = array<i64: 8193, 0, 0, 0>, out_shape = array<i64: 1, 32, 32, 16>, stride = array<i64: 1, 1>} :
              (tensor<1x32x32x8xf32>, tensor<16x1x1x8xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

func.func @test_transpose_conv2d_pad_bottom(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x1x1x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.transpose_conv2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = "tosa.transpose_conv2d"(%arg0, %arg1, %arg2) {acc_type = f32, out_pad = array<i64: 0, 8193, 0, 0>, out_shape = array<i64: 1, 32, 32, 16>, stride = array<i64: 1, 1>} :
              (tensor<1x32x32x8xf32>, tensor<16x1x1x8xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

func.func @test_transpose_conv2d_pad_left(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x1x1x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.transpose_conv2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = "tosa.transpose_conv2d"(%arg0, %arg1, %arg2) {acc_type = f32, out_pad = array<i64: 0, 0, 8193, 0>, out_shape = array<i64: 1, 32, 32, 16>, stride = array<i64: 1, 1>} :
              (tensor<1x32x32x8xf32>, tensor<16x1x1x8xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

func.func @test_transpose_conv2d_pad_right(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x1x1x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.transpose_conv2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = "tosa.transpose_conv2d"(%arg0, %arg1, %arg2) {acc_type = f32, out_pad = array<i64: 0, 0, 0, 8193>, out_shape = array<i64: 1, 32, 32, 16>, stride = array<i64: 1, 1>} :
              (tensor<1x32x32x8xf32>, tensor<16x1x1x8xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

func.func @test_transpose_conv2d_stride_y(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x1x1x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.transpose_conv2d' op failed level check: stride <= MAX_STRIDE}}
  %0 = "tosa.transpose_conv2d"(%arg0, %arg1, %arg2) {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: 1, 32, 32, 16>, stride = array<i64: 8193, 1>} :
              (tensor<1x32x32x8xf32>, tensor<16x1x1x8xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

func.func @test_transpose_conv2d_stride_x(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x1x1x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.transpose_conv2d' op failed level check: stride <= MAX_STRIDE}}
  %0 = "tosa.transpose_conv2d"(%arg0, %arg1, %arg2) {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: 1, 32, 32, 16>, stride = array<i64: 1, 8193>} :
              (tensor<1x32x32x8xf32>, tensor<16x1x1x8xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

func.func @test_resize_scale_y(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x64x64x8xf32> {
  // expected-error@+1 {{'tosa.resize' op failed level check: scale_y_n/scale_y_d <= MAX_SCALE}}
  %1 = "tosa.resize"(%arg0) { scale = array<i64: 257, 1, 4, 2>, offset = array<i64: -1, -1>, border = array<i64: 1, 1>, mode = "BILINEAR"} :
                (tensor<1x32x32x8xf32>) -> tensor<1x64x64x8xf32>
  return %1 : tensor<1x64x64x8xf32>
}

// -----

func.func @test_resize_scale_x(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x64x64x8xf32> {
  // expected-error@+1 {{'tosa.resize' op failed level check: scale_x_n/scale_x_d <= MAX_SCALE}}
  %1 = "tosa.resize"(%arg0) { scale = array<i64: 4, 2, 257, 1>, offset = array<i64: -1, -1>, border = array<i64: 1, 1>, mode = "BILINEAR"} :
                (tensor<1x32x32x8xf32>) -> tensor<1x64x64x8xf32>
  return %1 : tensor<1x64x64x8xf32>
}

// -----

// CHECK-LABEL: @test_cond_if
func.func @test_cond_if(%arg0: tensor<1x1x1x1x1x1x1xf32>, %arg1: tensor<1x1x1x1x1x1x1xf32>, %arg2: tensor<i1>) -> tensor<1x1x1x1x1x1x1xf32> {
  %0 = "tosa.cond_if"(%arg2, %arg0, %arg1) ({
  ^bb0(%arg3: tensor<1x1x1x1x1x1x1xf32>, %arg4: tensor<1x1x1x1x1x1x1xf32>):
    "tosa.yield"(%arg3) : (tensor<1x1x1x1x1x1x1xf32>) -> ()
  },  {
  ^bb0(%arg3: tensor<1x1x1x1x1x1x1xf32>, %arg4: tensor<1x1x1x1x1x1x1xf32>):
    "tosa.yield"(%arg4) : (tensor<1x1x1x1x1x1x1xf32>) -> ()
  }) : (tensor<i1>, tensor<1x1x1x1x1x1x1xf32>, tensor<1x1x1x1x1x1x1xf32>) -> tensor<1x1x1x1x1x1x1xf32>
  return %0 : tensor<1x1x1x1x1x1x1xf32>
}

// -----

// CHECK-LABEL: @test_while_loop
func.func @test_while_loop(%arg0: tensor<1x1x1x1x1x1x1xf32>, %arg1: tensor<i32>) {
  %0 = "tosa.const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %1:2 = "tosa.while_loop"(%0, %arg0) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<1x1x1x1x1x1x1xf32>):
    %2 = "tosa.greater_equal"(%arg3, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = "tosa.logical_not"(%2) : (tensor<i1>) -> tensor<i1>
    "tosa.yield"(%3) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<1x1x1x1x1x1x1xf32>):
    %2 = "tosa.const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %3 = "tosa.add"(%arg3, %2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "tosa.yield"(%3, %arg4) : (tensor<i32>, tensor<1x1x1x1x1x1x1xf32>) -> ()
  }) : (tensor<i32>, tensor<1x1x1x1x1x1x1xf32>) -> (tensor<i32>, tensor<1x1x1x1x1x1x1xf32>)
  return
}

// -----

// CHECK-LABEL: @test_custom
func.func @test_custom(%arg0: tensor<1x1x1x1x1x1x10xi32>) -> tensor<1x1x1x1x1x1x10xi32> {
  %0 = "tosa.custom"(%arg0) {operator_name="custom_test", domain_name="tosa_mlir_test", implementation_attrs=""} :
           (tensor<1x1x1x1x1x1x10xi32>) -> (tensor<1x1x1x1x1x1x10xi32>)
  return %0 : tensor<1x1x1x1x1x1x10xi32>
}

// -----

// CHECK-LABEL: unranked_tensor
func.func @test_unranked_tensor(%arg0: tensor<*xf32>) {
  %0 = tosa.const_shape {value = dense<[0]> : tensor<1xindex>} : () -> !tosa.shape<1>
  %1 = tosa.const_shape {value = dense<[1]> : tensor<1xindex>} : () -> !tosa.shape<1>

  // expected-error@+1 {{'tosa.slice' op failed level check: unranked tensor}}
  %2= tosa.slice %arg0, %0, %1 : (tensor<*xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<*xf32>
  return
}
