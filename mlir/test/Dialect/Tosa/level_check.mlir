// RUN: mlir-opt %s -split-input-file -verify-diagnostics --tosa-validate

func.func @test_argmax_rank_invalid(%arg0: tensor<1x1x1x1x29x29x4xf32>) -> tensor<1x1x1x1x29x4xi32> {
  // expected-error@+1 {{'tosa.argmax' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = "tosa.argmax"(%arg0) {axis = 4 : i32} : (tensor<1x1x1x1x29x29x4xf32>) -> tensor<1x1x1x1x29x4xi32>
  return %0 : tensor<1x1x1x1x29x4xi32>
}

// -----

func.func @test_clamp_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.clamp' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.clamp %arg0 {min_val = -3.40282347E+38 : f32, max_val = 3.40282347E+38 : f32} : (tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_erf_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.erf' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.erf %arg0 : (tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_sigmoid_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.sigmoid' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.sigmoid %arg0 : (tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_tanh_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.tanh' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.tanh %arg0 : (tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_add_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xf32>, %arg1: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.add' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.add %arg0, %arg1 : (tensor<1x1x1x1x13x21x3xf32>, tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_arithmetic_right_shift_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xf32>, %arg1: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.arithmetic_right_shift' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = false} : (tensor<1x1x1x1x13x21x3xf32>, tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_bitwise_and_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xi32>, %arg1: tensor<1x1x1x1x13x21x3xi32>) -> tensor<1x1x1x1x13x21x3xi32> {
  // expected-error@+1 {{'tosa.bitwise_and' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<1x1x1x1x13x21x3xi32>, tensor<1x1x1x1x13x21x3xi32>) -> tensor<1x1x1x1x13x21x3xi32>
  return %0 : tensor<1x1x1x1x13x21x3xi32>
}

// -----

func.func @test_bitwise_or_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xi32>, %arg1: tensor<1x1x1x1x13x21x3xi32>) -> tensor<1x1x1x1x13x21x3xi32> {
  // expected-error@+1 {{'tosa.bitwise_or' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<1x1x1x1x13x21x3xi32>, tensor<1x1x1x1x13x21x3xi32>) -> tensor<1x1x1x1x13x21x3xi32>
  return %0 : tensor<1x1x1x1x13x21x3xi32>
}

// -----

func.func @test_bitwise_xor_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xi32>, %arg1: tensor<1x1x1x1x13x21x3xi32>) -> tensor<1x1x1x1x13x21x3xi32> {
  // expected-error@+1 {{'tosa.bitwise_xor' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<1x1x1x1x13x21x3xi32>, tensor<1x1x1x1x13x21x3xi32>) -> tensor<1x1x1x1x13x21x3xi32>
  return %0 : tensor<1x1x1x1x13x21x3xi32>
}

// -----

func.func @test_int_div_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xi32>, %arg1: tensor<1x1x1x1x13x21x3xi32>) -> tensor<1x1x1x1x13x21x3xi32> {
  // expected-error@+1 {{'tosa.int_div' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.int_div %arg0, %arg1 : (tensor<1x1x1x1x13x21x3xi32>, tensor<1x1x1x1x13x21x3xi32>) -> tensor<1x1x1x1x13x21x3xi32>
  return %0 : tensor<1x1x1x1x13x21x3xi32>
}

// -----

func.func @test_logical_and_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xi1>, %arg1: tensor<1x1x1x1x13x21x3xi1>) -> tensor<1x1x1x1x13x21x3xi1> {
  // expected-error@+1 {{'tosa.logical_and' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.logical_and %arg0, %arg1 : (tensor<1x1x1x1x13x21x3xi1>, tensor<1x1x1x1x13x21x3xi1>) -> tensor<1x1x1x1x13x21x3xi1>
  return %0 : tensor<1x1x1x1x13x21x3xi1>
}

// -----

func.func @test_logical_left_shift_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xi32>, %arg1: tensor<1x1x1x1x13x21x3xi32>) -> tensor<1x1x1x1x13x21x3xi32> {
  // expected-error@+1 {{'tosa.logical_left_shift' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<1x1x1x1x13x21x3xi32>, tensor<1x1x1x1x13x21x3xi32>) -> tensor<1x1x1x1x13x21x3xi32>
  return %0 : tensor<1x1x1x1x13x21x3xi32>
}

// -----

func.func @test_logical_right_shift_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xi32>, %arg1: tensor<1x1x1x1x13x21x3xi32>) -> tensor<1x1x1x1x13x21x3xi32> {
  // expected-error@+1 {{'tosa.logical_right_shift' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<1x1x1x1x13x21x3xi32>, tensor<1x1x1x1x13x21x3xi32>) -> tensor<1x1x1x1x13x21x3xi32>
  return %0 : tensor<1x1x1x1x13x21x3xi32>
}

// -----

func.func @test_logical_or_rank_invalid(%arg0: tensor<1x1x1x1x13x1x3xi1>, %arg1: tensor<1x1x1x1x13x21x3xi1>) -> tensor<1x1x1x1x13x21x3xi1> {
  // expected-error@+1 {{'tosa.logical_or' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.logical_or %arg0, %arg1 : (tensor<1x1x1x1x13x1x3xi1>, tensor<1x1x1x1x13x21x3xi1>) -> tensor<1x1x1x1x13x21x3xi1>
  return %0 : tensor<1x1x1x1x13x21x3xi1>
}

// -----

func.func @test_logical_xor_rank_invalid(%arg0: tensor<1x1x1x1x13x1x3xi1>, %arg1: tensor<1x1x1x1x13x21x3xi1>) -> tensor<1x1x1x1x13x21x3xi1> {
  // expected-error@+1 {{'tosa.logical_xor' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.logical_xor %arg0, %arg1 : (tensor<1x1x1x1x13x1x3xi1>, tensor<1x1x1x1x13x21x3xi1>) -> tensor<1x1x1x1x13x21x3xi1>
  return %0 : tensor<1x1x1x1x13x21x3xi1>
}

// -----

func.func @test_max_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xf32>, %arg1: tensor<1x1x1x1x13x21x1xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.maximum' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.maximum %arg0, %arg1 : (tensor<1x1x1x1x13x21x3xf32>, tensor<1x1x1x1x13x21x1xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_min_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xf32>, %arg1: tensor<1x1x1x1x1x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.minimum' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.minimum %arg0, %arg1 : (tensor<1x1x1x1x13x21x3xf32>, tensor<1x1x1x1x1x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_mul_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xf32>, %arg1: tensor<1x1x1x1x13x1x3xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  // expected-error@+1 {{'tosa.mul' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.mul %arg0, %arg1, %shift : (tensor<1x1x1x1x13x21x3xf32>, tensor<1x1x1x1x13x1x3xf32>, tensor<1xi8>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_pow_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xf32>, %arg1: tensor<1x1x1x1x13x21x1xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.pow' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.pow %arg0, %arg1 : (tensor<1x1x1x1x13x21x3xf32>, tensor<1x1x1x1x13x21x1xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_sub_rank_invalid(%arg0: tensor<1x1x1x1x1x21x3xf32>, %arg1: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.sub' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.sub %arg0, %arg1 : (tensor<1x1x1x1x1x21x3xf32>, tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_table_rank_invalid(%arg0: tensor<1x1x1x1x1x1x64xi32>, %arg1: tensor<513xi16>) -> tensor<1x1x1x1x1x1x64xi16> {
  // expected-error@+1 {{'tosa.table' op failed level check: operand rank(shape) <= MAX_RANK}}
    %0 = tosa.table %arg0, %arg1 : (tensor<1x1x1x1x1x1x64xi32>, tensor<513xi16>) -> tensor<1x1x1x1x1x1x64xi16>
    return %0 : tensor<1x1x1x1x1x1x64xi16>
}

// -----

func.func @test_abs_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.abs' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.abs %arg0 : (tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_bitwise_not_rank_invalid(%arg0: tensor<1x1x1x1x13x21x1xi32>) -> tensor<1x1x1x1x13x21x1xi32> {
  // expected-error@+1 {{'tosa.bitwise_not' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.bitwise_not %arg0 : (tensor<1x1x1x1x13x21x1xi32>) -> tensor<1x1x1x1x13x21x1xi32>
  return %0 : tensor<1x1x1x1x13x21x1xi32>
}

// -----

func.func @test_ceil_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.ceil' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.ceil %arg0 : (tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_clz_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xi32>) -> tensor<1x1x1x1x13x21x3xi32> {
  // expected-error@+1 {{'tosa.clz' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.clz %arg0 : (tensor<1x1x1x1x13x21x3xi32>) -> tensor<1x1x1x1x13x21x3xi32>
  return %0 : tensor<1x1x1x1x13x21x3xi32>
}

// -----

func.func @test_cos_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.cos' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.cos %arg0 : (tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_exp_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.exp' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.exp %arg0 : (tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_floor_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.floor' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.floor %arg0 : (tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_log_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.log' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.log %arg0 : (tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_logical_not_rank_invalid(%arg0: tensor<1x1x1x1x1x21x3xi1>) -> tensor<1x1x1x1x1x21x3xi1> {
  // expected-error@+1 {{'tosa.logical_not' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.logical_not %arg0 : (tensor<1x1x1x1x1x21x3xi1>) -> tensor<1x1x1x1x1x21x3xi1>
  return %0 : tensor<1x1x1x1x1x21x3xi1>
}

// -----

func.func @test_negate_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xf32>, %arg1: tensor<1xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.negate' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.negate %arg0, %arg1, %arg1 : (tensor<1x1x1x1x13x21x3xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_reciprocal_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.reciprocal' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.reciprocal %arg0 : (tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_rsqrt_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.rsqrt' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.rsqrt %arg0 : (tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_sin_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.sin' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.sin %arg0 : (tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_select_rank_invalid(%arg0: tensor<1x1x1x1x1x1x1xi1>, %arg1: tensor<1x1x1x1x13x21x3xf32>, %arg2: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.select' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.select %arg0, %arg1, %arg2 : (tensor<1x1x1x1x1x1x1xi1>, tensor<1x1x1x1x13x21x3xf32>, tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_equal_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xf32>, %arg1: tensor<1x1x1x1x13x1x3xf32>) -> tensor<1x1x1x1x13x21x3xi1> {
  // expected-error@+1 {{'tosa.equal' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.equal %arg0, %arg1 : (tensor<1x1x1x1x13x21x3xf32>, tensor<1x1x1x1x13x1x3xf32>) -> tensor<1x1x1x1x13x21x3xi1>
  return %0 : tensor<1x1x1x1x13x21x3xi1>
}

// -----

func.func @test_greater_rank_invalid(%arg0: tensor<1x1x1x1x13x21x1xf32>, %arg1: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xi1> {
  // expected-error@+1 {{'tosa.greater' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.greater %arg0, %arg1 : (tensor<1x1x1x1x13x21x1xf32>, tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xi1>
  return %0 : tensor<1x1x1x1x13x21x3xi1>
}

// -----

func.func @test_greater_equal_rank_invalid(%arg0: tensor<1x1x1x1x13x1x3xf32>, %arg1: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xi1> {
  // expected-error@+1 {{'tosa.greater_equal' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.greater_equal %arg0, %arg1 : (tensor<1x1x1x1x13x1x3xf32>, tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xi1>
  return %0 : tensor<1x1x1x1x13x21x3xi1>
}

// -----

func.func @test_reduce_all_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xi1>) -> tensor<1x1x1x1x1x21x3xi1> {
  // expected-error@+1 {{'tosa.reduce_all' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = "tosa.reduce_all"(%arg0) {axis = 4 : i32} : (tensor<1x1x1x1x13x21x3xi1>) -> tensor<1x1x1x1x1x21x3xi1>
  return %0 : tensor<1x1x1x1x1x21x3xi1>
}

// -----

func.func @test_reduce_any_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xi1>) -> tensor<1x1x1x1x13x21x3xi1> {
  // expected-error@+1 {{'tosa.reduce_any' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = "tosa.reduce_any"(%arg0) {axis = 0 : i32} : (tensor<1x1x1x1x13x21x3xi1>) -> tensor<1x1x1x1x13x21x3xi1>
  return %0 : tensor<1x1x1x1x13x21x3xi1>
}

// -----

func.func @test_reduce_max_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.reduce_max' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = "tosa.reduce_max"(%arg0) {axis = 0 : i32} : (tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_reduce_min_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.reduce_min' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = "tosa.reduce_min"(%arg0) {axis = 0 : i32} : (tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_reduce_prod_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.reduce_product' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = "tosa.reduce_product"(%arg0) {axis = 0 : i32} : (tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_reduce_sum_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.reduce_sum' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = "tosa.reduce_sum"(%arg0) {axis = 0 : i32} : (tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_concat_rank_invalid(%arg0: tensor<1x1x1x13x21x3x8xf32>, %arg1: tensor<1x1x1x13x21x3x8xf32>) -> tensor<1x1x1x26x21x3x8xf32> {
  // expected-error@+1 {{'tosa.concat' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = "tosa.concat"(%arg0, %arg1) {axis = 3 : i32} : (tensor<1x1x1x13x21x3x8xf32>, tensor<1x1x1x13x21x3x8xf32>) -> tensor<1x1x1x26x21x3x8xf32>
  return %0 : tensor<1x1x1x26x21x3x8xf32>
}

// -----

func.func @test_pad_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  %pad_const = "tosa.const"() {values = dense<3.14> : tensor<1xf32>} : () -> tensor<1xf32>
  %padding = tosa.const_shape {values = dense<0> : tensor<14xindex>} : () -> !tosa.shape<14>
  // expected-error@+1 {{'tosa.pad' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.pad %arg0, %padding, %pad_const : (tensor<1x1x1x1x13x21x3xf32>, !tosa.shape<14>, tensor<1xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----

func.func @test_reshape_rank_invalid(%arg0: tensor<13x21x3xf32>) -> tensor<1x1x1x1x1x1x819xf32> {
  %1 = tosa.const_shape {values = dense<[1, 1, 1, 1, 1, 1, 819]> : tensor<7xindex>} : () -> !tosa.shape<7>
  // expected-error@+1 {{'tosa.reshape' op failed level check: result rank(shape) <= MAX_RANK}}
  %0 = "tosa.reshape"(%arg0, %1) : (tensor<13x21x3xf32>, !tosa.shape<7>) -> tensor<1x1x1x1x1x1x819xf32>
  return %0 : tensor<1x1x1x1x1x1x819xf32>
}

// -----

func.func @test_reverse_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32> {
  // expected-error@+1 {{'tosa.reverse' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = "tosa.reverse"(%arg0) {axis = 0 : i32} : (tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x13x21x3xf32>
  return %0 : tensor<1x1x1x1x13x21x3xf32>
}

// -----
// CHECK-LABEL: slice
func.func @test_slice_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x4x11x1xf32> {
  %0 = tosa.const_shape {values = dense<[0, 0, 0, 0, 6, 8, 0]> : tensor<7xindex>} : () -> !tosa.shape<7>
  %1 = tosa.const_shape {values = dense<[1, 1, 1, 1, 4, 11, 1]> : tensor<7xindex>} : () -> !tosa.shape<7>
  // expected-error@+1 {{'tosa.slice' op failed level check: operand rank(shape) <= MAX_RANK}}
  %2= tosa.slice %arg0, %0, %1 : (tensor<1x1x1x1x13x21x3xf32>, !tosa.shape<7>, !tosa.shape<7>) -> tensor<1x1x1x1x4x11x1xf32>
  return %2 : tensor<1x1x1x1x4x11x1xf32>
}

// -----
// CHECK-LABEL: tile
func.func @test_tile_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xf32>) -> tensor<1x1x1x1x39x21x6xf32> {
  %cst = tosa.const_shape { values = dense<[1, 1, 1, 1, 3, 1, 2]> : tensor<7xindex> } : () -> !tosa.shape<7>
  // expected-error@+1 {{'tosa.tile' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.tile %arg0, %cst : (tensor<1x1x1x1x13x21x3xf32>, !tosa.shape<7>) -> tensor<1x1x1x1x39x21x6xf32>
  return %0 : tensor<1x1x1x1x39x21x6xf32>
}

// -----

func.func @test_transpose_rank_invalid(%arg0: tensor<13x21x3x1x1x1x1xf32>) -> tensor<3x13x21x1x1x1x1xf32> {
  // expected-error@+1 {{'tosa.transpose' op failed level check: operand rank(shape) <= MAX_RANK}}
  %1 = "tosa.transpose"(%arg0) {perms = array<i32: 2, 0, 1, 3, 4, 5, 6>} : (tensor<13x21x3x1x1x1x1xf32>) -> tensor<3x13x21x1x1x1x1xf32>
  return %1 : tensor<3x13x21x1x1x1x1xf32>
}

// -----

func.func @test_cast_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xi32>) -> tensor<1x1x1x1x13x21x3xi16> {
  // expected-error@+1 {{'tosa.cast' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.cast %arg0 : (tensor<1x1x1x1x13x21x3xi32>) -> tensor<1x1x1x1x13x21x3xi16>
  return %0 : tensor<1x1x1x1x13x21x3xi16>
}

// -----

func.func @test_rescale_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xi8>) -> tensor<1x1x1x1x13x21x3xi8> {
  %multiplier = "tosa.const"() {values = dense<1073741824> : tensor<1xi32>} : () -> tensor<1xi32>
  %shift = "tosa.const"() {values = dense<30> : tensor<1xi8>} : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<127> : tensor<1xi8>} : () -> tensor<1xi8>
  %output_zp = "tosa.const"() {values = dense<-1> : tensor<1xi8>} : () -> tensor<1xi8>
  // expected-error@+1 {{'tosa.rescale' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {rounding_mode = "SINGLE_ROUND", input_zp = 127 : i32, output_zp = -1 : i32, per_channel = false, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<1x1x1x1x13x21x3xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x1x1x1x13x21x3xi8>
  return %0 : tensor<1x1x1x1x13x21x3xi8>
}

// -----
func.func @test_const(%arg0 : tensor<1x1xi32>) -> tensor<1x1x1x1x1x1x1xi32> {
  // expected-error@+1 {{'tosa.const' op failed level check: attribute rank(shape) <= MAX_RANK}}
  %0 = "tosa.const"() {values = dense<0> : tensor<1x1x1x1x1x1x1xi32>} : () -> tensor<1x1x1x1x1x1x1xi32>
  return %0: tensor<1x1x1x1x1x1x1xi32>
}

// -----

func.func @test_add_rank_valid(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @test_const_rank_valid(%arg0 : tensor<i32>) -> tensor<i32> {
  %0 = "tosa.const"() {values = dense<0> : tensor<i32>} : () -> tensor<i32>
  return %0: tensor<i32>
}

// -----

func.func @test_const_i2(%arg0 : tensor<1xi2>) {
  // expected-error@+1 {{'tosa.const' op is not profile-aligned: element type 'i2' is not legal}}
  %0 = "tosa.const"() {values = dense<0> : tensor<1xi2>} : () -> tensor<1xi2>
  return
}

// -----

func.func @test_const_ui32(%arg0 : tensor<1xui32>) {
  // expected-error@+1 {{'tosa.const' op is not profile-aligned: element type 'ui32' is not legal}}
  %0 = "tosa.const"() {values = dense<0> : tensor<1xui32>} : () -> tensor<1xui32>
  return
}

// -----

func.func @test_const_f64(%arg0 : tensor<1xf64>) {
  // expected-error@+1 {{'tosa.const' op is not profile-aligned: element type 'f64' is not legal}}
  %0 = "tosa.const"() {values = dense<0.0> : tensor<1xf64>} : () -> tensor<1xf64>
  return
}

// -----

func.func @test_const_ui8(%arg0 : tensor<1xui8>) {
  // expected-error@+1 {{'tosa.const' op is not profile-aligned: element type 'ui8' is not legal}}
  %0 = "tosa.const"() {values = dense<0> : tensor<1xui8>} : () -> tensor<1xui8>
  return
}

// -----

func.func @test_identity_rank_invalid(%arg0: tensor<1x1x1x1x13x21x3xi32>) -> tensor<1x1x1x1x13x21x3xi32> {
  // expected-error@+1 {{'tosa.identity' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = tosa.identity %arg0 : (tensor<1x1x1x1x13x21x3xi32>) -> tensor<1x1x1x1x13x21x3xi32>
  return %0 : tensor<1x1x1x1x13x21x3xi32>
}

// -----

func.func @test_identity_rank_valid(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = tosa.identity %arg0 : (tensor<i32>) -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

func.func @test_avgpool2d_kernel_y(%arg0: tensor<1x8194x32x8xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> tensor<1x2x32x8xf32> {
  // expected-error@+1 {{'tosa.avg_pool2d' op failed level check: kernel <= MAX_KERNEL}}
  %0 = "tosa.avg_pool2d"(%arg0, %arg1, %arg2) {kernel = array<i64: 8193, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, acc_type = f32} :
         (tensor<1x8194x32x8xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x2x32x8xf32>
  return %0 : tensor<1x2x32x8xf32>
}

// -----

func.func @test_avgpool2d_kernel_x(%arg0: tensor<1x32x8194x8xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> tensor<1x32x2x8xf32> {
  // expected-error@+1 {{'tosa.avg_pool2d' op failed level check: kernel <= MAX_KERNEL}}
  %0 = "tosa.avg_pool2d"(%arg0, %arg1, %arg2) {kernel = array<i64: 1, 8193>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, acc_type = f32} :
         (tensor<1x32x8194x8xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x2x8xf32>
  return %0 : tensor<1x32x2x8xf32>
}

// -----

func.func @test_avgpool2d_stride_y(%arg0: tensor<1x8194x32x8xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> tensor<1x2x32x8xf32> {
  // expected-error@+1 {{'tosa.avg_pool2d' op failed level check: stride <= MAX_STRIDE}}
  %0 = "tosa.avg_pool2d"(%arg0, %arg1, %arg2) {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 8193, 1>, acc_type = f32} :
         (tensor<1x8194x32x8xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x2x32x8xf32>
  return %0 : tensor<1x2x32x8xf32>
}

// -----

func.func @test_avgpool2d_stride_x(%arg0: tensor<1x32x8194x8xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> tensor<1x32x2x8xf32> {
  // expected-error@+1 {{'tosa.avg_pool2d' op failed level check: stride <= MAX_STRIDE}}
  %0 = "tosa.avg_pool2d"(%arg0, %arg1, %arg2) {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 8193>, acc_type = f32} :
         (tensor<1x32x8194x8xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x2x8xf32>
  return %0 : tensor<1x32x2x8xf32>
}

// -----

func.func @test_conv2d_dilation_y(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x2x2x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>) -> tensor<*xf32> {
  // expected-error@+1 {{'tosa.conv2d' op failed level check: dilation_y * KH <= MAX_KERNEL}}
  %0 = tosa.conv2d %arg0, %arg1, %arg2, %arg3, %arg3 {acc_type = f32, dilation = array<i64: 4097, 1>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} :
            (tensor<1x32x32x8xf32>, tensor<16x2x2x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

func.func @test_conv2d_dilation_x(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x2x2x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>) -> tensor<*xf32> {
  // expected-error@+1 {{'tosa.conv2d' op failed level check: dilation_x * KW <= MAX_KERNEL}}
  %0 = tosa.conv2d %arg0, %arg1, %arg2, %arg3, %arg3 {acc_type = f32, dilation = array<i64: 1, 4097>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} :
            (tensor<1x32x32x8xf32>, tensor<16x2x2x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

func.func @test_conv2d_pad_top(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x2x2x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>) -> tensor<1x8225x32x16xf32> {
  // expected-error@+1 {{'tosa.conv2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = tosa.conv2d %arg0, %arg1, %arg2, %arg3, %arg3 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 8193, 1, 0, 1>, stride = array<i64: 1, 1>} :
            (tensor<1x32x32x8xf32>, tensor<16x2x2x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x8225x32x16xf32>
  return %0 : tensor<1x8225x32x16xf32>
}

// -----

func.func @test_conv2d_pad_bottom(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x2x2x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>) -> tensor<1x8224x32x16xf32> {
  // expected-error@+1 {{'tosa.conv2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = tosa.conv2d %arg0, %arg1, %arg2, %arg3, %arg3 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 8193, 0, 1>, stride = array<i64: 1, 1>} :
            (tensor<1x32x32x8xf32>, tensor<16x2x2x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x8224x32x16xf32>
  return %0 : tensor<1x8224x32x16xf32>
}

// -----

func.func @test_conv2d_pad_left(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x2x2x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>) -> tensor<1x32x8225x16xf32> {
  // expected-error@+1 {{'tosa.conv2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = tosa.conv2d %arg0, %arg1, %arg2, %arg3, %arg3 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 8193, 1>, stride = array<i64: 1, 1>} :
            (tensor<1x32x32x8xf32>, tensor<16x2x2x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x8225x16xf32>
  return %0 : tensor<1x32x8225x16xf32>
}

// -----

func.func @test_conv2d_pad_right(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x2x2x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>) -> tensor<1x32x8224x16xf32> {
  // expected-error@+1 {{'tosa.conv2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = tosa.conv2d %arg0, %arg1, %arg2, %arg3, %arg3 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 8193>, stride = array<i64: 1, 1>} :
            (tensor<1x32x32x8xf32>, tensor<16x2x2x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x8224x16xf32>
  return %0 : tensor<1x32x8224x16xf32>
}

// -----

func.func @test_conv2d_stride_y(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x2x2x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>) -> tensor<*xf32> {
  // expected-error@+1 {{'tosa.conv2d' op failed level check: stride <= MAX_STRIDE}}
  %0 = tosa.conv2d %arg0, %arg1, %arg2, %arg3, %arg3 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 8193, 1>} :
            (tensor<1x32x32x8xf32>, tensor<16x2x2x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

func.func @test_conv2d_stride_x(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x2x2x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>) -> tensor<*xf32> {
  // expected-error@+1 {{'tosa.conv2d' op failed level check: stride <= MAX_STRIDE}}
  %0 = tosa.conv2d %arg0, %arg1, %arg2, %arg3, %arg3 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 8193>} :
            (tensor<1x32x32x8xf32>, tensor<16x2x2x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

func.func @test_conv3d_dilation_d(%arg0: tensor<1x1x32x32x8xf32>, %arg1: tensor<16x2x2x2x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv3d' op failed level check: dilation_d * KD <= MAX_KERNEL}}
  %0 = tosa.conv3d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, dilation = array<i64: 4097, 1, 1>, pad = array<i64: 0, 1, 0, 1, 0, 1>, stride = array<i64: 1, 1, 1>} :
            (tensor<1x1x32x32x8xf32>, tensor<16x2x2x2x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x32x32x16xf32>
  return %0 : tensor<1x1x32x32x16xf32>
}

// -----

func.func @test_conv3d_dilation_y(%arg0: tensor<1x1x32x32x8xf32>, %arg1: tensor<16x2x2x2x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv3d' op failed level check: dilation_y * KH <= MAX_KERNEL}}
  %0 = tosa.conv3d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, dilation = array<i64: 1, 4097, 1>, pad = array<i64: 0, 1, 0, 1, 0, 1>, stride = array<i64: 1, 1, 1>} :
            (tensor<1x1x32x32x8xf32>, tensor<16x2x2x2x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x32x32x16xf32>
  return %0 : tensor<1x1x32x32x16xf32>
}

// -----

func.func @test_conv3d_dilation_x(%arg0: tensor<1x1x32x32x8xf32>, %arg1: tensor<16x2x2x2x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv3d' op failed level check: dilation_x * KW <= MAX_KERNEL}}
  %0 = tosa.conv3d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, dilation = array<i64: 1, 1, 4097>, pad = array<i64: 0, 1, 0, 1, 0, 1>, stride = array<i64: 1, 1, 1>} :
            (tensor<1x1x32x32x8xf32>, tensor<16x2x2x2x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x32x32x16xf32>
  return %0 : tensor<1x1x32x32x16xf32>
}

// -----

func.func @test_conv3d_pad_d0(%arg0: tensor<1x1x32x32x8xf32>, %arg1: tensor<16x2x2x2x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv3d' op failed level check: pad <= MAX_KERNEL}}
  %0 = tosa.conv3d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 8193, 1, 0, 1, 0, 1>, stride = array<i64: 1, 1, 1>} :
            (tensor<1x1x32x32x8xf32>, tensor<16x2x2x2x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x32x32x16xf32>
  return %0 : tensor<1x1x32x32x16xf32>
}

// -----

func.func @test_conv3d_pad_d1(%arg0: tensor<1x1x32x32x8xf32>, %arg1: tensor<16x2x2x2x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv3d' op failed level check: pad <= MAX_KERNEL}}
  %0 = tosa.conv3d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 1, 8193, 0, 1, 0, 1>, stride = array<i64: 1, 1, 1>} :
            (tensor<1x1x32x32x8xf32>, tensor<16x2x2x2x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x32x32x16xf32>
  return %0 : tensor<1x1x32x32x16xf32>
}

// -----

func.func @test_conv3d_pad_top(%arg0: tensor<1x1x32x32x8xf32>, %arg1: tensor<16x2x2x2x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv3d' op failed level check: pad <= MAX_KERNEL}}
  %0 = tosa.conv3d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 1, 8193, 1, 0, 1>, stride = array<i64: 1, 1, 1>} :
            (tensor<1x1x32x32x8xf32>, tensor<16x2x2x2x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x32x32x16xf32>
  return %0 : tensor<1x1x32x32x16xf32>
}

// -----

func.func @test_conv3d_pad_bottom(%arg0: tensor<1x1x32x32x8xf32>, %arg1: tensor<16x2x2x2x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv3d' op failed level check: pad <= MAX_KERNEL}}
  %0 = tosa.conv3d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 1, 0, 8193, 0, 1>, stride = array<i64: 1, 1, 1>} :
            (tensor<1x1x32x32x8xf32>, tensor<16x2x2x2x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x32x32x16xf32>
  return %0 : tensor<1x1x32x32x16xf32>
}

// -----

func.func @test_conv3d_pad_left(%arg0: tensor<1x1x32x32x8xf32>, %arg1: tensor<16x2x2x2x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv3d' op failed level check: pad <= MAX_KERNEL}}
  %0 = tosa.conv3d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 1, 0, 1, 8193, 1>, stride = array<i64: 1, 1, 1>} :
            (tensor<1x1x32x32x8xf32>, tensor<16x2x2x2x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x32x32x16xf32>
  return %0 : tensor<1x1x32x32x16xf32>
}

// -----

func.func @test_conv3d_pad_right(%arg0: tensor<1x1x32x32x8xf32>, %arg1: tensor<16x2x2x2x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv3d' op failed level check: pad <= MAX_KERNEL}}
  %0 = tosa.conv3d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 1, 0, 1, 0, 8193>, stride = array<i64: 1, 1, 1>} :
            (tensor<1x1x32x32x8xf32>, tensor<16x2x2x2x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x32x32x16xf32>
  return %0 : tensor<1x1x32x32x16xf32>
}

// -----

func.func @test_conv3d_stride_d(%arg0: tensor<1x1x32x32x8xf32>, %arg1: tensor<16x2x2x2x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv3d' op failed level check: stride <= MAX_STRIDE}}
  %0 = tosa.conv3d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 1, 0, 1, 0, 1>, stride = array<i64: 8193, 1, 1>} :
            (tensor<1x1x32x32x8xf32>, tensor<16x2x2x2x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x32x32x16xf32>
  return %0 : tensor<1x1x32x32x16xf32>
}

// -----

func.func @test_conv3d_stride_y(%arg0: tensor<1x1x32x32x8xf32>, %arg1: tensor<16x2x2x2x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv3d' op failed level check: stride <= MAX_STRIDE}}
  %0 = tosa.conv3d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 1, 0, 1, 0, 1>, stride = array<i64: 1, 8193, 1>} :
            (tensor<1x1x32x32x8xf32>, tensor<16x2x2x2x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x32x32x16xf32>
  return %0 : tensor<1x1x32x32x16xf32>
}

// -----

func.func @test_conv3d_stride_x(%arg0: tensor<1x1x32x32x8xf32>, %arg1: tensor<16x2x2x2x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.conv3d' op failed level check: stride <= MAX_STRIDE}}
  %0 = tosa.conv3d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 1, 0, 1, 0, 1>, stride = array<i64: 1, 1, 8193>} :
            (tensor<1x1x32x32x8xf32>, tensor<16x2x2x2x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x32x32x16xf32>
  return %0 : tensor<1x1x32x32x16xf32>
}

// -----

func.func @test_depthwise_conv2d_dilation_y(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<2x2x8x8xf32>, %arg2: tensor<64xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x32x32x64xf32> {
  // expected-error@+1 {{'tosa.depthwise_conv2d' op failed level check: dilation_y * KH <= MAX_KERNEL}}
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, dilation = array<i64: 4097, 1>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} :
            (tensor<1x32x32x8xf32>, tensor<2x2x8x8xf32>, tensor<64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x64xf32>
  return %0 : tensor<1x32x32x64xf32>
}

// -----

func.func @test_depthwise_conv2d_dilation_x(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<2x2x8x8xf32>, %arg2: tensor<64xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x32x32x64xf32> {
  // expected-error@+1 {{'tosa.depthwise_conv2d' op failed level check: dilation_x * KW <= MAX_KERNEL}}
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, dilation = array<i64: 1, 4097>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} :
            (tensor<1x32x32x8xf32>, tensor<2x2x8x8xf32>, tensor<64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x64xf32>
  return %0 : tensor<1x32x32x64xf32>
}

// -----

func.func @test_depthwise_conv2d_pad_top(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<2x2x8x8xf32>, %arg2: tensor<64xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x32x32x64xf32> {
  // expected-error@+1 {{'tosa.depthwise_conv2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 8193, 1, 0, 1>, stride = array<i64: 1, 1>} :
            (tensor<1x32x32x8xf32>, tensor<2x2x8x8xf32>, tensor<64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x64xf32>
  return %0 : tensor<1x32x32x64xf32>
}

// -----

func.func @test_depthwise_conv2d_pad_bottom(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<2x2x8x8xf32>, %arg2: tensor<64xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x32x32x64xf32> {
  // expected-error@+1 {{'tosa.depthwise_conv2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 8193, 0, 1>, stride = array<i64: 1, 1>} :
            (tensor<1x32x32x8xf32>, tensor<2x2x8x8xf32>, tensor<64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x64xf32>
  return %0 : tensor<1x32x32x64xf32>
}

// -----

func.func @test_depthwise_conv2d_pad_left(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<2x2x8x8xf32>, %arg2: tensor<64xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x32x32x64xf32> {
  // expected-error@+1 {{'tosa.depthwise_conv2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 8193, 1>, stride = array<i64: 1, 1>} :
            (tensor<1x32x32x8xf32>, tensor<2x2x8x8xf32>, tensor<64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x64xf32>
  return %0 : tensor<1x32x32x64xf32>
}

// -----

func.func @test_depthwise_conv2d_pad_right(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<2x2x8x8xf32>, %arg2: tensor<64xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x32x32x64xf32> {
  // expected-error@+1 {{'tosa.depthwise_conv2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 8193>, stride = array<i64: 1, 1>} :
            (tensor<1x32x32x8xf32>, tensor<2x2x8x8xf32>, tensor<64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x64xf32>
  return %0 : tensor<1x32x32x64xf32>
}

// -----

func.func @test_depthwise_conv2d_stride_y(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<2x2x8x8xf32>, %arg2: tensor<64xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x32x32x64xf32> {
  // expected-error@+1 {{'tosa.depthwise_conv2d' op failed level check: stride <= MAX_STRIDE}}
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 8193, 1>} :
            (tensor<1x32x32x8xf32>, tensor<2x2x8x8xf32>, tensor<64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x64xf32>
  return %0 : tensor<1x32x32x64xf32>
}

// -----

func.func @test_depthwise_conv2d_stride_x(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<2x2x8x8xf32>, %arg2: tensor<64xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x32x32x64xf32> {
  // expected-error@+1 {{'tosa.depthwise_conv2d' op failed level check: stride <= MAX_STRIDE}}
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 8193>} :
            (tensor<1x32x32x8xf32>, tensor<2x2x8x8xf32>, tensor<64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x64xf32>
  return %0 : tensor<1x32x32x64xf32>
}

// -----

func.func @test_fft2d_real_h(%arg0: tensor<32x16384x32xf32>, %arg1: tensor<32x16384x32xf32>) -> (tensor<32x16384x32xf32>, tensor<32x16384x32xf32>) {
  // expected-error@+1 {{'tosa.fft2d' op failed level check: H <= MAX_KERNEL}}
  %0, %1 = "tosa.fft2d"(%arg0, %arg1) { inverse = false } :
            (tensor<32x16384x32xf32>, tensor<32x16384x32xf32>) -> (tensor<32x16384x32xf32>, tensor<32x16384x32xf32>)
  return %0, %1 : tensor<32x16384x32xf32>, tensor<32x16384x32xf32>
}

// -----

func.func @test_fft2d_real_w(%arg0: tensor<32x32x16384xf32>, %arg1: tensor<32x32x16384xf32>) -> (tensor<32x32x16384xf32>, tensor<32x32x16384xf32>) {
  // expected-error@+1 {{'tosa.fft2d' op failed level check: W <= MAX_KERNEL}}
  %0, %1 = "tosa.fft2d"(%arg0, %arg1) { inverse = false } :
            (tensor<32x32x16384xf32>, tensor<32x32x16384xf32>) -> (tensor<32x32x16384xf32>, tensor<32x32x16384xf32>)
  return %0, %1 : tensor<32x32x16384xf32>, tensor<32x32x16384xf32>
}

// -----

func.func @test_fft2d_imag_h(%arg0: tensor<32x16384x32xf32>, %arg1: tensor<32x16384x32xf32>) -> (tensor<32x16384x32xf32>, tensor<32x16384x32xf32>) {
  // expected-error@+1 {{'tosa.fft2d' op failed level check: H <= MAX_KERNEL}}
  %0, %1 = "tosa.fft2d"(%arg0, %arg1) { inverse = false } :
            (tensor<32x16384x32xf32>, tensor<32x16384x32xf32>) -> (tensor<32x16384x32xf32>, tensor<32x16384x32xf32>)
  return %0, %1 : tensor<32x16384x32xf32>, tensor<32x16384x32xf32>
}

// -----

func.func @test_fft2d_imag_w(%arg0: tensor<32x32x16384xf32>, %arg1: tensor<32x32x16384xf32>) -> (tensor<32x32x16384xf32>, tensor<32x32x16384xf32>) {
  // expected-error@+1 {{'tosa.fft2d' op failed level check: W <= MAX_KERNEL}}
  %0, %1 = "tosa.fft2d"(%arg0, %arg1) { inverse = false } :
            (tensor<32x32x16384xf32>, tensor<32x32x16384xf32>) -> (tensor<32x32x16384xf32>, tensor<32x32x16384xf32>)
  return %0, %1 : tensor<32x32x16384xf32>, tensor<32x32x16384xf32>
}

// -----

func.func @test_maxpool2d_kernel_y(%arg0: tensor<1x8194x32x8xf32>) -> tensor<1x2x32x8xf32> {
  // expected-error@+1 {{'tosa.max_pool2d' op failed level check: kernel <= MAX_KERNEL}}
  %0 = "tosa.max_pool2d"(%arg0) {kernel = array<i64: 8193, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} :
         (tensor<1x8194x32x8xf32>) -> tensor<1x2x32x8xf32>
  return %0 : tensor<1x2x32x8xf32>
}

// -----

func.func @test_maxpool2d_kernel_x(%arg0: tensor<1x32x8194x8xf32>) -> tensor<1x32x2x8xf32> {
  // expected-error@+1 {{'tosa.max_pool2d' op failed level check: kernel <= MAX_KERNEL}}
  %0 = "tosa.max_pool2d"(%arg0) {kernel = array<i64: 1, 8193>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} :
         (tensor<1x32x8194x8xf32>) -> tensor<1x32x2x8xf32>
  return %0 : tensor<1x32x2x8xf32>
}

// -----

func.func @test_maxpool2d_stride_y(%arg0: tensor<1x8194x32x8xf32>) -> tensor<1x2x32x8xf32> {
  // expected-error@+1 {{'tosa.max_pool2d' op failed level check: stride <= MAX_STRIDE}}
  %0 = "tosa.max_pool2d"(%arg0) {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 8193, 1>} :
         (tensor<1x8194x32x8xf32>) -> tensor<1x2x32x8xf32>
  return %0 : tensor<1x2x32x8xf32>
}

// -----

func.func @test_maxpool2d_stride_x(%arg0: tensor<1x32x8194x8xf32>) -> tensor<1x32x2x8xf32> {
  // expected-error@+1 {{'tosa.max_pool2d' op failed level check: stride <= MAX_STRIDE}}
  %0 = "tosa.max_pool2d"(%arg0) {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 8193>} :
         (tensor<1x32x8194x8xf32>) -> tensor<1x32x2x8xf32>
  return %0 : tensor<1x32x2x8xf32>
}

// -----

func.func @test_rfft2d_input_h(%arg0: tensor<13x16384x16xf32>) -> (tensor<13x16384x9xf32>, tensor<13x16384x9xf32>) {
  // expected-error@+1 {{'tosa.rfft2d' op failed level check: H <= MAX_KERNEL}}
  %0, %1 = "tosa.rfft2d"(%arg0) {} : (tensor<13x16384x16xf32>) -> (tensor<13x16384x9xf32>, tensor<13x16384x9xf32>)
  return %0, %1 : tensor<13x16384x9xf32>, tensor<13x16384x9xf32>
}

// -----

func.func @test_rfft2d_input_w(%arg0: tensor<13x8x16384xf32>) -> (tensor<13x8x8193xf32>, tensor<13x8x8193xf32>) {
  // expected-error@+1 {{'tosa.rfft2d' op failed level check: W <= MAX_KERNEL}}
  %0, %1 = "tosa.rfft2d"(%arg0) {} : (tensor<13x8x16384xf32>) -> (tensor<13x8x8193xf32>, tensor<13x8x8193xf32>)
  return %0, %1 : tensor<13x8x8193xf32>, tensor<13x8x8193xf32>
}

// -----

func.func @test_transpose_conv2d_weight_h(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x8193x1x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.transpose_conv2d' op failed level check: KH <= MAX_KERNEL}}
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: 1, 32, 32, 16>, stride = array<i64: 1, 1>} :
              (tensor<1x32x32x8xf32>, tensor<16x8193x1x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

func.func @test_transpose_conv2d_weight_w(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x1x8193x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.transpose_conv2d' op failed level check: KW <= MAX_KERNEL}}
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: 1, 32, 32, 16>, stride = array<i64: 1, 1>} :
              (tensor<1x32x32x8xf32>, tensor<16x1x8193x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

func.func @test_transpose_conv2d_pad_top(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x1x1x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.transpose_conv2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, out_pad = array<i64: 8193, 0, 0, 0>, out_shape = array<i64: 1, 32, 32, 16>, stride = array<i64: 1, 1>} :
              (tensor<1x32x32x8xf32>, tensor<16x1x1x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

func.func @test_transpose_conv2d_pad_bottom(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x1x1x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.transpose_conv2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, out_pad = array<i64: 0, 8193, 0, 0>, out_shape = array<i64: 1, 32, 32, 16>, stride = array<i64: 1, 1>} :
              (tensor<1x32x32x8xf32>, tensor<16x1x1x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

func.func @test_transpose_conv2d_pad_left(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x1x1x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.transpose_conv2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, out_pad = array<i64: 0, 0, 8193, 0>, out_shape = array<i64: 1, 32, 32, 16>, stride = array<i64: 1, 1>} :
              (tensor<1x32x32x8xf32>, tensor<16x1x1x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

func.func @test_transpose_conv2d_pad_right(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x1x1x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.transpose_conv2d' op failed level check: pad <= MAX_KERNEL}}
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, out_pad = array<i64: 0, 0, 0, 8193>, out_shape = array<i64: 1, 32, 32, 16>, stride = array<i64: 1, 1>} :
              (tensor<1x32x32x8xf32>, tensor<16x1x1x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

func.func @test_transpose_conv2d_stride_y(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x1x1x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.transpose_conv2d' op failed level check: stride <= MAX_STRIDE}}
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: 1, 32, 32, 16>, stride = array<i64: 8193, 1>} :
              (tensor<1x32x32x8xf32>, tensor<16x1x1x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

func.func @test_transpose_conv2d_stride_x(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x1x1x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.transpose_conv2d' op failed level check: stride <= MAX_STRIDE}}
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: 1, 32, 32, 16>, stride = array<i64: 1, 8193>} :
              (tensor<1x32x32x8xf32>, tensor<16x1x1x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

func.func @test_resize_scale_y(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x7970x64x8xf32> {
  %scale = tosa.const_shape { values = dense<[257, 1, 4, 2]> : tensor<4xindex> } : () -> !tosa.shape<4>
  %offset = tosa.const_shape { values = dense<[-1, -1]> : tensor<2xindex> } : () -> !tosa.shape<2>
  %border = tosa.const_shape { values = dense<[1, 1]> : tensor<2xindex> } : () -> !tosa.shape<2>
  // expected-error@+1 {{'tosa.resize' op failed level check: scale_y_n/scale_y_d <= MAX_SCALE}}
  %1 = tosa.resize %arg0, %scale, %offset, %border {mode = "BILINEAR"} :
                (tensor<1x32x32x8xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x7970x64x8xf32>
  return %1 : tensor<1x7970x64x8xf32>
}

// -----

func.func @test_resize_scale_x(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x64x7970x8xf32> {
  %scale = tosa.const_shape { values = dense<[4, 2, 257, 1]> : tensor<4xindex> } : () -> !tosa.shape<4>
  %offset = tosa.const_shape { values = dense<[-1, -1]> : tensor<2xindex> } : () -> !tosa.shape<2>
  %border = tosa.const_shape { values = dense<[1, 1]> : tensor<2xindex> } : () -> !tosa.shape<2>
  // expected-error@+1 {{'tosa.resize' op failed level check: scale_x_n/scale_x_d <= MAX_SCALE}}
  %1 = tosa.resize %arg0, %scale, %offset, %border {mode = "BILINEAR"} :
                (tensor<1x32x32x8xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x64x7970x8xf32>
  return %1 : tensor<1x64x7970x8xf32>
}

// -----

func.func @test_tensor_size_valid(%arg0: tensor<1x536870911xf32>) {
  %0 = tosa.const_shape {values = dense<0> : tensor<2xindex>} : () -> !tosa.shape<2>
  %1 = tosa.const_shape {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
  %2= tosa.slice %arg0, %0, %1 : (tensor<1x536870911xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x1xf32>
  return
}

// -----

func.func @test_slice_tensor_size_invalid(%arg0: tensor<1x536870912xf32>) {
  %0 = tosa.const_shape {values = dense<0> : tensor<2xindex>} : () -> !tosa.shape<2>
  %1 = tosa.const_shape {values = dense<536870912> : tensor<2xindex>} : () -> !tosa.shape<2>
  // expected-error@+1 {{'tosa.slice' op failed level check: operand tensor size (in bytes) <= (1 << MAX_LOG2_SIZE - 1)}}
  %2= tosa.slice %arg0, %0, %1 : (tensor<1x536870912xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x1xf32>
  return
}


// -----

func.func @test_resize_tensor_size_invalid(%arg0: tensor<1x23178x23178x1xf32>) {
  %scale = tosa.const_shape { values = dense<[127, 49, 12, 49]> : tensor<4xindex> } : () -> !tosa.shape<4>
  %offset = tosa.const_shape { values = dense<0> : tensor<2xindex> } : () -> !tosa.shape<2>
  %border = tosa.const_shape { values = dense<0> : tensor<2xindex> } : () -> !tosa.shape<2>
  // expected-error@+1 {{'tosa.resize' op failed level check: operand tensor size (in bytes) <= (1 << MAX_LOG2_SIZE - 1)}}
  %0 = tosa.resize %arg0, %scale, %offset, %border {mode = "NEAREST_NEIGHBOR"} : (tensor<1x23178x23178x1xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<?x?x?x?xf32>
  return
}

// -----

func.func @test_avg_pool2d_tensor_size_invalid(%arg0: tensor<1x23178x23178x9xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> tensor<1x23178x23178x9xf32> {
  // expected-error@+1 {{'tosa.avg_pool2d' op failed level check: operand tensor size (in bytes) <= (1 << MAX_LOG2_SIZE - 1)}}
  %0 = tosa.avg_pool2d %arg0, %arg1, %arg2 {acc_type = f32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} : (tensor<1x23178x23178x9xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x23178x23178x9xf32>
  return %0 : tensor<1x23178x23178x9xf32>
}

// -----

func.func @test_conv2d_tensor_size_invalid(%arg0: tensor<1x23178x23178x4xf32>, %arg1: tensor<8x1x1x4xf32>, %arg2: tensor<8xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x23178x23178x8xf32> {
  // expected-error@+1 {{'tosa.conv2d' op failed level check: operand tensor size (in bytes) <= (1 << MAX_LOG2_SIZE - 1)}}
  %0 = tosa.conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, local_bound = true} : (tensor<1x23178x23178x4xf32>, tensor<8x1x1x4xf32>, tensor<8xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x23178x23178x8xf32>
  return %0 : tensor<1x23178x23178x8xf32>
}

// -----

func.func @test_fft2d_tensor_size_invalid(%arg0: tensor<123456x8192x8192xf32>, %arg1: tensor<123456x8192x8192xf32>) -> (tensor<123456x8192x8192xf32>, tensor<123456x8192x8192xf32>) {
  // expected-error@+1 {{'tosa.fft2d' op failed level check: operand tensor size (in bytes) <= (1 << MAX_LOG2_SIZE - 1)}}
  %0, %1 = tosa.fft2d %arg0, %arg1 {inverse = false} : (tensor<123456x8192x8192xf32>, tensor<123456x8192x8192xf32>) -> (tensor<123456x8192x8192xf32>, tensor<123456x8192x8192xf32>)
  return %0, %1 : tensor<123456x8192x8192xf32>, tensor<123456x8192x8192xf32>
}

// -----

func.func @test_rfft2d_tensor_size_invalid(%arg0: tensor<536870912x8x16xf32>) -> (tensor<536870912x8x9xf32>, tensor<536870912x8x9xf32>) {
  // expected-error@+1 {{'tosa.rfft2d' op failed level check: operand tensor size (in bytes) <= (1 << MAX_LOG2_SIZE - 1)}}
  %0, %1 = tosa.rfft2d %arg0 : (tensor<536870912x8x16xf32>) -> (tensor<536870912x8x9xf32>, tensor<536870912x8x9xf32>)
  return %0, %1 : tensor<536870912x8x9xf32>, tensor<536870912x8x9xf32>
}

// -----

func.func @test_matmul_tensor_size_invalid(%arg0: tensor<23178x20000x19xf32>, %arg1: tensor<23178x19x28xf32>) -> tensor<23178x20000x28xf32> {
  %zero = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
  // expected-error@+1 {{'tosa.matmul' op failed level check: operand tensor size (in bytes) <= (1 << MAX_LOG2_SIZE - 1)}}
  %0 = tosa.matmul %arg0, %arg1, %zero, %zero : (tensor<23178x20000x19xf32>, tensor<23178x19x28xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<23178x20000x28xf32>
  return %0 : tensor<23178x20000x28xf32>
}

// -----

func.func @test_gather_tensor_size_invalid(%arg0: tensor<536870912x21x3xf32>, %arg1: tensor<536870912x26xi32>) -> tensor<536870912x26x3xf32> {
  // expected-error@+1 {{'tosa.gather' op failed level check: operand tensor size (in bytes) <= (1 << MAX_LOG2_SIZE - 1)}}
  %0 = tosa.gather %arg0, %arg1 : (tensor<536870912x21x3xf32>, tensor<536870912x26xi32>) -> tensor<536870912x26x3xf32>
  return %0 : tensor<536870912x26x3xf32>
}

// -----

func.func @test_custom_tensor_size_invalid(%arg0: tensor<536870912xi32>) -> tensor<536870912xi32> {
  // expected-error@+1 {{'tosa.custom' op failed level check: operand tensor size (in bytes) <= (1 << MAX_LOG2_SIZE - 1)}}
  %0 = tosa.custom %arg0 {operator_name="custom_test", domain_name="tosa.mlir_test", implementation_attrs="" } : (tensor<536870912xi32>) -> (tensor<536870912xi32>)
  return %0 : tensor<536870912xi32>
}

// -----

func.func @test_gather_tensor_size_invalid(%arg0: tensor<268435456x21x3xf32>, %arg1: tensor<268435456x26xi32>) -> tensor<268435456x26x3xf32> {
  // expected-error@+1 {{'tosa.gather' op failed level check: operand tensor size (in bytes) <= (1 << MAX_LOG2_SIZE - 1)}}
  %0 = tosa.gather %arg0, %arg1 : (tensor<268435456x21x3xf32>, tensor<268435456x26xi32>) -> tensor<268435456x26x3xf32>
  return %0 : tensor<268435456x26x3xf32>
}

// -----

func.func @test_scatter_tensor_size_invalid(%arg0: tensor<13x210000000x3xf32>, %arg1: tensor<13x260000000xi32>, %arg2: tensor<13x260000000x3xf32>) -> tensor<13x210000000x3xf32> {
  // expected-error@+1 {{'tosa.scatter' op failed level check: operand tensor size (in bytes) <= (1 << MAX_LOG2_SIZE - 1)}}
  %0 = tosa.scatter %arg0, %arg1, %arg2 : (tensor<13x210000000x3xf32>, tensor<13x260000000xi32>, tensor<13x260000000x3xf32>) -> tensor<13x210000000x3xf32>
  return %0 : tensor<13x210000000x3xf32>
}

// -----

func.func @test_variable_read_write_tensor_size_invalid() -> () {
  tosa.variable @stored_var = dense<3.14> : tensor<536870912xf32>
  // expected-error@+1 {{'tosa.variable.read' op failed level check: result tensor size (in bytes) <= (1 << MAX_LOG2_SIZE - 1)}}
  %0 = tosa.variable.read @stored_var : tensor<536870912xf32>
  // expected-error@+1 {{'tosa.variable.write' op failed level check: operand tensor size (in bytes) <= (1 << MAX_LOG2_SIZE - 1)}}
  tosa.variable.write @stored_var, %0 : tensor<536870912xf32>
  return
}

// -----

func.func @test_while_loop_tensor_size_invalid(%arg0: tensor<536870912xi32>, %arg1: tensor<i32>) {
  %0 = "tosa.const"() {values = dense<0> : tensor<i32>} : () -> tensor<i32>
  // expected-error@+1 {{'tosa.while_loop' op failed level check: operand tensor size (in bytes) <= (1 << MAX_LOG2_SIZE - 1)}}
  %1:3 = tosa.while_loop (%arg2 = %0, %arg3 = %0, %arg4 = %arg0) : (tensor<i32>, tensor<i32>, tensor<536870912xi32>) -> (tensor<i32>, tensor<i32>, tensor<536870912xi32>) {
    %2 = tosa.greater_equal %arg3, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = tosa.logical_not %2 : (tensor<i1>) -> tensor<i1>
    tosa.yield %2 : tensor<i1>
  } do {
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<536870912xi32>):
    %2 = "tosa.const"() {values = dense<1> : tensor<i32>} : () -> tensor<i32>
    %3 = "tosa.const"() {values = dense<4> : tensor<1xi32>} : () -> tensor<1xi32>
    %4 = tosa.add %arg3, %2 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    // expected-error@+1 {{'tosa.add' op failed level check: operand tensor size (in bytes) <= (1 << MAX_LOG2_SIZE - 1)}}
    %5 = tosa.add %arg4, %3 : (tensor<536870912xi32>, tensor<1xi32>) -> tensor<536870912xi32>
    %6 = tosa.add %arg2, %2 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    tosa.yield %6, %4, %5 : tensor<i32>, tensor<i32>, tensor<536870912xi32>
  }
  return
}

// -----

func.func @test_const_shape() -> !tosa.shape<4> {
  %cst = tosa.const_shape {values = dense<[1, 1, 536870912, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  return %cst : !tosa.shape<4>
}

// -----

func.func @test_cond_if_rank_valid(%arg0: tensor<1x1x1x1x1x1x1xf32>, %arg1: tensor<1x1x1x1x1x1x1xf32>, %arg2: tensor<i1>) -> tensor<1x1x1x1x1x1x1xf32> {
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

func.func @test_cond_if_rank_invalid(%arg0: tensor<1x1x1x1x1x1x1x1xf32>, %arg1: tensor<1x1x1x1x1x1x1x1xf32>, %arg2: tensor<1x1x1x1x1x1x1x1xi1>) -> tensor<1x1x1x1x1x1x1x1xf32> {
  // expected-error@+1 {{'tosa.cond_if' op failed level check: operand rank(shape) <= MAX_RANK}}
  %0 = "tosa.cond_if"(%arg2, %arg0, %arg1) ({
  ^bb0(%arg3: tensor<1x1x1x1x1x1x1x1xf32>, %arg4: tensor<1x1x1x1x1x1x1x1xf32>):
    "tosa.yield"(%arg3) : (tensor<1x1x1x1x1x1x1x1xf32>) -> ()
  },  {
  ^bb0(%arg3: tensor<1x1x1x1x1x1x1x1xf32>, %arg4: tensor<1x1x1x1x1x1x1x1xf32>):
    "tosa.yield"(%arg4) : (tensor<1x1x1x1x1x1x1x1xf32>) -> ()
  }) : (tensor<1x1x1x1x1x1x1x1xi1>, tensor<1x1x1x1x1x1x1x1xf32>, tensor<1x1x1x1x1x1x1x1xf32>) -> tensor<1x1x1x1x1x1x1x1xf32>
  return %0 : tensor<1x1x1x1x1x1x1x1xf32>
}

// -----

func.func @test_variable_read_write_rank_invalid() -> () {
  // expected-error@+1 {{'tosa.variable' op failed level check: attribute rank(shape) <= MAX_RANK}}
  tosa.variable @stored_var = dense<3.14> : tensor<1x1x1x1x1x1x1x1xf32>
  // expected-error@+1 {{'tosa.variable.read' op failed level check: result rank(shape) <= MAX_RANK}}
  %0 = tosa.variable.read @stored_var : tensor<1x1x1x1x1x1x1x1xf32>
  // expected-error@+1 {{'tosa.variable.write' op failed level check: operand rank(shape) <= MAX_RANK}}
  tosa.variable.write @stored_var, %0 : tensor<1x1x1x1x1x1x1x1xf32>
  return
}

// -----

// CHECK-LABEL: @test_while_loop
func.func @test_while_loop(%arg0: tensor<1x1x1x1x1x1x1xf32>, %arg1: tensor<i32>) {
  %0 = "tosa.const"() {values = dense<0> : tensor<i32>} : () -> tensor<i32>
  %1:2 = "tosa.while_loop"(%0, %arg0) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<1x1x1x1x1x1x1xf32>):
    %2 = "tosa.greater_equal"(%arg3, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = "tosa.logical_not"(%2) : (tensor<i1>) -> tensor<i1>
    "tosa.yield"(%3) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<1x1x1x1x1x1x1xf32>):
    %2 = "tosa.const"() {values = dense<1> : tensor<i32>} : () -> tensor<i32>
    %3 = "tosa.add"(%arg3, %2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "tosa.yield"(%3, %arg4) : (tensor<i32>, tensor<1x1x1x1x1x1x1xf32>) -> ()
  }) : (tensor<i32>, tensor<1x1x1x1x1x1x1xf32>) -> (tensor<i32>, tensor<1x1x1x1x1x1x1xf32>)
  return
}

// -----

// CHECK-LABEL: @test_custom_rank_valid
func.func @test_custom_rank_valid(%arg0: tensor<1x1x1x1x1x1x10xi32>) -> tensor<1x1x1x1x1x1x10xi32> {
  %0 = "tosa.custom"(%arg0) {operator_name="custom_test", domain_name="tosa_mlir_test", implementation_attrs=""} :
           (tensor<1x1x1x1x1x1x10xi32>) -> (tensor<1x1x1x1x1x1x10xi32>)
  return %0 : tensor<1x1x1x1x1x1x10xi32>
}

// -----

// CHECK-LABEL: unranked_tensor
func.func @test_unranked_tensor(%arg0: tensor<*xf32>) {
  %0 = tosa.const_shape {values = dense<[0]> : tensor<1xindex>} : () -> !tosa.shape<1>
  %1 = tosa.const_shape {values = dense<[1]> : tensor<1xindex>} : () -> !tosa.shape<1>

  // expected-error@+1 {{'tosa.slice' op failed level check: unranked tensor}}
  %2= tosa.slice %arg0, %0, %1 : (tensor<*xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<*xf32>
  return
}

// -----

// CHECK-LABEL: test_concat_tensor_list_size
func.func @test_concat_tensor_list_size() {
  %0 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  // expected-error@+1 {{'tosa.concat' op failed level check for MAX_TENSOR_LIST_SIZE: input1}}
  %1= tosa.concat %0, %0, %0, %0, %0, %0, %0, %0,
                  %0, %0, %0, %0, %0, %0, %0, %0,
                  %0, %0, %0, %0, %0, %0, %0, %0,
                  %0, %0, %0, %0, %0, %0, %0, %0,
                  %0, %0, %0, %0, %0, %0, %0, %0,
                  %0, %0, %0, %0, %0, %0, %0, %0,
                  %0, %0, %0, %0, %0, %0, %0, %0,
                  %0, %0, %0, %0, %0, %0, %0, %0,
                  %0 { axis = 0 : i32 }:
                  (
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>
                  ) -> tensor<65xi32>
  return
}

// -----

// CHECK-LABEL: test_custom_tensor_list_size
func.func @test_custom_tensor_list_size() {
  %0 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  // expected-error@+1 {{'tosa.custom' op failed level check for MAX_TENSOR_LIST_SIZE: input_list}}
  %1= tosa.custom %0, %0, %0, %0, %0, %0, %0, %0,
                  %0, %0, %0, %0, %0, %0, %0, %0,
                  %0, %0, %0, %0, %0, %0, %0, %0,
                  %0, %0, %0, %0, %0, %0, %0, %0,
                  %0, %0, %0, %0, %0, %0, %0, %0,
                  %0, %0, %0, %0, %0, %0, %0, %0,
                  %0, %0, %0, %0, %0, %0, %0, %0,
                  %0, %0, %0, %0, %0, %0, %0, %0,
                  %0 { domain_name = "tosa_mlir_test", operator_name = "custom_test", implementation_attrs = "" }:
                  (
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>
                  ) -> tensor<65xi32>
  return
}

// -----

// CHECK-LABEL: test_custom_tensor_list_size_results
func.func @test_custom_tensor_list_size_results() {
  %0 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>

  // expected-error@+1 {{'tosa.custom' op failed level check for MAX_TENSOR_LIST_SIZE: output_list}}
  %r:65 = tosa.custom %0 { domain_name = "tosa_mlir_test", operator_name = "custom_test", implementation_attrs = "" }:
                  ( tensor<1xi32> )
                  -> (
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>
                  )
  return
}

// -----

// CHECK-LABEL: test_if_tensor_list_size
func.func @test_if_tensor_list_size(%arg0 : tensor<i1>) {
  %0 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  // expected-error@+1 {{'tosa.cond_if' op failed level check for MAX_TENSOR_LIST_SIZE: inputs}}
  %1 = "tosa.cond_if"(%arg0,   // condition
                  %0, %0, %0, %0, %0, %0, %0, %0,
                  %0, %0, %0, %0, %0, %0, %0, %0,
                  %0, %0, %0, %0, %0, %0, %0, %0,
                  %0, %0, %0, %0, %0, %0, %0, %0,
                  %0, %0, %0, %0, %0, %0, %0, %0,
                  %0, %0, %0, %0, %0, %0, %0, %0,
                  %0, %0, %0, %0, %0, %0, %0, %0,
                  %0, %0, %0, %0, %0, %0, %0, %0,
                  %0) ({
  ^bb0(%arg3: tensor<1xi32>):
    "tosa.yield"(%arg3) : (tensor<1xi32>) -> ()
  },  {
  ^bb0(%arg3: tensor<1xi32>):
    "tosa.yield"(%arg3) : (tensor<1xi32>) -> ()
  }) : (
                    tensor<i1>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>
                  ) -> tensor<1xi32>

  return
}

// -----

// CHECK-LABEL: test_if_tensor_list_size_outputs
func.func @test_if_tensor_list_size_outputs(%arg0 : tensor<i1>) {
  %0 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>

  // expected-error@+1 {{'tosa.cond_if' op failed level check for MAX_TENSOR_LIST_SIZE: outputs}}
  %r:65 = "tosa.cond_if"(%arg0) ({
  ^bb0(%arg3: tensor<1xi32>):
    "tosa.yield"(%arg3) : (tensor<1xi32>) -> ()
  },  {
  ^bb0(%arg3: tensor<1xi32>):
    "tosa.yield"(%arg3) : (tensor<1xi32>) -> ()
  }) : (tensor<i1>) -> (
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>
                  )

  return
}

// -----

// CHECK-LABEL: test_while_tensor_list_size
func.func @test_while_tensor_list_size(%arg0: tensor<1x1x1x1x1x1x1xf32>, %arg1: tensor<1xi32>) {
  %0 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  // expected-error@+1 {{'tosa.while_loop' op failed level check for MAX_TENSOR_LIST_SIZE: inputs}}
  %1:2 = "tosa.while_loop"(%0, %arg0,
                  %0, %0, %0, %0, %0, %0, %0, %0,
                  %0, %0, %0, %0, %0, %0, %0, %0,
                  %0, %0, %0, %0, %0, %0, %0, %0,
                  %0, %0, %0, %0, %0, %0, %0, %0,
                  %0, %0, %0, %0, %0, %0, %0, %0,
                  %0, %0, %0, %0, %0, %0, %0, %0,
                  %0, %0, %0, %0, %0, %0, %0, %0,
                  %0, %0, %0, %0, %0, %0, %0
  ) ({
  ^bb0(%arg3: tensor<1xi32>, %arg4: tensor<1x1x1x1x1x1x1xf32>):
    %2 = "tosa.greater_equal"(%arg3, %arg1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %3 = "tosa.logical_not"(%2) : (tensor<1xi1>) -> tensor<1xi1>
    "tosa.yield"(%3) : (tensor<1xi1>) -> ()
  },  {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<1x1x1x1x1x1x1xf32>):
    %2 = "tosa.const"() {values = dense<1> : tensor<i32>} : () -> tensor<i32>
    %3 = "tosa.add"(%arg3, %2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "tosa.yield"(%3, %arg4) : (tensor<i32>, tensor<1x1x1x1x1x1x1xf32>) -> ()
  }) : (tensor<1xi32>, tensor<1x1x1x1x1x1x1xf32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>
  ) -> (tensor<i32>, tensor<1x1x1x1x1x1x1xf32>)

  return
}

// -----

// CHECK-LABEL: test_while_tensor_list_size_outputs
func.func @test_while_tensor_list_size_outputs(%arg0: tensor<1x1x1x1x1x1x1xf32>, %arg1: tensor<1xi32>) {
  %0 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  // expected-error@+1 {{'tosa.while_loop' op failed level check for MAX_TENSOR_LIST_SIZE: outputs}}
  %1:65 = "tosa.while_loop"(%0, %arg0) ({
  ^bb0(%arg3: tensor<1xi32>, %arg4: tensor<1x1x1x1x1x1x1xf32>):
    %2 = "tosa.greater_equal"(%arg3, %arg1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %3 = "tosa.logical_not"(%2) : (tensor<1xi1>) -> tensor<1xi1>
    "tosa.yield"(%3) : (tensor<1xi1>) -> ()
  },  {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<1x1x1x1x1x1x1xf32>):
    %2 = "tosa.const"() {values = dense<1> : tensor<i32>} : () -> tensor<i32>
    %3 = "tosa.add"(%arg3, %2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "tosa.yield"(%3, %arg4) : (tensor<i32>, tensor<1x1x1x1x1x1x1xf32>) -> ()
  }) : (tensor<1xi32>, tensor<1x1x1x1x1x1x1xf32>) -> ( tensor<i32>, tensor<1x1x1x1x1x1x1xf32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>,
                    tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>
  )

  return
}
