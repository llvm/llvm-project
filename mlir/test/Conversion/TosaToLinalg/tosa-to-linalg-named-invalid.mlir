// RUN: mlir-opt --split-input-file -pass-pipeline="builtin.module(func.func(tosa-to-linalg-named))" %s -verify-diagnostics

func.func @max_pool2d_adaptive_not_canonicalized(
    %arg0: tensor<1x4x4x1xf32>) -> tensor<1x2x2x1xf32> {
  %kernel = tosa.const_shape values(dense<[2, 2]> : tensor<2xindex>) : () -> !tosa.shape<2>
  %stride = tosa.const_shape values(dense<[2, 2]> : tensor<2xindex>) : () -> !tosa.shape<2>
  %pad = tosa.const_shape values(dense<[0, 0, 0, 0]> : tensor<4xindex>) : () -> !tosa.shape<4>
  // expected-error@+1 {{failed to legalize operation 'tosa.max_pool2d_adaptive'}}
  %0 = tosa.max_pool2d_adaptive %arg0, %kernel, %stride, %pad nan_mode<IGNORE> :
    (tensor<1x4x4x1xf32>, !tosa.shape<2>, !tosa.shape<2>, !tosa.shape<4>) -> tensor<1x2x2x1xf32>
  return %0 : tensor<1x2x2x1xf32>
}

// -----

func.func @avg_pool2d_adaptive_not_canonicalized(
    %arg0: tensor<1x4x4x1xf32>) -> tensor<1x2x2x1xf32> {
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %output_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %kernel = tosa.const_shape values(dense<[2, 2]> : tensor<2xindex>) : () -> !tosa.shape<2>
  %stride = tosa.const_shape values(dense<[2, 2]> : tensor<2xindex>) : () -> !tosa.shape<2>
  %pad = tosa.const_shape values(dense<[0, 0, 0, 0]> : tensor<4xindex>) : () -> !tosa.shape<4>
  // expected-error@+1 {{failed to legalize operation 'tosa.avg_pool2d_adaptive'}}
  %0 = tosa.avg_pool2d_adaptive %arg0, %input_zp, %output_zp, %kernel, %stride, %pad acc_type(f32) :
    (tensor<1x4x4x1xf32>, tensor<1xf32>, tensor<1xf32>, !tosa.shape<2>, !tosa.shape<2>, !tosa.shape<4>) -> tensor<1x2x2x1xf32>
  return %0 : tensor<1x2x2x1xf32>
}

// -----

// Unranked operands must fail conversion cleanly before accessing their shape.
func.func @matmul_unranked_a(
    %a: tensor<*xf32>, %b: tensor<1x3x6xf32>) -> tensor<1x5x6xf32> {
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  // expected-error@+1 {{failed to legalize operation 'tosa.matmul'}}
  %0 = tosa.matmul %a, %b, %a_zp, %b_zp : (tensor<*xf32>, tensor<1x3x6xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x5x6xf32>
  return %0 : tensor<1x5x6xf32>
}

// -----

// A's zero point must be statically known; keep B's zero point constant.
func.func @matmul_nonconstant_a_zero_point(
    %a: tensor<1x5x3xi8>, %b: tensor<1x3x6xi8>,
    %a_zp: tensor<1xi8>) -> tensor<1x5x6xi32> {
  %b_zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  // expected-error@+1 {{failed to legalize operation 'tosa.matmul'}}
  %0 = tosa.matmul %a, %b, %a_zp, %b_zp : (tensor<1x5x3xi8>, tensor<1x3x6xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x5x6xi32>
  return %0 : tensor<1x5x6xi32>
}

// -----

// B's zero point must be statically known after A's zero point is accepted.
func.func @matmul_nonconstant_b_zero_point(
    %a: tensor<1x5x3xi8>, %b: tensor<1x3x6xi8>,
    %b_zp: tensor<1xi8>) -> tensor<1x5x6xi32> {
  %a_zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  // expected-error@+1 {{failed to legalize operation 'tosa.matmul'}}
  %0 = tosa.matmul %a, %b, %a_zp, %b_zp : (tensor<1x5x3xi8>, tensor<1x3x6xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x5x6xi32>
  return %0 : tensor<1x5x6xi32>
}
