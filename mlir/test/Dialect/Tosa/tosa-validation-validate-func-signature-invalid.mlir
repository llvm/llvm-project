// RUN: mlir-opt %s -split-input-file -verify-diagnostics -tosa-attach-target="specification_version=1.1.draft profiles=pro_int,pro_fp extensions=shape level=8k" -tosa-validate="strict-op-spec-alignment validate-function-signature"

// expected-error@+1 {{func.func' op Function argument types must be a tensor type to be TOSA compliant, got !tosa.shape type}}
func.func @test_shape_func_input(%arg0: !tosa.shape<1>) {
  return
}

// -----

// expected-error@+1 {{'func.func' op Function return types must be a tensor type to be TOSA compliant, got !tosa.shape type}}
func.func @test_shape_func_output() -> !tosa.shape<4> {
  %cst = tosa.const_shape {values = dense<[1, 2, 3, 4]> : tensor<4xindex>} : () -> !tosa.shape<4>
  return %cst : !tosa.shape<4>
}

// -----

// expected-error@+1 {{'func.func' op failed level check: input argument 0 rank(shape) <= MAX_RANK}}
func.func @test_argument_level_check(%arg0: tensor<1x2x3x4x5x6x7x8x9xi8>) {
  return
}

// -----

// expected-error@+1 {{'func.func' op failed level check: return value 0 rank(shape) <= MAX_RANK}}
func.func @test_result_level_check() -> tensor<1x2x3x4x5x6x7x8xi8> {
  %0 = arith.constant dense<0> : tensor<1x2x3x4x5x6x7x8xi8>
  return %0 : tensor<1x2x3x4x5x6x7x8xi8>
}

// -----

// expected-error@+1 {{'func.func' op Function argument or return types must not have zero dimensions}}
func.func @test_argument_no_zero_dims(%arg0: tensor<1x0xi8>) {
  return
}

// -----

// expected-error@+1 {{'func.func' op Function argument or return types must not have zero dimensions}}
func.func @test_result_level_check() -> tensor<1x0xi8> {
  %0 = arith.constant dense<0> : tensor<1x0xi8>
  return %0 : tensor<1x0xi8>
}

// -----

// expected-error@+1 {{'func.func' op failed level check: unranked tensor}}
func.func @test_unranked_identity(%arg0: tensor<*xi8>) -> tensor<*xi8> {
  return %arg0 : tensor<*xi8>
}

// -----

// expected-error@+1 {{'func.func' op is not profile-aligned: element type 'f64' is not legal}}
func.func @test_unsupported_element_type(%arg0: tensor<1x2x1x4x5xf64>) -> tensor<1x2x1x4x5xf64> {
  return %arg0 : tensor<1x2x1x4x5xf64>
} 
