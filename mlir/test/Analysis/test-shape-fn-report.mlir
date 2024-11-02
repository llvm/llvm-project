// RUN: mlir-opt %s --test-shape-function-report -verify-diagnostics

module attributes {shape.lib = [@shape_lib]} {

// expected-remark@+1 {{associated shape function: same_result_shape}}
func.func @tanh(%arg: tensor<10x20xf32>) -> tensor<10x20xf32>
    attributes {shape.function = @shape_lib::@same_result_shape} {
  // expected-remark@+1 {{implements InferType op interface}}
  %0 = math.tanh %arg : tensor<10x20xf32>
  // expected-remark@+1 {{implements InferType op interface}}
  %1 = "test.same_operand_result_type"(%0) : (tensor<10x20xf32>) -> tensor<10x20xf32>
  return %1 : tensor<10x20xf32>
}

// The shape function library with some local functions.
shape.function_library @shape_lib {
  // Test shape function that returns the shape of input arg as result shape.
  func @same_result_shape(%arg: !shape.value_shape) -> !shape.shape {
    %0 = shape_of %arg : !shape.value_shape -> !shape.shape
    return %0 : !shape.shape
  }
} mapping {
  test.same_operand_result_type = @same_result_shape
}

}
