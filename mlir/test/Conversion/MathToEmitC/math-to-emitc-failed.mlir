// RUN: mlir-opt -split-input-file -convert-math-to-emitc -verify-diagnostics %s 

func.func @unsupported_tensor_type(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
// expected-error @+1 {{failed to legalize operation 'math.absf' that was explicitly marked illegal}}
  %0 = math.absf %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @unsupported_f16_type(%arg0 : f16) -> f16 {
// expected-error @+1 {{failed to legalize operation 'math.absf' that was explicitly marked illegal}}
  %0 = math.absf %arg0 : f16
  return %0 : f16
}

// -----

func.func @unsupported_f128_type(%arg0 : f128) -> f128 {
// expected-error @+1 {{failed to legalize operation 'math.absf' that was explicitly marked illegal}}
  %0 = math.absf %arg0 : f128
  return %0 : f128
}
