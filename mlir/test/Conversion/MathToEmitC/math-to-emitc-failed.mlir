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

// -----

func.func @unsupported_f16_classification(%arg0 : f16) -> i1 {
// expected-error @+1 {{failed to legalize operation 'math.isnan' that was explicitly marked illegal}}
  %0 = math.isnan %arg0 : f16
  return %0 : i1
}

// -----

func.func @unsupported_vector_classification(%arg0 : vector<4xf32>) -> vector<4xi1> {
// expected-error @+1 {{failed to legalize operation 'math.isnan' that was explicitly marked illegal}}
  %0 = math.isnan %arg0 : vector<4xf32>
  return %0 : vector<4xi1>
}

// -----

func.func @unsupported_fma_rounding_mode(%arg0 : f32, %arg1 : f32, %arg2 : f32) -> f32 {
// expected-error @+1 {{failed to legalize operation 'math.fma' that was explicitly marked illegal}}
  %0 = math.fma %arg0, %arg1, %arg2 to_nearest_even : f32
  return %0 : f32
}
