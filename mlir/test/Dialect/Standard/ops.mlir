// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

// CHECK-LABEL: @assert
func @assert(%arg : i1) {
  assert %arg, "Some message in case this assertion fails."
  return
}

// CHECK-LABEL: @atan
func @atan(%arg : f32) -> f32 {
  %result = math.atan %arg : f32
  return %result : f32
}

// CHECK-LABEL: @atan2
func @atan2(%arg0 : f32, %arg1 : f32) -> f32 {
  %result = math.atan2 %arg0, %arg1 : f32
  return %result : f32
}

// CHECK-LABEL: func @switch(
func @switch(%flag : i32, %caseOperand : i32) {
  switch %flag : i32, [
    default: ^bb1(%caseOperand : i32),
    42: ^bb2(%caseOperand : i32),
    43: ^bb3(%caseOperand : i32)
  ]

  ^bb1(%bb1arg : i32):
    return
  ^bb2(%bb2arg : i32):
    return
  ^bb3(%bb3arg : i32):
    return
}

// CHECK-LABEL: func @switch_i64(
func @switch_i64(%flag : i64, %caseOperand : i32) {
  switch %flag : i64, [
    default: ^bb1(%caseOperand : i32),
    42: ^bb2(%caseOperand : i32),
    43: ^bb3(%caseOperand : i32)
  ]

  ^bb1(%bb1arg : i32):
    return
  ^bb2(%bb2arg : i32):
    return
  ^bb3(%bb3arg : i32):
    return
}

// CHECK-LABEL: func @constant_complex_f32(
func @constant_complex_f32() -> complex<f32> {
  %result = constant [0.1 : f32, -1.0 : f32] : complex<f32>
  return %result : complex<f32>
}

// CHECK-LABEL: func @constant_complex_f64(
func @constant_complex_f64() -> complex<f64> {
  %result = constant [0.1 : f64, -1.0 : f64] : complex<f64>
  return %result : complex<f64>
}

// CHECK-LABEL: func @maximum
func @maximum(%v1: vector<4xf32>, %v2: vector<4xf32>,
               %f1: f32, %f2: f32,
               %i1: i32, %i2: i32) {
  %max_vector = maxf(%v1, %v2)
    : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %max_float = maxf(%f1, %f2) : (f32, f32) -> f32
  %max_signed = maxsi(%i1, %i2) : (i32, i32) -> i32
  %max_unsigned = maxui(%i1, %i2) : (i32, i32) -> i32
  return
}

// CHECK-LABEL: func @minimum
func @minimum(%v1: vector<4xf32>, %v2: vector<4xf32>,
               %f1: f32, %f2: f32,
               %i1: i32, %i2: i32) {
  %min_vector = minf(%v1, %v2)
    : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %min_float = minf(%f1, %f2) : (f32, f32) -> f32
  %min_signed = minsi(%i1, %i2) : (i32, i32) -> i32
  %min_unsigned = minui(%i1, %i2) : (i32, i32) -> i32
  return
}
