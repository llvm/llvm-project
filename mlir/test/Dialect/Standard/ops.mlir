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

// CHECK-LABEL: func @vector_splat_0d(
func @vector_splat_0d(%a: f32) -> vector<f32> {
  // CHECK: splat %{{.*}} : vector<f32>
  %0 = splat %a : vector<f32>
  return %0 : vector<f32>
}
