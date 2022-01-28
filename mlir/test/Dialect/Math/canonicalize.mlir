// RUN: mlir-opt %s -canonicalize | FileCheck %s 

// CHECK-LABEL: @ceil_fold
// CHECK: %[[cst:.+]] = arith.constant 1.000000e+00 : f32
// CHECK: return %[[cst]]
func @ceil_fold() -> f32 {
  %c = arith.constant 0.3 : f32
  %r = math.ceil %c : f32
  return %r : f32
}

// CHECK-LABEL: @ceil_fold2
// CHECK: %[[cst:.+]] = arith.constant 2.000000e+00 : f32
// CHECK: return %[[cst]]
func @ceil_fold2() -> f32 {
  %c = arith.constant 2.0 : f32
  %r = math.ceil %c : f32
  return %r : f32
}

// CHECK-LABEL: @log2_fold
// CHECK: %[[cst:.+]] = arith.constant 2.000000e+00 : f32
  // CHECK: return %[[cst]]
func @log2_fold() -> f32 {
  %c = arith.constant 4.0 : f32
  %r = math.log2 %c : f32
  return %r : f32
}

// CHECK-LABEL: @log2_fold2
// CHECK: %[[cst:.+]] = arith.constant 0xFF800000 : f32
  // CHECK: return %[[cst]]
func @log2_fold2() -> f32 {
  %c = arith.constant 0.0 : f32
  %r = math.log2 %c : f32
  return %r : f32
}

// CHECK-LABEL: @log2_nofold2
// CHECK: %[[cst:.+]] = arith.constant -1.000000e+00 : f32
// CHECK:  %[[res:.+]] = math.log2 %[[cst]] : f32
  // CHECK: return %[[res]]
func @log2_nofold2() -> f32 {
  %c = arith.constant -1.0 : f32
  %r = math.log2 %c : f32
  return %r : f32
}

// CHECK-LABEL: @log2_fold_64
// CHECK: %[[cst:.+]] = arith.constant 2.000000e+00 : f64
  // CHECK: return %[[cst]]
func @log2_fold_64() -> f64 {
  %c = arith.constant 4.0 : f64
  %r = math.log2 %c : f64
  return %r : f64
}

// CHECK-LABEL: @log2_fold2_64
// CHECK: %[[cst:.+]] = arith.constant 0xFFF0000000000000 : f64
  // CHECK: return %[[cst]]
func @log2_fold2_64() -> f64 {
  %c = arith.constant 0.0 : f64
  %r = math.log2 %c : f64
  return %r : f64
}

// CHECK-LABEL: @log2_nofold2_64
// CHECK: %[[cst:.+]] = arith.constant -1.000000e+00 : f64
// CHECK:  %[[res:.+]] = math.log2 %[[cst]] : f64
  // CHECK: return %[[res]]
func @log2_nofold2_64() -> f64 {
  %c = arith.constant -1.0 : f64
  %r = math.log2 %c : f64
  return %r : f64
}
