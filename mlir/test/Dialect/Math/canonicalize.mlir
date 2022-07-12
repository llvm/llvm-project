// RUN: mlir-opt %s -canonicalize | FileCheck %s 

// CHECK-LABEL: @ceil_fold
// CHECK: %[[cst:.+]] = arith.constant 1.000000e+00 : f32
// CHECK: return %[[cst]]
func.func @ceil_fold() -> f32 {
  %c = arith.constant 0.3 : f32
  %r = math.ceil %c : f32
  return %r : f32
}

// CHECK-LABEL: @ceil_fold2
// CHECK: %[[cst:.+]] = arith.constant 2.000000e+00 : f32
// CHECK: return %[[cst]]
func.func @ceil_fold2() -> f32 {
  %c = arith.constant 2.0 : f32
  %r = math.ceil %c : f32
  return %r : f32
}

// CHECK-LABEL: @log2_fold
// CHECK: %[[cst:.+]] = arith.constant 2.000000e+00 : f32
  // CHECK: return %[[cst]]
func.func @log2_fold() -> f32 {
  %c = arith.constant 4.0 : f32
  %r = math.log2 %c : f32
  return %r : f32
}

// CHECK-LABEL: @log2_fold2
// CHECK: %[[cst:.+]] = arith.constant 0xFF800000 : f32
  // CHECK: return %[[cst]]
func.func @log2_fold2() -> f32 {
  %c = arith.constant 0.0 : f32
  %r = math.log2 %c : f32
  return %r : f32
}

// CHECK-LABEL: @log2_nofold2
// CHECK: %[[cst:.+]] = arith.constant -1.000000e+00 : f32
// CHECK:  %[[res:.+]] = math.log2 %[[cst]] : f32
  // CHECK: return %[[res]]
func.func @log2_nofold2() -> f32 {
  %c = arith.constant -1.0 : f32
  %r = math.log2 %c : f32
  return %r : f32
}

// CHECK-LABEL: @log2_fold_64
// CHECK: %[[cst:.+]] = arith.constant 2.000000e+00 : f64
  // CHECK: return %[[cst]]
func.func @log2_fold_64() -> f64 {
  %c = arith.constant 4.0 : f64
  %r = math.log2 %c : f64
  return %r : f64
}

// CHECK-LABEL: @log2_fold2_64
// CHECK: %[[cst:.+]] = arith.constant 0xFFF0000000000000 : f64
  // CHECK: return %[[cst]]
func.func @log2_fold2_64() -> f64 {
  %c = arith.constant 0.0 : f64
  %r = math.log2 %c : f64
  return %r : f64
}

// CHECK-LABEL: @log2_nofold2_64
// CHECK: %[[cst:.+]] = arith.constant -1.000000e+00 : f64
// CHECK:  %[[res:.+]] = math.log2 %[[cst]] : f64
  // CHECK: return %[[res]]
func.func @log2_nofold2_64() -> f64 {
  %c = arith.constant -1.0 : f64
  %r = math.log2 %c : f64
  return %r : f64
}

// CHECK-LABEL: @log2_fold_vec
// CHECK: %[[cst:.+]] = arith.constant dense<[0.000000e+00, 1.000000e+00, 1.58496249, 2.000000e+00]> : vector<4xf32>
// CHECK: return %[[cst]]
func.func @log2_fold_vec() -> (vector<4xf32>) {
  %v1 = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : vector<4xf32>
  %0 = math.log2 %v1 : vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: @powf_fold
// CHECK: %[[cst:.+]] = arith.constant 4.000000e+00 : f32
// CHECK: return %[[cst]]
func.func @powf_fold() -> f32 {
  %c = arith.constant 2.0 : f32
  %r = math.powf %c, %c : f32
  return %r : f32
}

// CHECK-LABEL: @powf_fold_vec
// CHECK: %[[cst:.+]] = arith.constant dense<[1.000000e+00, 4.000000e+00, 9.000000e+00, 1.600000e+01]> : vector<4xf32>
// CHECK: return %[[cst]]
func.func @powf_fold_vec() -> (vector<4xf32>) {
  %v1 = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : vector<4xf32>
  %v2 = arith.constant dense<[2.0, 2.0, 2.0, 2.0]> : vector<4xf32>
  %0 = math.powf %v1, %v2 : vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: @sqrt_fold
// CHECK: %[[cst:.+]] = arith.constant 2.000000e+00 : f32
// CHECK: return %[[cst]]
func.func @sqrt_fold() -> f32 {
  %c = arith.constant 4.0 : f32
  %r = math.sqrt %c : f32
  return %r : f32
}

// CHECK-LABEL: @sqrt_fold_vec
// CHECK: %[[cst:.+]] = arith.constant dense<[1.000000e+00, 1.41421354, 1.73205078, 2.000000e+00]> : vector<4xf32>
// CHECK: return %[[cst]]
func.func @sqrt_fold_vec() -> (vector<4xf32>) {
  %v1 = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : vector<4xf32>
  %0 = math.sqrt %v1 : vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: @abs_fold
// CHECK: %[[cst:.+]] = arith.constant 4.000000e+00 : f32
// CHECK: return %[[cst]]
func.func @abs_fold() -> f32 {
  %c = arith.constant -4.0 : f32
  %r = math.abs %c : f32
  return %r : f32
}

// CHECK-LABEL: @copysign_fold
// CHECK: %[[cst:.+]] = arith.constant -4.000000e+00 : f32
// CHECK: return %[[cst]]
func.func @copysign_fold() -> f32 {
  %c1 = arith.constant 4.0 : f32
  %c2 = arith.constant -9.0 : f32
  %r = math.copysign %c1, %c2 : f32
  return %r : f32
}

// CHECK-LABEL: @ctlz_fold1
// CHECK: %[[cst:.+]] = arith.constant 31 : i32
// CHECK: return %[[cst]]
func.func @ctlz_fold1() -> i32 {
  %c = arith.constant 1 : i32
  %r = math.ctlz %c : i32
  return %r : i32
}

// CHECK-LABEL: @ctlz_fold2
// CHECK: %[[cst:.+]] = arith.constant 7 : i8
// CHECK: return %[[cst]]
func.func @ctlz_fold2() -> i8 {
  %c = arith.constant 1 : i8
  %r = math.ctlz %c : i8
  return %r : i8
}

// CHECK-LABEL: @cttz_fold
// CHECK: %[[cst:.+]] = arith.constant 8 : i32
// CHECK: return %[[cst]]
func.func @cttz_fold() -> i32 {
  %c = arith.constant 256 : i32
  %r = math.cttz %c : i32
  return %r : i32
}

// CHECK-LABEL: @ctpop_fold
// CHECK: %[[cst:.+]] = arith.constant 16 : i32
// CHECK: return %[[cst]]
func.func @ctpop_fold() -> i32 {
  %c = arith.constant 0xFF0000FF : i32
  %r = math.ctpop %c : i32
  return %r : i32
}
