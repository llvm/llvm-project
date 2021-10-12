// RUN: mlir-opt %s -test-math-algebraic-simplification | FileCheck %s --dump-input=always

// CHECK-LABEL: @pow_noop
func @pow_noop(%arg0: f32, %arg1 : vector<4xf32>) -> (f32, vector<4xf32>) {
  // CHECK: return %arg0, %arg1
  %c = arith.constant 1.0 : f32
  %v = arith.constant dense <1.0> : vector<4xf32>
  %0 = math.powf %arg0, %c : f32
  %1 = math.powf %arg1, %v : vector<4xf32>
  return %0, %1 : f32, vector<4xf32>
}

// CHECK-LABEL: @pow_square
func @pow_square(%arg0: f32, %arg1 : vector<4xf32>) -> (f32, vector<4xf32>) {
  // CHECK: %[[SCALAR:.*]] = arith.mulf %arg0, %arg0
  // CHECK: %[[VECTOR:.*]] = arith.mulf %arg1, %arg1
  // CHECK: return %[[SCALAR]], %[[VECTOR]]
  %c = arith.constant 2.0 : f32
  %v = arith.constant dense <2.0> : vector<4xf32>
  %0 = math.powf %arg0, %c : f32
  %1 = math.powf %arg1, %v : vector<4xf32>
  return %0, %1 : f32, vector<4xf32>
}

// CHECK-LABEL: @pow_cube
func @pow_cube(%arg0: f32, %arg1 : vector<4xf32>) -> (f32, vector<4xf32>) {
  // CHECK: %[[TMP_S:.*]] = arith.mulf %arg0, %arg0
  // CHECK: %[[SCALAR:.*]] = arith.mulf %arg0, %[[TMP_S]]
  // CHECK: %[[TMP_V:.*]] = arith.mulf %arg1, %arg1
  // CHECK: %[[VECTOR:.*]] = arith.mulf %arg1, %[[TMP_V]]
  // CHECK: return %[[SCALAR]], %[[VECTOR]]
  %c = arith.constant 3.0 : f32
  %v = arith.constant dense <3.0> : vector<4xf32>
  %0 = math.powf %arg0, %c : f32
  %1 = math.powf %arg1, %v : vector<4xf32>
  return %0, %1 : f32, vector<4xf32>
}

// CHECK-LABEL: @pow_recip
func @pow_recip(%arg0: f32, %arg1 : vector<4xf32>) -> (f32, vector<4xf32>) {
  // CHECK: %[[CST_S:.*]] = arith.constant 1.0{{.*}} : f32
  // CHECK: %[[CST_V:.*]] = arith.constant dense<1.0{{.*}}> : vector<4xf32>
  // CHECK: %[[SCALAR:.*]] = arith.divf %[[CST_S]], %arg0
  // CHECK: %[[VECTOR:.*]] = arith.divf %[[CST_V]], %arg1
  // CHECK: return %[[SCALAR]], %[[VECTOR]]
  %c = arith.constant -1.0 : f32
  %v = arith.constant dense <-1.0> : vector<4xf32>
  %0 = math.powf %arg0, %c : f32
  %1 = math.powf %arg1, %v : vector<4xf32>
  return %0, %1 : f32, vector<4xf32>
}

// CHECK-LABEL: @pow_sqrt
func @pow_sqrt(%arg0: f32, %arg1 : vector<4xf32>) -> (f32, vector<4xf32>) {
  // CHECK: %[[SCALAR:.*]] = math.sqrt %arg0
  // CHECK: %[[VECTOR:.*]] = math.sqrt %arg1
  // CHECK: return %[[SCALAR]], %[[VECTOR]]
  %c = arith.constant 0.5 : f32
  %v = arith.constant dense <0.5> : vector<4xf32>
  %0 = math.powf %arg0, %c : f32
  %1 = math.powf %arg1, %v : vector<4xf32>
  return %0, %1 : f32, vector<4xf32>
}

// CHECK-LABEL: @pow_rsqrt
func @pow_rsqrt(%arg0: f32, %arg1 : vector<4xf32>) -> (f32, vector<4xf32>) {
  // CHECK: %[[SCALAR:.*]] = math.rsqrt %arg0
  // CHECK: %[[VECTOR:.*]] = math.rsqrt %arg1
  // CHECK: return %[[SCALAR]], %[[VECTOR]]
  %c = arith.constant -0.5 : f32
  %v = arith.constant dense <-0.5> : vector<4xf32>
  %0 = math.powf %arg0, %c : f32
  %1 = math.powf %arg1, %v : vector<4xf32>
  return %0, %1 : f32, vector<4xf32>
}
