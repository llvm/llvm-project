// RUN: mlir-opt %s -test-math-polynomial-approximation | FileCheck %s

// CHECK-LABEL: @tanh_scalar
func @tanh_scalar(%arg0: f32) -> f32 {
  // CHECK-NOT: tanh
  %0 = math.tanh %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: @tanh_vector
func @tanh_vector(%arg0: vector<8xf32>) -> vector<8xf32> {
  // CHECK-NOT: tanh
  %0 = math.tanh %arg0 : vector<8xf32>
  return %0 : vector<8xf32>
}
