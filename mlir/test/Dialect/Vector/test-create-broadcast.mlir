// RUN: mlir-opt %s --test-create-vector-broadcast --allow-unregistered-dialect --split-input-file | FileCheck %s

func.func @foo(%a : f32) -> vector<1x2xf32> {
  %0 = "test_create_broadcast"(%a) {broadcast_dims = array<i64: 0, 1>} : (f32) -> vector<1x2xf32>
  // CHECK: vector.broadcast {{.*}} : f32 to vector<1x2xf32>
  // CHECK-NOT: vector.transpose
  return %0:  vector<1x2xf32>
}

// -----

func.func @foo(%a : vector<2x2xf32>) -> vector<2x2x3xf32> {
  %0 = "test_create_broadcast"(%a) {broadcast_dims = array<i64: 2>} 
    : (vector<2x2xf32>) -> vector<2x2x3xf32>
  // CHECK: vector.broadcast {{.*}} : vector<2x2xf32> to vector<3x2x2xf32>
  // CHECK: vector.transpose {{.*}}, [1, 2, 0] : vector<3x2x2xf32> to vector<2x2x3xf32>
  return %0: vector<2x2x3xf32>
}

// -----

func.func @foo(%a : vector<3x3xf32>) -> vector<4x3x3xf32> {
  %0 = "test_create_broadcast"(%a) {broadcast_dims = array<i64: 0>} 
    : (vector<3x3xf32>) -> vector<4x3x3xf32>
  // CHECK: vector.broadcast {{.*}} : vector<3x3xf32> to vector<4x3x3xf32>
  // CHECK-NOT: vector.transpose
  return %0: vector<4x3x3xf32>
}

// -----

func.func @foo(%a : vector<2x4xf32>) -> vector<1x2x3x4x5xf32> {
  %0 = "test_create_broadcast"(%a) {broadcast_dims = array<i64: 0, 2, 4>} 
    : (vector<2x4xf32>) -> vector<1x2x3x4x5xf32>
  // CHECK: vector.broadcast {{.*}} : vector<2x4xf32> to vector<1x3x5x2x4xf32>
  // CHECK: vector.transpose {{.*}}, [0, 3, 1, 4, 2] : vector<1x3x5x2x4xf32> to vector<1x2x3x4x5xf32>
  return %0: vector<1x2x3x4x5xf32>
}
