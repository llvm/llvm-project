// RUN: mlir-opt %s -test-vector-contraction-conversion=vector-shuffle-transpose=1 | FileCheck %s

// CHECK-LABEL: func @transpose
func @transpose(%arg0: vector<2x4xf32>) -> vector<4x2xf32> {
  // CHECK: vector.shape_cast %{{.*}} : vector<2x4xf32> to vector<8xf32>
  //            0 4
  // 0 1 2 3    1 5
  // 4 5 6 7 -> 2 6
  //            3 7
  // CHECK: vector.shuffle %{{.*}} [0, 4, 1, 5, 2, 6, 3, 7] : vector<8xf32>, vector<8xf32>
  // CHECK: vector.shape_cast %{{.*}} : vector<8xf32> to vector<4x2xf32>
  %0 = vector.transpose %arg0, [1, 0] : vector<2x4xf32> to vector<4x2xf32>
  return %0 : vector<4x2xf32>
}
