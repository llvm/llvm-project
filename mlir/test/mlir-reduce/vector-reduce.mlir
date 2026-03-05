// UNSUPPORTED: system-windows
// RUN: mlir-reduce %s -reduction-tree='traversal-mode=0 test=%S/vector-test.sh' | FileCheck %s

// CHECK-LABEL: func.func @reduce_vector_op
func.func @reduce_vector_op(%arg0: f32) -> vector<4xf32> {
  // CHECK-NOT: vector.broadcast
  // CHECK: %[[POISON:.*]] = ub.poison : vector<4xf32>
  // CHECK: return %[[POISON]] : vector<4xf32>
  %0 = vector.broadcast %arg0 : f32 to vector<4xf32>
  return %0 : vector<4xf32>
}
