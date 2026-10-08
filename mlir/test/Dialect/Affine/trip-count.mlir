// This test ensures that the LoopLikeInterfaceOp methods required
// for op-agnostic trip count analysis work for affine.for.

// RUN: mlir-opt %s -test-scf-for-utils --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @affine_constant_loops
func.func @affine_constant_loops() {
  // CHECK: "test.trip-count" = 10
  affine.for %i = 0 to 10 {
    affine.yield
  }
  // CHECK: "test.trip-count" = 5
  affine.for %i = 0 to 10 step 2 {
    affine.yield
  }
  // CHECK: "test.trip-count" = 0
  affine.for %i = 10 to 0 {
    affine.yield
  }
  return
}

// -----

// CHECK-LABEL: func.func @affine_symbolic_loops
func.func @affine_symbolic_loops(%N : index) {
  // CHECK: "test.trip-count" = "none"
  affine.for %i = 0 to %N {
    affine.yield
  }

  // CHECK: "test.trip-count" = 4
  affine.for %i = max affine_map<(d0) -> (d0)>(%N) to min affine_map<(d0) -> (d0 + 4)>(%N) {
    affine.yield
  }

  return
}
