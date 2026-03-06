// RUN: not mlir-opt %s -affine-super-vectorizer-test=vectorize-affine-loop-nest 2>&1 | FileCheck %s

// Regression test for https://github.com/llvm/llvm-project/issues/131135
// The vectorizer used to crash (assertion failure) when inner loop bounds
// reference an outer loop induction variable. Verify a clean error is emitted.

// CHECK: error:
// CHECK-NOT: Assertion

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 1)>

func.func @inner_loop_bounds_from_outer_iv(%arg0: memref<4x4xf32>, %arg1: memref<4xf32>) {
  affine.for %i = 0 to 4 {
    affine.for %j = #map(%i) to #map1(%i) {
      %0 = affine.load %arg0[%j, %j] : memref<4x4xf32>
      affine.store %0, %arg1[%j] : memref<4xf32>
    }
  }
  return
}
