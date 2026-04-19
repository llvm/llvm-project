// RUN: mlir-opt --affine-loop-tile="tile-sizes=16,16" %s | FileCheck %s

// Verify that tiling is NOT applied when there is an anti-dependence
// (lb == ub == -1 in the dependence component) that would be violated.
// Before the fix, the condition `*depComp.lb < *depComp.ub && *depComp.ub < 0`
// incorrectly skipped the lb==ub==-1 case, allowing illegal tiling.

// CHECK-LABEL: func.func @anti_dep
// CHECK:         affine.for %{{.*}} = 0 to 1023 {
// CHECK-NEXT:      affine.for %{{.*}} = 1 to 1024 {
// CHECK-NOT:       affine.for %{{.*}} = 0 to 1023 step 16

module {
  func.func @anti_dep(%arr: memref<1024x1024xi32>) {
    affine.for %i = 0 to 1023 {
      affine.for %j = 1 to 1024 {
        %val = affine.load %arr[%i + 1, %j - 1] : memref<1024x1024xi32>
        affine.store %val, %arr[%i, %j] : memref<1024x1024xi32>
      }
    }
    return
  }
}
