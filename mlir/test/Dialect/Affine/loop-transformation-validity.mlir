// RUN: mlir-opt %s -test-loop-permutation="permutation-map=1,0 check-validity=1" | FileCheck %s
// RUN: mlir-opt %s -affine-loop-tile="tile-size=4" | FileCheck %s

#dynamic_index = affine_map<()[s0, s1] -> (s0 * s1)>

// Dependence analysis cannot represent the common semi-affine index. The
// remaining indices carry a (1, -1) dependence, so both transforms must fail.
// CHECK-LABEL: func.func @unknown_dependence
func.func @unknown_dependence(
    %A: memref<?x9x9xi32>, %B: memref<9x9xi32>,
    %p: index, %q: index, %value: i32) {
  // CHECK:      affine.for %[[I:.*]] = 1 to 8 {
  // CHECK-NEXT:   affine.for %[[J:.*]] = 1 to 8 {
  affine.for %i = 1 to 8 {
    affine.for %j = 1 to 8 {
      %z = affine.apply #dynamic_index()[%p, %q]
      // CHECK: affine.store %{{.*}}, %{{.*}}[%{{.*}}, %[[I]], %[[J]]]
      affine.store %value, %A[%z, %i, %j] : memref<?x9x9xi32>
      %loaded = affine.load %A[%z, %i - 1, %j + 1]
          : memref<?x9x9xi32>
      affine.store %loaded, %B[%i, %j] : memref<9x9xi32>
    }
  }
  return
}
