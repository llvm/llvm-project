// RUN: split-file %s %t
// RUN: mlir-opt %t/scope-depth.mlir -affine-parallelize | FileCheck %s --check-prefix=SCOPE
// RUN: mlir-opt %t/parallel-dimensions.mlir -affine-parallelize | FileCheck %s --check-prefix=PARALLEL
// RUN: mlir-opt %t/output-dependence.mlir -affine-parallelize | FileCheck %s --check-prefix=OUTPUT

//--- scope-depth.mlir
// Loops outside gpu.launch must not shift the dependence-check dimension.
// FIXME: %i is incorrectly parallelized; the follow-up fix must keep it sequential.
// SCOPE-LABEL: func.func @scope_depth
// SCOPE: affine.for {{.*}} = 0 to 1 {
// SCOPE-NEXT: gpu.launch
// SCOPE-NEXT: affine.parallel ({{.*}}) = (1) to (5) {
// SCOPE-NEXT: affine.parallel ({{.*}}) = (0) to (3) {
func.func @scope_depth(%a: memref<6x4xf32>) {
  %c1 = arith.constant 1 : index
  affine.for %q = 0 to 1 {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      affine.for %i = 1 to 5 {
        affine.for %j = 0 to 3 {
          %v = affine.load %a[%i - 1, %j + 1] : memref<6x4xf32>
          affine.store %v, %a[%i, %j] : memref<6x4xf32>
        }
      }
      gpu.terminator
    }
  }
  return
}

//--- parallel-dimensions.mlir
// Each IV of an enclosing affine.parallel contributes an iteration dimension.
// PARALLEL-LABEL: func.func @parallel_dimensions
// PARALLEL: affine.parallel ({{.*}}, {{.*}}) = (0, 0) to (2, 2) {
// PARALLEL-NEXT: affine.for {{.*}} = 1 to 5 {
// PARALLEL-NEXT: affine.parallel ({{.*}}) = (0) to (3) {
func.func @parallel_dimensions(%a: memref<2x2x6x4xf32>) {
  affine.parallel (%p, %q) = (0, 0) to (2, 2) {
    affine.for %i = 1 to 5 {
      affine.for %j = 0 to 3 {
        %v = affine.load %a[%p, %q, %i - 1, %j + 1] : memref<2x2x6x4xf32>
        affine.store %v, %a[%p, %q, %i, %j] : memref<2x2x6x4xf32>
      }
    }
  }
  return
}

//--- output-dependence.mlir
// A store to a fixed address has an output dependence across loop iterations.
// FIXME: %i is incorrectly parallelized; the follow-up fix must keep it sequential.
// OUTPUT-LABEL: func.func @scope_depth_output_dependence
// OUTPUT: affine.for {{.*}} = 0 to 1 {
// OUTPUT-NEXT: gpu.launch
// OUTPUT-NEXT: affine.parallel ({{.*}}) = (0) to (4) {
// OUTPUT-NEXT: affine.store
func.func @scope_depth_output_dependence(%a: memref<1xindex>) {
  %c1 = arith.constant 1 : index
  affine.for %q = 0 to 1 {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      affine.for %i = 0 to 4 {
        affine.store %i, %a[0] : memref<1xindex>
      }
      gpu.terminator
    }
  }
  return
}
