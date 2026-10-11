// RUN: split-file %s %t
// RUN: mlir-opt %t/scope-depth.mlir -affine-parallelize | FileCheck %s --check-prefix=SCOPE
// RUN: mlir-opt %t/parallel-dimensions.mlir -affine-parallelize | FileCheck %s --check-prefix=PARALLEL
// RUN: mlir-opt %t/output-dependence.mlir -affine-parallelize | FileCheck %s --check-prefix=OUTPUT
// RUN: mlir-opt %t/multiple-outer-loops.mlir -affine-parallelize | FileCheck %s --check-prefix=MULTIPLE
// RUN: mlir-opt %t/scoped-parallel-dimensions.mlir -affine-parallelize | FileCheck %s --check-prefix=SCOPED

//--- scope-depth.mlir
// Loops outside gpu.launch must not shift the dependence-check dimension.
// SCOPE-LABEL: func.func @scope_depth
// SCOPE: affine.for {{.*}} = 0 to 1 {
// SCOPE-NEXT: gpu.launch
// SCOPE-NEXT: affine.for {{.*}} = 1 to 5 {
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
// OUTPUT-LABEL: func.func @scope_depth_output_dependence
// OUTPUT: affine.for {{.*}} = 0 to 1 {
// OUTPUT-NEXT: gpu.launch
// OUTPUT-NEXT: affine.for {{.*}} = 0 to 4 {
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

// These inputs used to abort with "Invalid depth" because loops outside the
// affine scope inflated the dependence-check depth. Check normal completion
// and the parallelization decisions after fixing the scoped depth.

//--- multiple-outer-loops.mlir
// Multiple out-of-scope loops must not exceed the scoped access-domain depth.
// MULTIPLE-LABEL: func.func @scope_depth_multiple_outer_loops
// MULTIPLE: affine.for {{.*}} = 0 to 1 {
// MULTIPLE-NEXT: affine.for {{.*}} = 0 to 1 {
// MULTIPLE-NEXT: gpu.launch
// MULTIPLE-NEXT: affine.for {{.*}} = 1 to 5 {
// MULTIPLE: affine.parallel ({{.*}}) = (0) to (4) {
func.func @scope_depth_multiple_outer_loops(%a: memref<6xf32>, %b: memref<4xf32>) {
  %c1 = arith.constant 1 : index
  affine.for %q = 0 to 1 {
    affine.for %r = 0 to 1 {
      gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
                 threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
        affine.for %i = 1 to 5 {
          %v = affine.load %a[%i - 1] : memref<6xf32>
          affine.store %v, %a[%i] : memref<6xf32>
        }
        affine.for %j = 0 to 4 {
          %v = affine.load %b[%j] : memref<4xf32>
          affine.store %v, %b[%j] : memref<4xf32>
        }
        gpu.terminator
      }
    }
  }
  return
}

//--- scoped-parallel-dimensions.mlir
// Count in-scope parallel dimensions, exclude out-of-scope ones, and ignore affine.if.
// SCOPED-LABEL: func.func @scope_depth_parallel_dimensions
// SCOPED: affine.parallel ({{.*}}, {{.*}}) = (0, 0) to (1, 1) {
// SCOPED-NEXT: gpu.launch
// SCOPED-NEXT: affine.parallel ({{.*}}, {{.*}}) = (0, 0) to (2, 2) {
// SCOPED-NEXT: affine.for {{.*}} = 1 to 5 {
// SCOPED-NEXT: affine.if
// SCOPED-NEXT: affine.parallel ({{.*}}) = (0) to (3) {
func.func @scope_depth_parallel_dimensions(%a: memref<2x2x6x4xf32>) {
  %c1 = arith.constant 1 : index
  affine.parallel (%r, %s) = (0, 0) to (1, 1) {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      affine.parallel (%p, %q) = (0, 0) to (2, 2) {
        affine.for %i = 1 to 5 {
          affine.if affine_set<(d0) : (d0 - 1 >= 0)>(%i) {
            affine.for %j = 0 to 3 {
              %v = affine.load %a[%p, %q, %i - 1, %j + 1] : memref<2x2x6x4xf32>
              affine.store %v, %a[%p, %q, %i, %j] : memref<2x2x6x4xf32>
            }
          }
        }
      }
      gpu.terminator
    }
  }
  return
}
