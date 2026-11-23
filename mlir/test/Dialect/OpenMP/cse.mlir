// RUN: mlir-opt %s -cse -split-input-file | FileCheck %s

// An implicit barrier synchronises shared memory, so loads on opposite sides
// must remain.
// CHECK-LABEL: func.func @sections_barrier_between_loads(
// CHECK: %[[BEFORE:.*]] = memref.load %[[SHARED:.*]][] : memref<i32>
// CHECK: omp.sections {
// CHECK: %[[AFTER:.*]] = memref.load %[[SHARED]][] : memref<i32>
// CHECK: return %[[BEFORE]], %[[AFTER]] : i32, i32
func.func @sections_barrier_between_loads(%shared: memref<i32>) -> (i32, i32) {
  %before = memref.load %shared[] : memref<i32>
  omp.sections {
    omp.section {
      omp.terminator
    }
    omp.terminator
  }
  %after = memref.load %shared[] : memref<i32>
  return %before, %after : i32, i32
}

// -----

// Without the private-clause assumption, arbitrary memory effects are assumed
// so no CSE is possible.
omp.private {type = private} @recipe : memref<i32>

// CHECK-LABEL: func.func @single_unbounded_private(
// CHECK-SAME: %[[X:[^:]+]]: memref<i32>
// CHECK: %[[BEFORE:.*]] = memref.load %[[X]][] : memref<i32>
// CHECK: omp.single nowait
// CHECK: %[[AFTER:.*]] = memref.load %[[X]][] : memref<i32>
// CHECK: return %[[BEFORE]], %[[AFTER]] : i32, i32
func.func @single_unbounded_private(%x: memref<i32>, %original: memref<i32>) -> (i32, i32) {
  %before = memref.load %x[] : memref<i32>
  omp.single nowait private(@recipe %original -> %copy : memref<i32>) {
    omp.terminator
  }
  %after = memref.load %x[] : memref<i32>
  return %before, %after : i32, i32
}
