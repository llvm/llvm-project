// RUN: mlir-opt %s -openmp-task-based-target -split-input-file | FileCheck %s

// CHECK-LABEL: @omp_target_depend
// CHECK-SAME: (%arg0: memref<i32>, %arg1: memref<i32>) {
func.func @omp_target_depend(%arg0: memref<i32>, %arg1: memref<i32>) {
  // CHECK: omp.task depend(taskdependin -> %arg0 : memref<i32>, taskdependin -> %arg1 : memref<i32>, taskdependinout -> %arg0 : memref<i32>) {
  // CHECK: omp.target {
  omp.target depend(taskdependin -> %arg0 : memref<i32>, taskdependin -> %arg1 : memref<i32>, taskdependinout -> %arg0 : memref<i32>) {
    // CHECK: omp.terminator
    omp.terminator
  } {operandSegmentSizes = array<i32: 0,0,0,3,0>}
  return
}
