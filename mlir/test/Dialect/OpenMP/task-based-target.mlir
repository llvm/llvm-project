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
// CHECK-LABEL: func @omp_target_enter_update_exit_data_depend
// CHECK-SAME:([[ARG0:%.*]]: memref<?xi32>, [[ARG1:%.*]]: memref<?xi32>, [[ARG2:%.*]]: memref<?xi32>) {
func.func @omp_target_enter_update_exit_data_depend(%a: memref<?xi32>, %b: memref<?xi32>, %c: memref<?xi32>) {
// CHECK-NEXT: [[MAP0:%.*]] = omp.map_info
// CHECK-NEXT: [[MAP1:%.*]] = omp.map_info
// CHECK-NEXT: [[MAP2:%.*]] = omp.map_info
  %map_a = omp.map_info var_ptr(%a: memref<?xi32>, tensor<?xi32>) map_clauses(to) capture(ByRef) -> memref<?xi32>
  %map_b = omp.map_info var_ptr(%b: memref<?xi32>, tensor<?xi32>) map_clauses(from) capture(ByRef) -> memref<?xi32>
  %map_c = omp.map_info var_ptr(%c: memref<?xi32>, tensor<?xi32>) map_clauses(exit_release_or_enter_alloc) capture(ByRef) -> memref<?xi32>

  // Do some work on the host that writes to 'a'
  omp.task depend(taskdependout -> %a : memref<?xi32>) {
    "test.foo"(%a) : (memref<?xi32>) -> ()
    omp.terminator
  }

  // Then map that over to the target
  // CHECK: omp.task depend(taskdependin -> [[ARG0]] : memref<?xi32>)
  // CHECK: omp.target_enter_data nowait map_entries([[MAP0]], [[MAP2]] : memref<?xi32>, memref<?xi32>)
  omp.target_enter_data nowait map_entries(%map_a, %map_c: memref<?xi32>, memref<?xi32>) depend(taskdependin ->  %a: memref<?xi32>)

  // Compute 'b' on the target and copy it back
  // CHECK: omp.target map_entries([[MAP1]] -> {{%.*}} : memref<?xi32>) {
  omp.target map_entries(%map_b -> %arg0 : memref<?xi32>) {
    ^bb0(%arg0: memref<?xi32>) :
      "test.foo"(%arg0) : (memref<?xi32>) -> ()
      omp.terminator
  }

  // Update 'a' on the host using 'b'
  omp.task depend(taskdependout -> %a: memref<?xi32>){
    "test.bar"(%a, %b) : (memref<?xi32>, memref<?xi32>) -> ()
  }

  // Copy the updated 'a' onto the target
  // CHECK: omp.task depend(taskdependin -> [[ARG0]] : memref<?xi32>)
  // CHECK: omp.target_update_data nowait motion_entries([[MAP0]] : memref<?xi32>)
  omp.target_update_data motion_entries(%map_a :  memref<?xi32>) depend(taskdependin -> %a : memref<?xi32>) nowait

  // Compute 'c' on the target and copy it back
  // CHECK:[[MAP3:%.*]] = omp.map_info var_ptr([[ARG2]] : memref<?xi32>, tensor<?xi32>) map_clauses(from) capture(ByRef) -> memref<?xi32>
  %map_c_from = omp.map_info var_ptr(%c: memref<?xi32>, tensor<?xi32>) map_clauses(from) capture(ByRef) -> memref<?xi32>
  // CHECK: omp.task depend(taskdependout -> [[ARG2]] : memref<?xi32>)
  // CHECK: omp.target map_entries([[MAP0]] -> {{%.*}}, [[MAP3]] -> {{%.*}} : memref<?xi32>, memref<?xi32>) {
  omp.target map_entries(%map_a -> %arg0, %map_c_from -> %arg1 : memref<?xi32>, memref<?xi32>) depend(taskdependout -> %c : memref<?xi32>) {
  ^bb0(%arg0 : memref<?xi32>, %arg1 : memref<?xi32>) :
    "test.foobar"() : ()->()
    omp.terminator
  }
  // CHECK: omp.task depend(taskdependin -> [[ARG2]] : memref<?xi32>) {
  // CHECK: omp.target_exit_data map_entries([[MAP2]] : memref<?xi32>)
  omp.target_exit_data map_entries(%map_c : memref<?xi32>) depend(taskdependin -> %c : memref<?xi32>)
  return
}
