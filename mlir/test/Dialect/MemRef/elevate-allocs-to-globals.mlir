// RUN: mlir-opt --elevate-allocs-to-globals --split-input-file %s | FileCheck %s

/// Test that a single static memref.alloc is elevated to a memref.global, references
/// are replaced with memref.get_global, and the associated memref.dealloc is removed.

func.func @single_alloc(%val: f32, %idx: index) {
  %0 = memref.alloc() {alignment = 64 : i64} : memref<10xf32>
  memref.store %val, %0[%idx] : memref<10xf32>
  memref.dealloc %0 : memref<10xf32>
  return
}

// CHECK-LABEL: func.func @single_alloc(
// CHECK-SAME: %[[ARG0:.*]]: f32, %[[ARG1:.*]]: index) {
// CHECK-NEXT: %[[MEM:.*]] = memref.get_global @global_alloc : memref<10xf32>
// CHECK-NEXT: memref.store %[[ARG0]], %[[MEM]][%[[ARG1]]] : memref<10xf32>
// CHECK-NEXT: return
// CHECK-NOT: memref.dealloc
// CHECK: memref.global "private" @global_alloc : memref<10xf32> = uninitialized {alignment = 64 : i64}

// -----

/// Test that multiple static memref.alloc ops in the same function are elevated
/// to global memrefs without symbol name collisions.

func.func @multiple_allocs(%val: f32, %idx: index) {
  %0 = memref.alloc() : memref<10xf32>
  %1 = memref.alloc() : memref<20xf32>
  memref.store %val, %0[%idx] : memref<10xf32>
  memref.store %val, %1[%idx] : memref<20xf32>
  return
}

// CHECK-LABEL: func.func @multiple_allocs(
// CHECK-SAME: %[[ARG0:.*]]: f32, %[[ARG1:.*]]: index) {
// CHECK-DAG: %[[MEM0:.*]] = memref.get_global @global_alloc_0 : memref<10xf32>
// CHECK-DAG: %[[MEM1:.*]] = memref.get_global @global_alloc : memref<20xf32>
// CHECK: memref.store %[[ARG0]], %[[MEM0]][%[[ARG1]]] : memref<10xf32>
// CHECK: memref.store %[[ARG0]], %[[MEM1]][%[[ARG1]]] : memref<20xf32>
// CHECK-DAG: memref.global "private" @global_alloc : memref<20xf32> = uninitialized
// CHECK-DAG: memref.global "private" @global_alloc_0 : memref<10xf32> = uninitialized

// -----

/// Test that a dynamically-shaped memref.alloc is ignored and not elevated to a global.

func.func @dynamic_alloc_ignored(%val: f32, %sz: index) {
  %0 = memref.alloc(%sz) : memref<?xf32>
  memref.store %val, %0[%sz] : memref<?xf32>
  return
}

// CHECK-LABEL: func.func @dynamic_alloc_ignored(
// CHECK-SAME: %[[ARG0:.*]]: f32, %[[ARG1:.*]]: index) {
// CHECK-NEXT: %[[MEM:.*]] = memref.alloc(%[[ARG1]]) : memref<?xf32>
// CHECK-NEXT: memref.store %[[ARG0]], %[[MEM]][%[[ARG1]]] : memref<?xf32>

// -----

/// Test that a partially dynamic memref.alloc is ignored and not elevated to a global.

func.func @partially_dynamic_alloc_ignored(%val: f32, %sz: index) {
  %0 = memref.alloc(%sz) : memref<10x?xf32>
  memref.store %val, %0[%sz, %sz] : memref<10x?xf32>
  return
}

// CHECK-LABEL: func.func @partially_dynamic_alloc_ignored(
// CHECK-SAME: %[[ARG0:.*]]: f32, %[[ARG1:.*]]: index) {
// CHECK-NEXT: %[[MEM:.*]] = memref.alloc(%[[ARG1]]) : memref<10x?xf32>
// CHECK-NEXT: memref.store %[[ARG0]], %[[MEM]][%[[ARG1]], %[[ARG1]]] : memref<10x?xf32>

// -----

/// Test that a static memref.alloc inside a loop is ignored and not elevated to a global.

func.func @alloc_in_loop_ignored(%lb: index, %ub: index, %step: index, %val: f32, %idx: index) {
  scf.for %i = %lb to %ub step %step {
    %0 = memref.alloc() : memref<10xf32>
    memref.store %val, %0[%idx] : memref<10xf32>
  }
  return
}

// CHECK-LABEL: func.func @alloc_in_loop_ignored(
// CHECK: scf.for
// CHECK-NEXT: %[[MEM:.*]] = memref.alloc() : memref<10xf32>
// CHECK-NEXT: memref.store %{{.*}}, %[[MEM]]

// -----

/// Test that a static memref.alloc inside control flow (scf.if) is ignored and not
/// elevated to a global.

func.func @alloc_in_control_flow_ignored(%cond: i1, %val: f32, %idx: index) {
  scf.if %cond {
    %0 = memref.alloc() : memref<10xf32>
    memref.store %val, %0[%idx] : memref<10xf32>
  }
  return
}

// CHECK-LABEL: func.func @alloc_in_control_flow_ignored(
// CHECK: scf.if
// CHECK-NEXT: %[[MEM:.*]] = memref.alloc() : memref<10xf32>
// CHECK-NEXT: memref.store %{{.*}}, %[[MEM]]
