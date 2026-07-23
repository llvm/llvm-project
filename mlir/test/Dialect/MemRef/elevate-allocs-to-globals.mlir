// RUN: mlir-opt --elevate-allocs-to-globals --split-input-file %s | FileCheck %s

/// Test that a single static memref.alloc is elevated to a memref.global, references
/// are replaced with memref.get_global, and the associated memref.dealloc is removed.

func.func @single_alloc() -> memref<10xf32> {
  %0 = memref.alloc() {alignment = 64 : i64} : memref<10xf32>
  memref.dealloc %0 : memref<10xf32>
  return %0 : memref<10xf32>
}

// CHECK-LABEL: func.func @single_alloc() -> memref<10xf32> {
// CHECK-NEXT:    %[[MEM:.*]] = memref.get_global @global_alloc : memref<10xf32>
// CHECK-NEXT:    return %[[MEM]] : memref<10xf32>
// CHECK-NOT:     memref.dealloc
// CHECK:       memref.global "private" @global_alloc : memref<10xf32> = uninitialized {alignment = 64 : i64}

// -----

/// Test that multiple static memref.alloc ops in the same function are elevated
/// to global memrefs without symbol name collisions.

func.func @multiple_allocs() -> (memref<10xf32>, memref<20xf32>) {
  %0 = memref.alloc() : memref<10xf32>
  %1 = memref.alloc() : memref<20xf32>
  return %0, %1 : memref<10xf32>, memref<20xf32>
}

// CHECK-LABEL: func.func @multiple_allocs() -> (memref<10xf32>, memref<20xf32>) {
// CHECK-DAG:    %[[MEM0:.*]] = memref.get_global @global_alloc_0 : memref<10xf32>
// CHECK-DAG:    %[[MEM1:.*]] = memref.get_global @global_alloc : memref<20xf32>
// CHECK:        return %[[MEM0]], %[[MEM1]] : memref<10xf32>, memref<20xf32>
// CHECK-DAG:    memref.global "private" @global_alloc : memref<20xf32> = uninitialized
// CHECK-DAG:    memref.global "private" @global_alloc_0 : memref<10xf32> = uninitialized

// -----

/// Test that a dynamically-shaped memref.alloc is ignored and not elevated to a global.

func.func @dynamic_alloc_ignored(%sz: index) -> memref<?xf32> {
  %0 = memref.alloc(%sz) : memref<?xf32>
  return %0 : memref<?xf32>
}

// CHECK-LABEL: func.func @dynamic_alloc_ignored(
// CHECK-SAME:  %[[SZ:.*]]: index) -> memref<?xf32> {
// CHECK-NEXT:    %[[MEM:.*]] = memref.alloc(%[[SZ]]) : memref<?xf32>
// CHECK-NEXT:    return %[[MEM]] : memref<?xf32>

// -----

/// Test that a partially dynamic memref.alloc is ignored and not elevated to a global.

func.func @partially_dynamic_alloc_ignored(%sz: index) -> memref<10x?xf32> {
  %0 = memref.alloc(%sz) : memref<10x?xf32>
  return %0 : memref<10x?xf32>
}

// CHECK-LABEL: func.func @partially_dynamic_alloc_ignored(
// CHECK-SAME:  %[[SZ:.*]]: index) -> memref<10x?xf32> {
// CHECK-NEXT:    %[[MEM:.*]] = memref.alloc(%[[SZ]]) : memref<10x?xf32>
// CHECK-NEXT:    return %[[MEM]] : memref<10x?xf32>

// -----

/// Test that a static memref.alloc inside a loop is ignored and not elevated to a global.

func.func @alloc_in_loop_ignored(%lb: index, %ub: index, %step: index) {
  scf.for %i = %lb to %ub step %step {
    %0 = memref.alloc() : memref<10xf32>
    memref.dealloc %0 : memref<10xf32>
  }
  return
}

// CHECK-LABEL: func.func @alloc_in_loop_ignored(
// CHECK:       scf.for
// CHECK-NEXT:    %[[MEM:.*]] = memref.alloc() : memref<10xf32>
// CHECK-NEXT:    memref.dealloc %[[MEM]] : memref<10xf32>

// -----

/// Test that a static memref.alloc inside control flow (scf.if) is ignored and not
/// elevated to a global.

func.func @alloc_in_control_flow_ignored(%cond: i1) {
  scf.if %cond {
    %0 = memref.alloc() : memref<10xf32>
    memref.dealloc %0 : memref<10xf32>
  }
  return
}

// CHECK-LABEL: func.func @alloc_in_control_flow_ignored(
// CHECK:       scf.if
// CHECK-NEXT:    %[[MEM:.*]] = memref.alloc() : memref<10xf32>
// CHECK-NEXT:    memref.dealloc %[[MEM]] : memref<10xf32>

// -----

/// Test that a static memref.alloc inside a non-ModuleOp symbol table is
/// ignored and not elevated to a global.

gpu.module @gpu_mod {
  gpu.func @kernel() {
    %0 = memref.alloc() : memref<10xf32>
    memref.dealloc %0 : memref<10xf32>
    gpu.return
  }
}

// CHECK-LABEL: gpu.module @gpu_mod
// CHECK:       gpu.func @kernel
// CHECK-NEXT:    %[[MEM:.*]] = memref.alloc() : memref<10xf32>
// CHECK-NEXT:    memref.dealloc %[[MEM]] : memref<10xf32>
// CHECK-NOT:   memref.global

// -----

/// Test that in a function with allocs both inside and outside of control flow,
/// only the alloc outside of control flow is elevated to a global.

func.func @mixed_control_flow_allocs(%cond: i1) -> memref<10xf32> {
  %outside = memref.alloc() : memref<10xf32>
  scf.if %cond {
    %inside = memref.alloc() : memref<20xf32>
    memref.dealloc %inside : memref<20xf32>
  }
  return %outside : memref<10xf32>
}

// CHECK-LABEL: func.func @mixed_control_flow_allocs(
// CHECK-SAME:  %[[COND:.*]]: i1) -> memref<10xf32> {
// CHECK-NEXT:    %[[OUTSIDE:.*]] = memref.get_global @global_alloc : memref<10xf32>
// CHECK-NEXT:    scf.if %[[COND]] {
// CHECK-NEXT:      %[[INSIDE:.*]] = memref.alloc() : memref<20xf32>
// CHECK-NEXT:      memref.dealloc %[[INSIDE]] : memref<20xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[OUTSIDE]] : memref<10xf32>
// CHECK:       memref.global "private" @global_alloc : memref<10xf32> = uninitialized
