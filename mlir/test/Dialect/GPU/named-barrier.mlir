// RUN: mlir-opt %s --split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @barrier_default
//       CHECK-NEXT: gpu.barrier{{$}}
//       CHECK-NEXT: return
func.func @barrier_default() {
  gpu.barrier
  return
}

// -----

// CHECK-LABEL: func @barrier_with_scope
//       CHECK-NEXT: gpu.barrier scope <subgroup>
//       CHECK-NEXT: return
func.func @barrier_with_scope() {
  gpu.barrier scope <subgroup>
  return
}

// -----

// CHECK-LABEL: func @barrier_workgroup_scope_not_printed
//       CHECK-NEXT: gpu.barrier{{$}}
//       CHECK-NEXT: return
func.func @barrier_workgroup_scope_not_printed() {
  gpu.barrier scope <workgroup>
  return
}

// -----

// CHECK-LABEL: func @barrier_memfence_and_scope
//       CHECK-NEXT: gpu.barrier memfence [#gpu.address_space<workgroup>] scope <subgroup>
//       CHECK-NEXT: return
func.func @barrier_memfence_and_scope() {
  gpu.barrier memfence [#gpu.address_space<workgroup>] scope <subgroup>
  return
}

// -----

// CHECK-LABEL: func @initialize_named_barrier
//       CHECK: %[[NB:.*]] = gpu.initialize_named_barrier %[[MEMBER_COUNT:.*]] : i32 -> !gpu.named_barrier
//       CHECK: gpu.barrier named(%[[NB]] : !gpu.named_barrier)
func.func @initialize_named_barrier(%member_count : i32) {
  %nb = gpu.initialize_named_barrier %member_count : i32 -> !gpu.named_barrier
  gpu.barrier named(%nb : !gpu.named_barrier)
  return
}

// -----

// CHECK-LABEL: func @named_barrier_with_memfence
//       CHECK: gpu.barrier named(%{{.*}} : !gpu.named_barrier) memfence [#gpu.address_space<workgroup>]
func.func @named_barrier_with_memfence(%member_count : i32) {
  %nb = gpu.initialize_named_barrier %member_count : i32 -> !gpu.named_barrier
  gpu.barrier named(%nb : !gpu.named_barrier) memfence [#gpu.address_space<workgroup>]
  return
}

// -----

func.func @named_barrier_non_workgroup_scope(%member_count : i32) {
  %nb = gpu.initialize_named_barrier %member_count : i32 -> !gpu.named_barrier
  // expected-error @+1 {{named barriers require workgroup scope}}
  gpu.barrier named(%nb : !gpu.named_barrier) scope <subgroup>
  return
}
