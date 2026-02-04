// RUN: mlir-opt %s --transform-interpreter | FileCheck %s
// RUN: mlir-opt %s --gpu-eliminate-barriers | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.gpu.eliminate_barriers
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: @read_read_write
func.func @read_read_write(%arg0: memref<?xf32>, %arg1: index) attributes {__parallel_region_boundary_for_test} {
  // CHECK: load
  %0 = memref.load %arg0[%arg1] : memref<?xf32>
  // The barrier between loads can be removed.
  // CHECK-NOT: barrier
  gpu.barrier
  // CHECK: load
  %1 = memref.load %arg0[%arg1] : memref<?xf32>
  %2 = arith.addf %0, %1 : f32
  // The barrier between load and store cannot be removed (unless we reason about accessed subsets).
  // CHECK: barrier
  gpu.barrier
  // CHECK: store
  memref.store %2, %arg0[%arg1] : memref<?xf32>
  return
}

// CHECK-LABEL: @write_read_read
func.func @write_read_read(%arg0: memref<?xf32>, %arg1: index, %arg2: f32) -> f32
attributes {__parallel_region_boundary_for_test} {
  // CHECK: store
  memref.store %arg2, %arg0[%arg1] : memref<?xf32>
  // The barrier between load and store cannot be removed (unless we reason about accessed subsets).
  // CHECK: barrier
  gpu.barrier
  // CHECK: load
  %0 = memref.load %arg0[%arg1] : memref<?xf32>
  // CHECK-NOT: barrier
  gpu.barrier
  // CHECK: load
  %1 = memref.load %arg0[%arg1] : memref<?xf32>
  %2 = arith.addf %0, %1 : f32
  return %2 : f32
}

// CHECK-LABEL: @write_in_a_loop
func.func @write_in_a_loop(%arg0: memref<?xf32>, %arg1: f32) attributes {__parallel_region_boundary_for_test} {
  %c0 = arith.constant 0 : index
  %c42 = arith.constant 42 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c42 step %c1 {
    memref.store %arg1, %arg0[%i] : memref<?xf32>
    // Cannot remove this barrier because it guards write-after-write between different iterations.
    // CHECK: barrier
    gpu.barrier
  }
  return
}

// CHECK-LABEL: @read_read_write_loop
func.func @read_read_write_loop(%arg0: memref<?xf32>, %arg1: f32) attributes {__parallel_region_boundary_for_test} {
  %c0 = arith.constant 0 : index
  %c42 = arith.constant 42 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c42 step %c1 {
    // (Note that if subscript were different, this would have been a race with the store at the end of the loop).
    %0 = memref.load %arg0[%i] : memref<?xf32>
    // Guards read-after-write where the write happens on the previous iteration.
    // CHECK: barrier
    gpu.barrier
    %1 = memref.load %arg0[%i] : memref<?xf32>
    %2 = arith.addf %0, %1 : f32
    // Guards write-after-read.
    // CHECK: barrier
    gpu.barrier
    memref.store %2, %arg0[%i] : memref<?xf32>
  }
  return
}

// CHECK-LABEL: @read_read_write_loop_trailing_sync
func.func @read_read_write_loop_trailing_sync(%arg0: memref<?xf32>, %arg1: f32) attributes {__parallel_region_boundary_for_test} {
  %c0 = arith.constant 0 : index
  %c42 = arith.constant 42 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c42 step %c1 {
    // CHECK: load
    %0 = memref.load %arg0[%i] : memref<?xf32>
    // This can be removed because it only guards a read-after-read.
    // CHECK-NOT: barrier
    gpu.barrier
    // CHECK: load
    %1 = memref.load %arg0[%i] : memref<?xf32>
    %2 = arith.addf %0, %1 : f32
    // CHECK: barrier
    gpu.barrier
    // CHECK: store
    memref.store %2, %arg0[%i] : memref<?xf32>
    // CHECK: barrier
    gpu.barrier
  }
  return
}

// CHECK-LABEL: @read_read_write_loop_trailing_sync_non_memory_barrier
func.func @read_read_write_loop_trailing_sync_non_memory_barrier(%arg0: memref<?xf32>, %arg1: f32) attributes {__parallel_region_boundary_for_test} {
  %c0 = arith.constant 0 : index
  %c42 = arith.constant 42 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c42 step %c1 {
    // CHECK: load
    %0 = memref.load %arg0[%i] : memref<?xf32>
    // This can't be removed because it's a barrier that isn't fencing memory. We
    // don't know why it's here, so we leave it alone.
    // CHECK: gpu.barrier memfence []
    gpu.barrier memfence []
    // However, this can be removed as with the previoius example.
    // CHECK-NOT: gpu.barrier
    gpu.barrier
    // CHECK: load
    %1 = memref.load %arg0[%i] : memref<?xf32>
    %2 = arith.addf %0, %1 : f32
    // CHECK: gpu.barrier
    gpu.barrier
    // CHECK: store
    memref.store %2, %arg0[%i] : memref<?xf32>
    // CHECK: gpu.barrier
    gpu.barrier
  }
  return
}

// CHECK-LABEL: @write_write_noalias
func.func @write_write_noalias(%arg0: index, %arg1: f32) -> (memref<42xf32>, memref<10xf32>)
attributes {__parallel_region_boundary_for_test} {
  %0 = memref.alloc() : memref<42xf32>
  %1 = memref.alloc() : memref<10xf32>
  // CHECK: store
  memref.store %arg1, %0[%arg0] : memref<42xf32>
  // This can be removed because we can prove two allocations don't alias.
  // CHECK-NOT: barrier
  gpu.barrier
  // CHECK: store
  memref.store %arg1, %1[%arg0] : memref<10xf32>
  return %0, %1 : memref<42xf32>, memref<10xf32>
}

// CHECK-LABEL: @write_write_alloc_arg_noalias
func.func @write_write_alloc_arg_noalias(%arg0: index, %arg1: f32, %arg2: memref<?xf32>) -> (memref<42xf32>)
attributes {__parallel_region_boundary_for_test} {
  %0 = memref.alloc() : memref<42xf32>
  // CHECK: store
  memref.store %arg1, %0[%arg0] : memref<42xf32>
  // This can be removed because we can prove local allocation doesn't alias with a function argument.
  // CHECK-NOT: barrier
  gpu.barrier
  // CHECK: store
  memref.store %arg1, %arg2[%arg0] : memref<?xf32>
  return %0 : memref<42xf32>
}

// CHECK-LABEL: @repeated_barrier
func.func @repeated_barrier(%arg0: memref<?xf32>, %arg1: index, %arg2: f32) -> f32
attributes {__parallel_region_boundary_for_test} {
  %0 = memref.load %arg0[%arg1] : memref<?xf32>
  // CHECK: gpu.barrier
  gpu.barrier
  // CHECK-NOT: gpu.barrier
  gpu.barrier
  memref.store %arg2, %arg0[%arg1] : memref<?xf32>
  return %0 : f32
}

// CHECK-LABEL: @symmetric_stop
func.func @symmetric_stop(%val: f32) -> (f32, f32, f32, f32, f32)
attributes {__parallel_region_boundary_for_test} {
  // CHECK: %[[A:.+]] = memref.alloc
  // CHECK: %[[B:.+]] = memref.alloc
  // CHECK: %[[C:.+]] = memref.alloc
  %A = memref.alloc() : memref<f32>
  %B = memref.alloc() : memref<f32>
  %C = memref.alloc() : memref<f32>
  // CHECK: memref.store %{{.*}}, %[[A]]
  memref.store %val, %A[] : memref<f32>
  // CHECK: gpu.barrier
  gpu.barrier
  // CHECK: memref.load %[[A]]
  %0 = memref.load %A[] : memref<f32>
  // CHECK: memref.store %{{.*}}, %[[B]]
  memref.store %val, %B[] : memref<f32>
  // This barrier is eliminated because the surrounding barriers are sufficient
  // to guard write/read on all memrefs.
  // CHECK-NOT: gpu.barrier
  gpu.barrier
  // CHECK: memref.load %[[A]]
  %1 = memref.load %A[] : memref<f32>
  // CHECK: memref.store %{{.*}} %[[C]]
  memref.store %val, %C[] : memref<f32>
  // CHECK: gpu.barrier
  gpu.barrier
  // CHECK: memref.load %[[A]]
  // CHECK: memref.load %[[B]]
  // CHECK: memref.load %[[C]]
  %2 = memref.load %A[] : memref<f32>
  %3 = memref.load %B[] : memref<f32>
  %4 = memref.load %C[] : memref<f32>
  return %0, %1, %2, %3, %4 : f32, f32, f32, f32, f32
}

// CHECK-LABEL: @nested_loop_barrier_only
func.func @nested_loop_barrier_only() attributes {__parallel_region_boundary_for_test} {
  %c0 = arith.constant 0 : index
  %c42 = arith.constant 42 : index
  %c1 = arith.constant 1 : index
  // Note: the barrier can be removed and as consequence the loops get folded
  // by the greedy rewriter.
  // CHECK-NOT: scf.for
  // CHECK-NOT: gpu.barrier
  scf.for %j = %c0 to %c42 step %c1 {
    scf.for %i = %c0 to %c42 step %c1 {
      gpu.barrier
    }
  }
  return
}


// CHECK-LABEL: @workgroup_barrier_global_memory
func.func @workgroup_barrier_global_memory(
    %global: memref<?xf32, #gpu.address_space<global>>,
    %idx: index, %val: f32) -> f32
attributes {__parallel_region_boundary_for_test} {
  // CHECK: store
  memref.store %val, %global[%idx] : memref<?xf32, #gpu.address_space<global>>
  // The barrier only fences workgroup memory, so the global memory write/read
  // conflict doesn't matter - barrier can be removed.
  // CHECK-NOT: gpu.barrier
  gpu.barrier memfence [#gpu.address_space<workgroup>]
  // CHECK: load
  %0 = memref.load %global[%idx] : memref<?xf32, #gpu.address_space<global>>
  return %0 : f32
}


// CHECK-LABEL: @workgroup_barrier_workgroup_memory
func.func @workgroup_barrier_workgroup_memory(
    %workgroup: memref<?xf32, #gpu.address_space<workgroup>>,
    %idx: index, %val: f32) -> f32
attributes {__parallel_region_boundary_for_test} {
  // CHECK: store
  memref.store %val, %workgroup[%idx] : memref<?xf32, #gpu.address_space<workgroup>>
  // The barrier fences workgroup memory and there's a write/read conflict on
  // workgroup memory - barrier must be retained.
  // CHECK: gpu.barrier memfence [#gpu.address_space<workgroup>]
  gpu.barrier memfence [#gpu.address_space<workgroup>]
  // CHECK: load
  %0 = memref.load %workgroup[%idx] : memref<?xf32, #gpu.address_space<workgroup>>
  return %0 : f32
}

// Two barriers with non-overlapping address space sets: the inner workgroup
// barrier should not stop at the outer global barrier.
// CHECK-LABEL: @non_overlapping_barriers
func.func @non_overlapping_barriers(
    %global: memref<?xf32, #gpu.address_space<global>>,
    %idx: index, %val: f32) -> f32
attributes {__parallel_region_boundary_for_test} {
  // CHECK: store
  memref.store %val, %global[%idx] : memref<?xf32, #gpu.address_space<global>>
  // This global barrier guards the write/read on global memory.
  // CHECK: gpu.barrier memfence [#gpu.address_space<global>]
  gpu.barrier memfence [#gpu.address_space<global>]
  // This workgroup barrier can be removed, but shouldn't cause the barrier above
  // to be removed.
  // CHECK-NOT: gpu.barrier memfence [#gpu.address_space<workgroup>]
  gpu.barrier memfence [#gpu.address_space<workgroup>]
  // CHECK: load
  %0 = memref.load %global[%idx] : memref<?xf32, #gpu.address_space<global>>
  return %0 : f32
}

// CHECK-LABEL: @unknown_address_space
func.func @unknown_address_space(
    %unknown: memref<?xf32>,
    %idx: index, %val: f32) -> f32
attributes {__parallel_region_boundary_for_test} {
  // CHECK: store
  memref.store %val, %unknown[%idx] : memref<?xf32>
  // This barrier cannot be removed because the unknown-memory-space memref could
  // point to workgroup memory.
  // CHECK: gpu.barrier memfence [#gpu.address_space<workgroup>]
  gpu.barrier memfence [#gpu.address_space<workgroup>]
  // CHECK: load
  %0 = memref.load %unknown[%idx] : memref<?xf32>
  return %0 : f32
}

// CHECK-LABEL: @mixed_address_spaces
func.func @mixed_address_spaces(
    %global: memref<?xf32, #gpu.address_space<global>>,
    %workgroup: memref<?xf32, #gpu.address_space<workgroup>>,
    %idx: index, %val: f32) -> f32
attributes {__parallel_region_boundary_for_test} {
  // CHECK: store
  memref.store %val, %global[%idx] : memref<?xf32, #gpu.address_space<global>>
  // CHECK: store
  memref.store %val, %workgroup[%idx] : memref<?xf32, #gpu.address_space<workgroup>>
  // Barrier fences both global and workgroup. There are conflicts on at least one of them,
  // so the barrier must be retained.
  // CHECK: gpu.barrier memfence [#gpu.address_space<global>, #gpu.address_space<workgroup>]
  gpu.barrier memfence [#gpu.address_space<global>, #gpu.address_space<workgroup>]
  // CHECK: load
  %0 = memref.load %global[%idx] : memref<?xf32, #gpu.address_space<global>>
  // CHECK: load
  %1 = memref.load %workgroup[%idx] : memref<?xf32, #gpu.address_space<workgroup>>
  %2 = arith.addf %0, %1 : f32
  return %2 : f32
}

// CHECK-LABEL: @full_barrier_with_global_conflict
func.func @full_barrier_with_global_conflict(
    %global: memref<?xf32, #gpu.address_space<global>>,
    %idx: index, %val: f32) -> f32
attributes {__parallel_region_boundary_for_test} {
  // CHECK: store
  memref.store %val, %global[%idx] : memref<?xf32, #gpu.address_space<global>>
  // CHECK: gpu.barrier{{$}}
  gpu.barrier
  // CHECK: load
  %0 = memref.load %global[%idx] : memref<?xf32, #gpu.address_space<global>>
  return %0 : f32
}

// CHECK-LABEL: @barrier_fencing_nothing_removed
func.func @barrier_fencing_nothing_removed()
attributes {__parallel_region_boundary_for_test} {
  // CHECK-NOT: gpu.barrier
  gpu.barrier
  return
}

// CHECK-LABEL: @empty_barrrier_retained
func.func @empty_barrrier_retained()
attributes {__parallel_region_boundary_for_test} {
  // CHECK: gpu.barrier memfence []
  gpu.barrier memfence []
  return
}

// CHECK-LABEL: @read_write_loop_no_workgroup
func.func @read_write_loop_no_workgroup(%arg0: memref<?xf32, #gpu.address_space<global>>, %arg1: f32) attributes {__parallel_region_boundary_for_test} {
  %c0 = arith.constant 0 : index
  %c42 = arith.constant 42 : index
  %c1 = arith.constant 1 : index
  // CHECK: scf.for
  scf.for %i = %c0 to %c42 step %c1 {
    // Barrier can be eliminated because it only fences workgroup memory, which
    // this loop does not use.
    // CHECK-NOT: gpu.barrier
    gpu.barrier memfence [#gpu.address_space<workgroup>]
    // CHECK: load
    %0 = memref.load %arg0[%i] : memref<?xf32, #gpu.address_space<global>>
    %1 = arith.addf %0, %0 : f32
    // Fences workgroup memory and so has no efect here.
    // CHECK-NOT: gpu.barrier
    gpu.barrier memfence [#gpu.address_space<workgroup>]
    // CHECK: store
    memref.store %0, %arg0[%i] : memref<?xf32, #gpu.address_space<global>>
  }
  return
}

// CHECK-LABEL: @global_barrier_buffer_cast
func.func @global_barrier_buffer_cast(
    %global: memref<?xf32, #gpu.address_space<global>>,
    %idx: index, %val: f32) -> f32
attributes {__parallel_region_boundary_for_test} {
  // CHECK: store
  memref.store %val, %global[%idx] : memref<?xf32, #gpu.address_space<global>>
  // The buffer cast below shouldn't make this barrier go away
  // CHECK: gpu.barrier memfence [#gpu.address_space<global>]
  gpu.barrier memfence [#gpu.address_space<global>]
  %cast = amdgpu.fat_raw_buffer_cast %global : memref<?xf32, #gpu.address_space<global>> to memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>
  // CHECK: load
  %0 = memref.load %cast[%idx] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>
  return %0 : f32
}

// CHECK-LABEL: @workgroup_barrier_buffer_cast
func.func @workgroup_barrier_buffer_cast(
    %global: memref<?xf32, #gpu.address_space<global>>,
    %idx: index, %val: f32) -> f32
attributes {__parallel_region_boundary_for_test} {
  // CHECK: store
  memref.store %val, %global[%idx] : memref<?xf32, #gpu.address_space<global>>
  // The amdgpu buffer resource is formed from global memory, and so can't alias
  // workgroup memory, so this barrier can be eliminated.
  // CHECK-NOT: gpu.barrierw
  gpu.barrier memfence [#gpu.address_space<workgroup>]
  %cast = amdgpu.fat_raw_buffer_cast %global : memref<?xf32, #gpu.address_space<global>> to memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>
  // CHECK: load
  %0 = memref.load %cast[%idx] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>
  return %0 : f32
}
