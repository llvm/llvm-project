// RUN: mlir-opt %s --test-transform-dialect-interpreter | FileCheck %s

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %0 {
    transform.apply_patterns.gpu.eliminate_barriers
  } : !transform.any_op
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

// CHECK-LABEL @read_read_write_loop
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
