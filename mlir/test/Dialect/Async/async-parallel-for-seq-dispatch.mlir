// RUN: mlir-opt %s -split-input-file -async-parallel-for=async-dispatch=false  \
// RUN: | FileCheck %s --dump-input=always

// The structure of @parallel_compute_fn checked in the async dispatch test.
// Here we only check the structure of the sequential dispatch loop.

// CHECK-LABEL: @loop_1d
func.func @loop_1d(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<?xf32>) {
  // CHECK: %[[GROUP:.*]] = async.create_group
  // CHECK: scf.for
  // CHECK:   %[[TOKEN:.*]] = async.execute
  // CHECK:     call @parallel_compute_fn
  // CHECK:     async.yield
  // CHECK:   async.add_to_group %[[TOKEN]], %[[GROUP]]
  // CHECK: call @parallel_compute_fn
  // CHECK: async.await_all %[[GROUP]]
  scf.parallel (%i) = (%arg0) to (%arg1) step (%arg2) {
    %one = arith.constant 1.0 : f32
    memref.store %one, %arg3[%i] : memref<?xf32>
  }
  return
}

// CHECK-LABEL: func private @parallel_compute_fn
// CHECK:       scf.for
// CHECK:         memref.store

// -----

// CHECK-LABEL: @loop_2d
func.func @loop_2d(%arg0: index, %arg1: index, %arg2: index, // lb, ub, step
              %arg3: index, %arg4: index, %arg5: index, // lb, ub, step
              %arg6: memref<?x?xf32>) {
  // CHECK: %[[GROUP:.*]] = async.create_group
  // CHECK: scf.for
  // CHECK:   %[[TOKEN:.*]] = async.execute
  // CHECK:     call @parallel_compute_fn
  // CHECK:     async.yield
  // CHECK:   async.add_to_group %[[TOKEN]], %[[GROUP]]
  // CHECK: call @parallel_compute_fn
  // CHECK: async.await_all %[[GROUP]]
  scf.parallel (%i0, %i1) = (%arg0, %arg3) to (%arg1, %arg4)
                            step (%arg2, %arg5) {
    %one = arith.constant 1.0 : f32
    memref.store %one, %arg6[%i0, %i1] : memref<?x?xf32>
  }
  return
}

// CHECK-LABEL: func private @parallel_compute_fn
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK:           memref.store
