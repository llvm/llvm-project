// RUN: mlir-opt %s -split-input-file                                          \
// RUN:    -async-parallel-for=async-dispatch=true                             \
// RUN: | FileCheck %s

// RUN: mlir-opt %s -split-input-file                                          \
// RUN:    -async-parallel-for=async-dispatch=false                            \
// RUN:    -canonicalize -inline -symbol-dce                                   \
// RUN: | FileCheck %s

// Check that constants defined outside of the `scf.parallel` body will be
// sunk into the parallel compute function to avoid blowing up the number
// of parallel compute function arguments.

// CHECK-LABEL: func @clone_constant(
func @clone_constant(%arg0: memref<?xf32>, %lb: index, %ub: index, %st: index) {
  %one = arith.constant 1.0 : f32

  scf.parallel (%i) = (%lb) to (%ub) step (%st) {
    memref.store %one, %arg0[%i] : memref<?xf32>
  }

  return
}

// CHECK-LABEL: func private @parallel_compute_fn(
// CHECK-SAME:   %[[BLOCK_INDEX:arg[0-9]+]]: index,
// CHECK-SAME:   %[[BLOCK_SIZE:arg[0-9]+]]: index,
// CHECK-SAME:   %[[TRIP_COUNT:arg[0-9]+]]: index,
// CHECK-SAME:   %[[LB:arg[0-9]+]]: index,
// CHECK-SAME:   %[[UB:arg[0-9]+]]: index,
// CHECK-SAME:   %[[STEP:arg[0-9]+]]: index,
// CHECK-SAME:   %[[MEMREF:arg[0-9]+]]: memref<?xf32>
// CHECK-SAME: ) {
// CHECK:        %[[CST:.*]] = arith.constant 1.0{{.*}} : f32
// CHECK:        scf.for
// CHECK:          memref.store %[[CST]], %[[MEMREF]]

// -----

// Check that constant loop bound sunk into the parallel compute function.

// CHECK-LABEL: func @sink_constant_step(
func @sink_constant_step(%arg0: memref<?xf32>, %lb: index, %ub: index) {
  %one = arith.constant 1.0 : f32
  %st = arith.constant 123 : index

  scf.parallel (%i) = (%lb) to (%ub) step (%st) {
    memref.store %one, %arg0[%i] : memref<?xf32>
  }

  return
}

// CHECK-LABEL: func private @parallel_compute_fn(
// CHECK-SAME:   %[[BLOCK_INDEX:arg[0-9]+]]: index,
// CHECK-SAME:   %[[BLOCK_SIZE:arg[0-9]+]]: index,
// CHECK-SAME:   %[[TRIP_COUNT:arg[0-9]+]]: index,
// CHECK-SAME:   %[[LB:arg[0-9]+]]: index,
// CHECK-SAME:   %[[UB:arg[0-9]+]]: index,
// CHECK-SAME:   %[[STEP:arg[0-9]+]]: index,
// CHECK-SAME:   %[[MEMREF:arg[0-9]+]]: memref<?xf32>
// CHECK-SAME: ) {
// CHECK:        %[[CSTEP:.*]] = arith.constant 123 : index
// CHECK-NOT:    %[[STEP]]
// CHECK:        scf.for %[[I:arg[0-9]+]]
// CHECK:          %[[TMP:.*]] = arith.muli %[[I]], %[[CSTEP]]
// CHECK:          %[[IDX:.*]] = arith.addi %[[LB]], %[[TMP]]
// CHECK:          memref.store