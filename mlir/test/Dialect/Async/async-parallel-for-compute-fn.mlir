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

// -----

// Check that for statically known inner loop bound block size is aligned and
// inner loop uses statically known loop trip counts.

// CHECK-LABEL: func @sink_constant_step(
func @sink_constant_step(%arg0: memref<?x10xf32>, %lb: index, %ub: index) {
  %one = arith.constant 1.0 : f32

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  scf.parallel (%i, %j) = (%lb, %c0) to (%ub, %c10) step (%c1, %c1) {
    memref.store %one, %arg0[%i, %j] : memref<?x10xf32>
  }

  return
}

// CHECK-LABEL: func private @parallel_compute_fn_with_aligned_loops(
// CHECK-SAME:   %[[BLOCK_INDEX:arg[0-9]+]]: index,
// CHECK-SAME:   %[[BLOCK_SIZE:arg[0-9]+]]: index,
// CHECK-SAME:   %[[TRIP_COUNT0:arg[0-9]+]]: index,
// CHECK-SAME:   %[[TRIP_COUNT1:arg[0-9]+]]: index,
// CHECK-SAME:   %[[LB0:arg[0-9]+]]: index,
// CHECK-SAME:   %[[LB1:arg[0-9]+]]: index,
// CHECK-SAME:   %[[UB0:arg[0-9]+]]: index,
// CHECK-SAME:   %[[UB1:arg[0-9]+]]: index,
// CHECK-SAME:   %[[STEP0:arg[0-9]+]]: index,
// CHECK-SAME:   %[[STEP1:arg[0-9]+]]: index,
// CHECK-SAME:   %[[MEMREF:arg[0-9]+]]: memref<?x10xf32>
// CHECK-SAME: ) {
// CHECK:        %[[C0:.*]] = arith.constant 0 : index
// CHECK:        %[[C1:.*]] = arith.constant 1 : index
// CHECK:        %[[C10:.*]] = arith.constant 10 : index
// CHECK:        scf.for %[[I:arg[0-9]+]]
// CHECK-NOT:      arith.select
// CHECK:          scf.for %[[J:arg[0-9]+]] = %c0 to %c10 step %c1

// CHECK-LABEL: func private @parallel_compute_fn(
// CHECK-SAME:   %[[BLOCK_INDEX:arg[0-9]+]]: index,
// CHECK-SAME:   %[[BLOCK_SIZE:arg[0-9]+]]: index,
// CHECK-SAME:   %[[TRIP_COUNT0:arg[0-9]+]]: index,
// CHECK-SAME:   %[[TRIP_COUNT1:arg[0-9]+]]: index,
// CHECK-SAME:   %[[LB0:arg[0-9]+]]: index,
// CHECK-SAME:   %[[LB1:arg[0-9]+]]: index,
// CHECK-SAME:   %[[UB0:arg[0-9]+]]: index,
// CHECK-SAME:   %[[UB1:arg[0-9]+]]: index,
// CHECK-SAME:   %[[STEP0:arg[0-9]+]]: index,
// CHECK-SAME:   %[[STEP1:arg[0-9]+]]: index,
// CHECK-SAME:   %[[MEMREF:arg[0-9]+]]: memref<?x10xf32>
// CHECK-SAME: ) {
// CHECK:        scf.for %[[I:arg[0-9]+]]
// CHECK:          arith.select
// CHECK:          scf.for %[[J:arg[0-9]+]]
// CHECK:          memref.store
