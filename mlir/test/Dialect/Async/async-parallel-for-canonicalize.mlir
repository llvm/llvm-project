// RUN: mlir-opt %s                                                            \
// RUN:    -async-parallel-for=async-dispatch=true                             \
// RUN:    -canonicalize -inline -symbol-dce                                   \
// RUN: | FileCheck %s

// RUN: mlir-opt %s                                                            \
// RUN:    -async-parallel-for=async-dispatch=false                            \
// RUN:    -canonicalize -inline -symbol-dce                                   \
// RUN: | FileCheck %s

// Check that if we statically know that the parallel operation has a single
// block then all async operations will be canonicalized away and we will
// end up with a single synchonous compute function call.

// CHECK-LABEL: @loop_1d(
// CHECK:       %[[MEMREF:.*]]: memref<?xf32>
func.func @loop_1d(%arg0: memref<?xf32>) {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C100:.*]] = arith.constant 100 : index
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK:     scf.for %[[I:.*]] = %[[C0]] to %[[C100]] step %[[C1]]
  // CHECK:       memref.store %[[ONE]], %[[MEMREF]][%[[I]]]
  %lb = arith.constant 0 : index
  %ub = arith.constant 100 : index
  %st = arith.constant 1 : index
  scf.parallel (%i) = (%lb) to (%ub) step (%st) {
    %one = arith.constant 1.0 : f32
    memref.store %one, %arg0[%i] : memref<?xf32>
  }

  return
}
