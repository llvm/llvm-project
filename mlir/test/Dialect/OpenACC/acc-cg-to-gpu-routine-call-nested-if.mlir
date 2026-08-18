// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" | FileCheck %s

// Test that vector routine calls inside nested control flow (scf.if) are
// properly detected. Without the fix, the routine call would not be found
// and the predication would incorrectly add thread_id.x == 0, causing a hang.

// CHECK-LABEL: @routine_call_in_nested_if
// CHECK: gpu.launch
// The routine calls should be inside the user's scf.if (based on block index),
// NOT wrapped in an additional thread_id.x == 0 predicate.
// CHECK: scf.if
// CHECK: func.call @vector_routine
// CHECK: } else {
// CHECK: func.call @vector_routine
func.func @routine_call_in_nested_if(%arg0: memref<4xf32>, %arg1: memref<4xf32>) {
  acc.compute_region ins(%arg10 = %arg0, %arg11 = %arg1) : (memref<4xf32>, memref<4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.parallel (%iv) = (%c0) to (%c2) step (%c1) {
      %cond = arith.cmpi eq, %iv, %c0 : index
      scf.if %cond {
        func.call @vector_routine(%arg10) : (memref<4xf32>) -> ()
      } else {
        func.call @vector_routine(%arg11) : (memref<4xf32>) -> ()
      }
      scf.reduce
    } {acc.par_dims = #acc<par_dims[block_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

func.func private @vector_routine(memref<4xf32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_vector]>}
acc.routine @acc_routine_vector func(@vector_routine) vector
