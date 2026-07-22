// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-bind-routine),acc-routine-to-gpu-func,func.func(acc-cg-to-gpu))" | FileCheck %s

// Bound acc.routine metadata must remain available after ACCBindRoutine and
// ACCRoutineToGPUFunc so ACCCGToGPU can classify the bound callee as vector.
// Otherwise the call inside acc.predicate_region is incorrectly predicated to
// thread_id.x == 0.

// CHECK-LABEL: func.func @bound_vector_call
// CHECK: gpu.launch
// CHECK-NOT: scf.if
// CHECK: func.call @bound_vector
func.func @bound_vector_call(%arg0: memref<4xf32>) {
  %c32 = arith.constant 32 : index
  %thread_x = acc.par_width %c32 {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%tx = %thread_x) ins(%arg10 = %arg0) : (memref<4xf32>) {
    acc.predicate_region {
      func.call @wrapped_vector(%arg10) : (memref<4xf32>) -> ()
    }
    acc.yield
  } {origin = "acc.parallel"}
  return
}

func.func private @wrapped_vector(memref<4xf32>)
    attributes {acc.routine_info = #acc.routine_info<[@acc_routine_bound_vector]>}
acc.routine @acc_routine_bound_vector func(@wrapped_vector) vector bind("bound_vector")
func.func private @bound_vector(memref<4xf32>)
    attributes {acc.routine_info = #acc.routine_info<[@acc_routine_bound_vector]>}
