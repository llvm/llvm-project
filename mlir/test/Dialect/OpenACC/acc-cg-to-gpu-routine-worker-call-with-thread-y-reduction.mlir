// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" | FileCheck %s

// Regression test for a workgroup-barrier deadlock when an `acc routine worker`
// call sits inside an `acc.predicate_region` of a kernel that also has a
// ThreadY reduction.  `processPredicateRegion`'s `hasThreadYReduction`
// branch used to add ThreadX to the region's inactive parDims unless the
// region contained a ThreadX (vector) routine call.  That exception did
// not cover ThreadY (worker) routine calls, so the call ended up gated
// on `thread_id.x == 0` — only 4 of the 128 workgroup threads entered the routine,
// while its body still emitted workgroup-wide `gpu.barrier`s, deadlocking
// across gang-loop iterations.
//
// After the fix, the worker routine call is NOT wrapped in a
// `thread_id.x == 0` predicate: it sits directly under the gang scf.parallel
// (i.e. the gpu.launch body, with no enclosing scf.if).

// CHECK-LABEL: func.func @test_worker_routine_with_thread_y_reduction
// CHECK:       gpu.launch blocks(%{{.*}}, %{{.*}}, %{{.*}}) in {{.*}} threads(%[[TIDX:.*]], %{{.*}}, %{{.*}}) in
// The worker routine call appears at top level of the launch body, with
// no `arith.cmpi eq, %tidx, ...` / `scf.if` sequence preceding it.
// CHECK-NOT:   arith.cmpi eq, %[[TIDX]]
// CHECK:       func.call @worker_routine

module attributes {gpu.container_module} {
  func.func private @worker_routine(memref<16xi32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_worker]>}
  acc.routine @acc_routine_worker func(@worker_routine) worker

  gpu.module @cuda_device_mod {
    gpu.func @test_worker_routine_kernel() kernel {
      gpu.return
    }
  }

  func.func @test_worker_routine_with_thread_y_reduction(%arg0: memref<16xi32>) {
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %bx = acc.par_width %c4 {par_dim = #acc.par_dim<block_x>}
    %tx = acc.par_width %c32 {par_dim = #acc.par_dim<thread_x>}
    %ty = acc.par_width %c4 {par_dim = #acc.par_dim<thread_y>}
    %priv = acc.privatize : () -> !acc.private_type<memref<i32>>
    acc.compute_region launch(%kbx = %bx, %ktx = %tx, %kty = %ty)
        ins(%arg10 = %priv, %arg11 = %arg0)
        : (!acc.private_type<memref<i32>>, memref<16xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %out_priv = acc.private_local %arg10 : (!acc.private_type<memref<i32>>) -> memref<i32>
      // Gang loop: contains the worker-routine call (in an
      // acc.predicate_region as it would be after InsertPredicateRegions)
      // followed by a worker reduction that sets hasThreadYReduction.
      scf.parallel (%bx_iv) = (%c0) to (%kbx) step (%c1) {
        acc.predicate_region {
          func.call @worker_routine(%arg11) : (memref<16xi32>) -> ()
        }
        scf.parallel (%ty_iv) = (%c0) to (%kty) step (%c1) {
          %loop_red = scf.parallel (%j) = (%ty_iv) to (%c16) step (%kty) init (%c0_i32) -> i32 {
            scf.reduce(%c1_i32 : i32) {
            ^bb0(%lhs: i32, %rhs: i32):
              %sum = arith.addi %lhs, %rhs : i32
              scf.reduce.return %sum : i32
            }
          } {acc.par_dims = #acc<par_dims[sequential]>}
          acc.reduction_accumulate %loop_red to %out_priv <add> : i32 -> memref<i32> {par_dims = #acc<par_dims[thread_y]>}
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_y]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      acc.yield
    } {kernel_func_name = @test_worker_routine_kernel, kernel_module_name = @cuda_device_mod, origin = "acc.parallel"}
    return
  }
}
