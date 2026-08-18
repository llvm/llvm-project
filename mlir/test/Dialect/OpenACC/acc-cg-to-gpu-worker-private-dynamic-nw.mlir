// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" --verify-diagnostics

// Verify that worker-private shared memory with dynamic num_workers emits
// a diagnostic instead of silently miscompiling.

func.func @test_worker_private_dynamic_nw(%nw: index) {
  %c32 = arith.constant 32 : index
  %c5 = arith.constant 5 : index
  %block_x = acc.par_width %c5 {par_dim = #acc.par_dim<block_x>}
  %thread_y = acc.par_width %nw {par_dim = #acc.par_dim<thread_y>}
  %thread_x = acc.par_width %c32 {par_dim = #acc.par_dim<thread_x>}
  acc.kernel_environment {
    %priv = acc.privatize : () -> !acc.private_type<memref<2xi32>>
    // expected-error @below {{failed to legalize operation 'acc.compute_region'}}
    acc.compute_region launch(%arg0 = %block_x, %arg1 = %thread_y, %arg2 = %thread_x) ins(%arg10 = %priv) : (!acc.private_type<memref<2xi32>>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      scf.parallel (%iv) = (%c0) to (%arg1) step (%c1) {
        // expected-error @below {{worker-private variables in shared memory require compile-time constant num_workers}}
        %local = acc.private_local %arg10 : (!acc.private_type<memref<2xi32>>) -> memref<2xi32>
        memref.store %c0_i32, %local[%c0] : memref<2xi32>
        scf.reduce
      } {acc.par_dims = #acc<par_dims[thread_y]>}
      acc.yield
    } {origin = "acc.parallel"}
  }
  return
}
