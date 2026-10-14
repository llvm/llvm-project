// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" | FileCheck %s

// A gang-private slot written in a sequential `scf.for` is one copy per
// workgroup and is reused every iteration. A worker-level acc routine in the
// body runs on every workgroup thread and synchronizes internally, so threads
// can leave the call in different iterations. A workgroup barrier at the top
// of the `scf.for` body orders the next iteration's store after the previous
// iteration's last read.

module attributes {gpu.container_module} {
  func.func private @worker_routine(memref<i32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_worker]>}
  acc.routine @acc_routine_worker func(@worker_routine) worker

  // CHECK-LABEL: func.func @seq_for_worker_call_gang_private_reuse
  // CHECK:       gpu.launch
  // CHECK:         scf.for
  // CHECK-NEXT:      gpu.barrier
  // CHECK-NEXT:      memref.store
  // CHECK-NEXT:      func.call @worker_routine
  func.func @seq_for_worker_call_gang_private_reuse() {
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %bx = acc.par_width %c4 par_dim(#acc.par_dim<block_x>)
    %tx = acc.par_width %c32 par_dim(#acc.par_dim<thread_x>)
    %ty = acc.par_width %c4 par_dim(#acc.par_dim<thread_y>)
    %priv = acc.privatize : () -> !acc.private_type<memref<i32>>
    acc.compute_region launch(%kbx = %bx, %ktx = %tx, %kty = %ty)
        ins(%arg10 = %priv)
        : (!acc.private_type<memref<i32>>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      scf.parallel (%g) = (%c0) to (%kbx) step (%c1) {
        %pl = acc.private_local %arg10 : (!acc.private_type<memref<i32>>) -> memref<i32>
        scf.for %i = %c0 to %c2 step %c1 {
          memref.store %c0_i32, %pl[] : memref<i32>
          func.call @worker_routine(%pl) : (memref<i32>) -> ()
          %v = memref.load %pl[] : memref<i32>
        }
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      acc.yield
    } <{origin = "acc.parallel"}>
    return
  }

  // No worker/vector call: the seq loop stays a single-thread predicated
  // region in the usual case, and this helper must not insert a body barrier.
  // CHECK-LABEL: func.func @seq_for_no_routine_no_in_loop_barrier
  // CHECK:         scf.for
  // CHECK-NEXT:      memref.store
  // CHECK-NOT:       gpu.barrier
  func.func @seq_for_no_routine_no_in_loop_barrier() {
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %bx = acc.par_width %c4 par_dim(#acc.par_dim<block_x>)
    %tx = acc.par_width %c32 par_dim(#acc.par_dim<thread_x>)
    %ty = acc.par_width %c4 par_dim(#acc.par_dim<thread_y>)
    %priv = acc.privatize : () -> !acc.private_type<memref<i32>>
    acc.compute_region launch(%kbx = %bx, %ktx = %tx, %kty = %ty)
        ins(%arg10 = %priv)
        : (!acc.private_type<memref<i32>>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      scf.parallel (%g) = (%c0) to (%kbx) step (%c1) {
        %pl = acc.private_local %arg10 : (!acc.private_type<memref<i32>>) -> memref<i32>
        scf.for %i = %c0 to %c2 step %c1 {
          memref.store %c0_i32, %pl[] : memref<i32>
          %v = memref.load %pl[] : memref<i32>
        }
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      acc.yield
    } <{origin = "acc.parallel"}>
    return
  }

  // Sequential scf.parallel is the remainder of a partitioned gang loop, not
  // `acc loop seq`. Scalar stores already reconverge; do not put a reuse
  // barrier at the top of that remnant.
  // CHECK-LABEL: func.func @seq_parallel_remnant_worker_call_no_in_loop_barrier
  // CHECK:         scf.parallel
  // CHECK-NEXT:      memref.store
  // CHECK-NEXT:      func.call @worker_routine
  func.func @seq_parallel_remnant_worker_call_no_in_loop_barrier() {
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %bx = acc.par_width %c4 par_dim(#acc.par_dim<block_x>)
    %tx = acc.par_width %c32 par_dim(#acc.par_dim<thread_x>)
    %ty = acc.par_width %c4 par_dim(#acc.par_dim<thread_y>)
    %priv = acc.privatize : () -> !acc.private_type<memref<i32>>
    acc.compute_region launch(%kbx = %bx, %ktx = %tx, %kty = %ty)
        ins(%arg10 = %priv)
        : (!acc.private_type<memref<i32>>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      scf.parallel (%g) = (%c0) to (%kbx) step (%c1) {
        scf.parallel (%rem) = (%g) to (%c8) step (%kbx) {
          %pl = acc.private_local %arg10 : (!acc.private_type<memref<i32>>) -> memref<i32>
          memref.store %c0_i32, %pl[] : memref<i32>
          func.call @worker_routine(%pl) : (memref<i32>) -> ()
          scf.reduce
        } {acc.par_dims = #acc<par_dims[sequential]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      acc.yield
    } <{origin = "acc.parallel"}>
    return
  }

  // A workgroup barrier inside `scf.if` would hang threads that skip the then
  // block, so the in-loop reuse barrier is not emitted there.
  // CHECK-LABEL: func.func @seq_for_worker_call_inside_if_no_in_loop_barrier
  // CHECK:         scf.if
  // CHECK:           scf.for
  // CHECK-NEXT:        memref.store
  // CHECK-NEXT:        func.call @worker_routine
  func.func @seq_for_worker_call_inside_if_no_in_loop_barrier() {
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %bx = acc.par_width %c4 par_dim(#acc.par_dim<block_x>)
    %tx = acc.par_width %c32 par_dim(#acc.par_dim<thread_x>)
    %ty = acc.par_width %c4 par_dim(#acc.par_dim<thread_y>)
    %priv = acc.privatize : () -> !acc.private_type<memref<i32>>
    acc.compute_region launch(%kbx = %bx, %ktx = %tx, %kty = %ty)
        ins(%arg10 = %priv)
        : (!acc.private_type<memref<i32>>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      scf.parallel (%g) = (%c0) to (%kbx) step (%c1) {
        %pl = acc.private_local %arg10 : (!acc.private_type<memref<i32>>) -> memref<i32>
        %cond = arith.cmpi eq, %g, %c0 : index
        scf.if %cond {
          scf.for %i = %c0 to %c2 step %c1 {
            memref.store %c0_i32, %pl[] : memref<i32>
            func.call @worker_routine(%pl) : (memref<i32>) -> ()
          }
        }
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      acc.yield
    } <{origin = "acc.parallel"}>
    return
  }

  // Thread-level worksharing ancestors have per-thread trip counts; a
  // workgroup barrier in the seq body would deadlock.
  // CHECK-LABEL: func.func @seq_for_worker_call_under_thread_y_no_in_loop_barrier
  // CHECK:         scf.for
  // CHECK-NEXT:      memref.store
  // CHECK-NEXT:      func.call @worker_routine
  func.func @seq_for_worker_call_under_thread_y_no_in_loop_barrier() {
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %bx = acc.par_width %c4 par_dim(#acc.par_dim<block_x>)
    %tx = acc.par_width %c32 par_dim(#acc.par_dim<thread_x>)
    %ty = acc.par_width %c4 par_dim(#acc.par_dim<thread_y>)
    %priv = acc.privatize : () -> !acc.private_type<memref<i32>>
    acc.compute_region launch(%kbx = %bx, %ktx = %tx, %kty = %ty)
        ins(%arg10 = %priv)
        : (!acc.private_type<memref<i32>>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      scf.parallel (%g) = (%c0) to (%kbx) step (%c1) {
        scf.parallel (%w) = (%c0) to (%kty) step (%c1) {
          %pl = acc.private_local %arg10 : (!acc.private_type<memref<i32>>) -> memref<i32>
          scf.for %i = %c0 to %c2 step %c1 {
            memref.store %c0_i32, %pl[] : memref<i32>
            func.call @worker_routine(%pl) : (memref<i32>) -> ()
          }
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_y]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      acc.yield
    } <{origin = "acc.parallel"}>
    return
  }
}
