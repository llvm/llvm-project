// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu{device-type=nvidia}))" --split-input-file | FileCheck %s

// A sequential tag does not make thread-dependent trip counts uniform.
// CHECK-LABEL: func.func @shared_under_divergent_parallel
// CHECK: gpu.launch
// CHECK: memref.store
// CHECK-NEXT: }
// CHECK-NEXT: %{{.*}} = gpu.block_dim x
// CHECK-NEXT: %{{.*}} = gpu.block_dim y
// CHECK: nvvm.barrier id = %{{.*}} number_of_threads = %{{.*}} aligned = false
// CHECK-NOT: gpu.barrier
// CHECK: gpu.terminator

func.func @shared_under_divergent_parallel(%arg0: memref<4xi32>) {
  %0 = acc.copyin varPtr(%arg0 : memref<4xi32>) dataClause(acc_reduction) implicit(true) name("r") -> memref<4xi32>
  acc.kernel_environment dataOperands(%0 : memref<4xi32>) {
    %c1_pw = arith.constant 1 : index
    %c4_pw = arith.constant 4 : index
    %c64_pw = arith.constant 64 : index
    %bx = acc.par_width %c1_pw par_dim(#acc.par_dim<block_x>)
    %wy = acc.par_width %c4_pw par_dim(#acc.par_dim<thread_y>)
    %tx = acc.par_width %c64_pw par_dim(#acc.par_dim<thread_x>)
    %private = acc.privatize par_dims(#acc<par_dims[block_x, thread_y]>) : () -> !acc.private_type<memref<4xi32>>
    acc.compute_region launch(%kbx = %bx, %kwy = %wy, %ktx = %tx) ins(%arg2 = %0, %priv = %private) : (memref<4xi32>, !acc.private_type<memref<4xi32>>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0_i32 = arith.constant 0 : i32
      %local = acc.private_local %priv {acc.active_par_dims = #acc<active_par_dims[block_x]>, acc.par_dims = #acc<par_dims[block_x, thread_y]>} : (!acc.private_type<memref<4xi32>>) -> memref<4xi32>
      %row = gpu.thread_id y
      %ub = arith.addi %row, %c1 : index
      scf.parallel (%i) = (%c0) to (%ub) step (%c1) {
        acc.predicate_region {
          memref.store %c0_i32, %local[%c0] : memref<4xi32>
        } {acc.active_par_dims = #acc<active_par_dims[block_x, thread_y]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[sequential]>}
      acc.yield
    } <{origin = "acc.parallel"}>
  }
  acc.copyout accPtr(%0 : memref<4xi32>) to varPtr(%arg0 : memref<4xi32>) dataClause(acc_reduction) implicit(true) name("r")
  return
}

// -----

// Some worker rows skip the enclosing branch entirely.
// CHECK-LABEL: func.func @shared_under_divergent_if
// CHECK: gpu.launch
// CHECK: memref.store
// CHECK-NEXT: }
// CHECK-NEXT: %{{.*}} = gpu.block_dim x
// CHECK-NEXT: %{{.*}} = gpu.block_dim y
// CHECK: nvvm.barrier id = %{{.*}} number_of_threads = %{{.*}} aligned = false
// CHECK-NOT: gpu.barrier
// CHECK: gpu.terminator

func.func @shared_under_divergent_if(%arg0: memref<4xi32>) {
  %0 = acc.copyin varPtr(%arg0 : memref<4xi32>) dataClause(acc_reduction) implicit(true) name("r") -> memref<4xi32>
  acc.kernel_environment dataOperands(%0 : memref<4xi32>) {
    %c1_pw = arith.constant 1 : index
    %c4_pw = arith.constant 4 : index
    %c64_pw = arith.constant 64 : index
    %bx = acc.par_width %c1_pw par_dim(#acc.par_dim<block_x>)
    %wy = acc.par_width %c4_pw par_dim(#acc.par_dim<thread_y>)
    %tx = acc.par_width %c64_pw par_dim(#acc.par_dim<thread_x>)
    %private = acc.privatize par_dims(#acc<par_dims[block_x, thread_y]>) : () -> !acc.private_type<memref<4xi32>>
    acc.compute_region launch(%kbx = %bx, %kwy = %wy, %ktx = %tx) ins(%arg2 = %0, %priv = %private) : (memref<4xi32>, !acc.private_type<memref<4xi32>>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0_i32 = arith.constant 0 : i32
      %local = acc.private_local %priv {acc.active_par_dims = #acc<active_par_dims[block_x]>, acc.par_dims = #acc<par_dims[block_x, thread_y]>} : (!acc.private_type<memref<4xi32>>) -> memref<4xi32>
      %row = gpu.thread_id y
      %cond = arith.cmpi eq, %row, %c0 : index
      scf.if %cond {
        acc.predicate_region {
          memref.store %c0_i32, %local[%c0] : memref<4xi32>
        } {acc.active_par_dims = #acc<active_par_dims[block_x, thread_y]>}
      }
      acc.yield
    } <{origin = "acc.parallel"}>
  }
  acc.copyout accPtr(%0 : memref<4xi32>) to varPtr(%arg0 : memref<4xi32>) dataClause(acc_reduction) implicit(true) name("r")
  return
}

// -----

// Ordinary loops can also have different trip counts on each row.
// CHECK-LABEL: func.func @shared_under_divergent_for
// CHECK: gpu.launch
// CHECK: memref.store
// CHECK-NEXT: }
// CHECK-NEXT: %{{.*}} = gpu.block_dim x
// CHECK-NEXT: %{{.*}} = gpu.block_dim y
// CHECK: nvvm.barrier id = %{{.*}} number_of_threads = %{{.*}} aligned = false
// CHECK-NOT: gpu.barrier
// CHECK: gpu.terminator

func.func @shared_under_divergent_for(%arg0: memref<4xi32>) {
  %0 = acc.copyin varPtr(%arg0 : memref<4xi32>) dataClause(acc_reduction) implicit(true) name("r") -> memref<4xi32>
  acc.kernel_environment dataOperands(%0 : memref<4xi32>) {
    %c1_pw = arith.constant 1 : index
    %c4_pw = arith.constant 4 : index
    %c64_pw = arith.constant 64 : index
    %bx = acc.par_width %c1_pw par_dim(#acc.par_dim<block_x>)
    %wy = acc.par_width %c4_pw par_dim(#acc.par_dim<thread_y>)
    %tx = acc.par_width %c64_pw par_dim(#acc.par_dim<thread_x>)
    %private = acc.privatize par_dims(#acc<par_dims[block_x, thread_y]>) : () -> !acc.private_type<memref<4xi32>>
    acc.compute_region launch(%kbx = %bx, %kwy = %wy, %ktx = %tx) ins(%arg2 = %0, %priv = %private) : (memref<4xi32>, !acc.private_type<memref<4xi32>>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0_i32 = arith.constant 0 : i32
      %local = acc.private_local %priv {acc.active_par_dims = #acc<active_par_dims[block_x]>, acc.par_dims = #acc<par_dims[block_x, thread_y]>} : (!acc.private_type<memref<4xi32>>) -> memref<4xi32>
      %row = gpu.thread_id y
      %ub = arith.addi %row, %c1 : index
      scf.for %i = %c0 to %ub step %c1 {
        acc.predicate_region {
          memref.store %c0_i32, %local[%c0] : memref<4xi32>
        } {acc.active_par_dims = #acc<active_par_dims[block_x, thread_y]>}
      }
      acc.yield
    } <{origin = "acc.parallel"}>
  }
  acc.copyout accPtr(%0 : memref<4xi32>) to varPtr(%arg0 : memref<4xi32>) dataClause(acc_reduction) implicit(true) name("r")
  return
}

// -----

// A scalar loaded before the region does not access its source storage inside it.
// CHECK-LABEL: func.func @external_scalar_load
// CHECK: gpu.launch
// CHECK: memref.store
// CHECK-NEXT: }
// CHECK-NEXT: %{{.*}} = gpu.block_dim x
// CHECK-NEXT: %{{.*}} = gpu.block_dim y
// CHECK: nvvm.barrier id = %{{.*}} number_of_threads = %{{.*}} aligned = false
// CHECK-NOT: gpu.barrier
// CHECK: gpu.terminator

func.func @external_scalar_load(%arg0: memref<4xi32>) {
  %0 = acc.copyin varPtr(%arg0 : memref<4xi32>) dataClause(acc_reduction) implicit(true) name("r") -> memref<4xi32>
  acc.kernel_environment dataOperands(%0 : memref<4xi32>) {
    %c1_pw = arith.constant 1 : index
    %c4_pw = arith.constant 4 : index
    %c64_pw = arith.constant 64 : index
    %bx = acc.par_width %c1_pw par_dim(#acc.par_dim<block_x>)
    %wy = acc.par_width %c4_pw par_dim(#acc.par_dim<thread_y>)
    %tx = acc.par_width %c64_pw par_dim(#acc.par_dim<thread_x>)
    %private = acc.privatize par_dims(#acc<par_dims[block_x, thread_y]>) : () -> !acc.private_type<memref<4xi32>>
    acc.compute_region launch(%kbx = %bx, %kwy = %wy, %ktx = %tx) ins(%arg2 = %0, %priv = %private) : (memref<4xi32>, !acc.private_type<memref<4xi32>>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0_i32 = arith.constant 0 : i32
      %local = acc.private_local %priv {acc.active_par_dims = #acc<active_par_dims[block_x]>, acc.par_dims = #acc<par_dims[block_x, thread_y]>} : (!acc.private_type<memref<4xi32>>) -> memref<4xi32>
      %scalar = memref.load %local[%c0] : memref<4xi32>
      acc.predicate_region {
        memref.store %scalar, %arg2[%c0] : memref<4xi32>
      } {acc.active_par_dims = #acc<active_par_dims[block_x, thread_y]>}
      acc.yield
    } <{origin = "acc.parallel"}>
  }
  acc.copyout accPtr(%0 : memref<4xi32>) to varPtr(%arg0 : memref<4xi32>) dataClause(acc_reduction) implicit(true) name("r")
  return
}

// -----

// Either select arm can name storage shared across worker rows.
// CHECK-LABEL: func.func @selected_shared_true
// CHECK: gpu.launch
// CHECK: memref.store
// CHECK-NEXT: }
// CHECK-NEXT: gpu.barrier{{$}}
// CHECK-NOT: nvvm.barrier
// CHECK: gpu.terminator

func.func @selected_shared_true(%arg0: memref<4xi32>) {
  %0 = acc.copyin varPtr(%arg0 : memref<4xi32>) dataClause(acc_reduction) implicit(true) name("r") -> memref<4xi32>
  acc.kernel_environment dataOperands(%0 : memref<4xi32>) {
    %c1_pw = arith.constant 1 : index
    %c4_pw = arith.constant 4 : index
    %c64_pw = arith.constant 64 : index
    %bx = acc.par_width %c1_pw par_dim(#acc.par_dim<block_x>)
    %wy = acc.par_width %c4_pw par_dim(#acc.par_dim<thread_y>)
    %tx = acc.par_width %c64_pw par_dim(#acc.par_dim<thread_x>)
    %private = acc.privatize par_dims(#acc<par_dims[block_x, thread_y]>) : () -> !acc.private_type<memref<4xi32>>
    acc.compute_region launch(%kbx = %bx, %kwy = %wy, %ktx = %tx) ins(%arg2 = %0, %priv = %private) : (memref<4xi32>, !acc.private_type<memref<4xi32>>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0_i32 = arith.constant 0 : i32
      %local = acc.private_local %priv {acc.active_par_dims = #acc<active_par_dims[block_x]>, acc.par_dims = #acc<par_dims[block_x, thread_y]>} : (!acc.private_type<memref<4xi32>>) -> memref<4xi32>
      %row_local = acc.private_local %priv {acc.active_par_dims = #acc<active_par_dims[block_x, thread_y]>, acc.par_dims = #acc<par_dims[block_x, thread_y]>} : (!acc.private_type<memref<4xi32>>) -> memref<4xi32>
      %row = gpu.thread_id y
      %cond = arith.cmpi eq, %row, %c0 : index
      %selected = arith.select %cond, %local, %row_local : memref<4xi32>
      acc.predicate_region {
        memref.store %c0_i32, %selected[%c0] : memref<4xi32>
      } {acc.active_par_dims = #acc<active_par_dims[block_x, thread_y]>}
      acc.yield
    } <{origin = "acc.parallel"}>
  }
  acc.copyout accPtr(%0 : memref<4xi32>) to varPtr(%arg0 : memref<4xi32>) dataClause(acc_reduction) implicit(true) name("r")
  return
}

// -----

// Either select arm can name storage shared across worker rows.
// CHECK-LABEL: func.func @selected_shared_false
// CHECK: gpu.launch
// CHECK: memref.store
// CHECK-NEXT: }
// CHECK-NEXT: gpu.barrier{{$}}
// CHECK-NOT: nvvm.barrier
// CHECK: gpu.terminator

func.func @selected_shared_false(%arg0: memref<4xi32>) {
  %0 = acc.copyin varPtr(%arg0 : memref<4xi32>) dataClause(acc_reduction) implicit(true) name("r") -> memref<4xi32>
  acc.kernel_environment dataOperands(%0 : memref<4xi32>) {
    %c1_pw = arith.constant 1 : index
    %c4_pw = arith.constant 4 : index
    %c64_pw = arith.constant 64 : index
    %bx = acc.par_width %c1_pw par_dim(#acc.par_dim<block_x>)
    %wy = acc.par_width %c4_pw par_dim(#acc.par_dim<thread_y>)
    %tx = acc.par_width %c64_pw par_dim(#acc.par_dim<thread_x>)
    %private = acc.privatize par_dims(#acc<par_dims[block_x, thread_y]>) : () -> !acc.private_type<memref<4xi32>>
    acc.compute_region launch(%kbx = %bx, %kwy = %wy, %ktx = %tx) ins(%arg2 = %0, %priv = %private) : (memref<4xi32>, !acc.private_type<memref<4xi32>>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0_i32 = arith.constant 0 : i32
      %local = acc.private_local %priv {acc.active_par_dims = #acc<active_par_dims[block_x]>, acc.par_dims = #acc<par_dims[block_x, thread_y]>} : (!acc.private_type<memref<4xi32>>) -> memref<4xi32>
      %row_local = acc.private_local %priv {acc.active_par_dims = #acc<active_par_dims[block_x, thread_y]>, acc.par_dims = #acc<par_dims[block_x, thread_y]>} : (!acc.private_type<memref<4xi32>>) -> memref<4xi32>
      %row = gpu.thread_id y
      %cond = arith.cmpi eq, %row, %c0 : index
      %selected = arith.select %cond, %row_local, %local : memref<4xi32>
      acc.predicate_region {
        memref.store %c0_i32, %selected[%c0] : memref<4xi32>
      } {acc.active_par_dims = #acc<active_par_dims[block_x, thread_y]>}
      acc.yield
    } <{origin = "acc.parallel"}>
  }
  acc.copyout accPtr(%0 : memref<4xi32>) to varPtr(%arg0 : memref<4xi32>) dataClause(acc_reduction) implicit(true) name("r")
  return
}

// -----

// Missing active dimensions must use the same inference as allocation lowering.
// CHECK-LABEL: func.func @inferred_worker_shared
// CHECK: gpu.launch
// CHECK: memref.store
// CHECK-NEXT: }
// CHECK-NEXT: gpu.barrier{{$}}
// CHECK-NOT: nvvm.barrier
// CHECK: gpu.terminator

func.func @inferred_worker_shared(%arg0: memref<4xi32>) {
  %0 = acc.copyin varPtr(%arg0 : memref<4xi32>) dataClause(acc_reduction) implicit(true) name("r") -> memref<4xi32>
  acc.kernel_environment dataOperands(%0 : memref<4xi32>) {
    %c1_pw = arith.constant 1 : index
    %c4_pw = arith.constant 4 : index
    %c64_pw = arith.constant 64 : index
    %bx = acc.par_width %c1_pw par_dim(#acc.par_dim<block_x>)
    %wy = acc.par_width %c4_pw par_dim(#acc.par_dim<thread_y>)
    %tx = acc.par_width %c64_pw par_dim(#acc.par_dim<thread_x>)
    %private = acc.privatize par_dims(#acc<par_dims[block_x, thread_y]>) : () -> !acc.private_type<memref<4xi32>>
    acc.compute_region launch(%kbx = %bx, %kwy = %wy, %ktx = %tx) ins(%arg2 = %0, %priv = %private) : (memref<4xi32>, !acc.private_type<memref<4xi32>>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0_i32 = arith.constant 0 : i32
      %local = acc.private_local %priv {acc.par_dims = #acc<par_dims[block_x, thread_y]>} : (!acc.private_type<memref<4xi32>>) -> memref<4xi32>
      acc.predicate_region {
        memref.store %c0_i32, %local[%c0] : memref<4xi32>
      } {acc.active_par_dims = #acc<active_par_dims[block_x, thread_y]>}
      acc.yield
    } <{origin = "acc.parallel"}>
  }
  acc.copyout accPtr(%0 : memref<4xi32>) to varPtr(%arg0 : memref<4xi32>) dataClause(acc_reduction) implicit(true) name("r")
  return
}

// -----

// Every thread runs a loop with uniform bounds, so the workgroup barrier is
// still legal there.
// CHECK-LABEL: func.func @shared_under_uniform_for
// CHECK: gpu.launch
// CHECK: scf.for
// CHECK: memref.store
// CHECK: gpu.barrier{{$}}
// CHECK-NOT: nvvm.barrier
// CHECK: gpu.terminator

func.func @shared_under_uniform_for(%arg0: memref<4xi32>) {
  %0 = acc.copyin varPtr(%arg0 : memref<4xi32>) dataClause(acc_reduction) implicit(true) name("r") -> memref<4xi32>
  acc.kernel_environment dataOperands(%0 : memref<4xi32>) {
    %c1_pw = arith.constant 1 : index
    %c4_pw = arith.constant 4 : index
    %c64_pw = arith.constant 64 : index
    %bx = acc.par_width %c1_pw par_dim(#acc.par_dim<block_x>)
    %wy = acc.par_width %c4_pw par_dim(#acc.par_dim<thread_y>)
    %tx = acc.par_width %c64_pw par_dim(#acc.par_dim<thread_x>)
    %private = acc.privatize par_dims(#acc<par_dims[block_x, thread_y]>) : () -> !acc.private_type<memref<4xi32>>
    acc.compute_region launch(%kbx = %bx, %kwy = %wy, %ktx = %tx) ins(%arg2 = %0, %priv = %private) : (memref<4xi32>, !acc.private_type<memref<4xi32>>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0_i32 = arith.constant 0 : i32
      %local = acc.private_local %priv {acc.active_par_dims = #acc<active_par_dims[block_x]>, acc.par_dims = #acc<par_dims[block_x, thread_y]>} : (!acc.private_type<memref<4xi32>>) -> memref<4xi32>
      scf.for %u = %c0 to %c4 step %c1 {
        acc.predicate_region {
          memref.store %c0_i32, %local[%u] : memref<4xi32>
        } {acc.active_par_dims = #acc<active_par_dims[block_x, thread_y]>}
      }
      acc.yield
    } <{origin = "acc.parallel"}>
  }
  acc.copyout accPtr(%0 : memref<4xi32>) to varPtr(%arg0 : memref<4xi32>) dataClause(acc_reduction) implicit(true) name("r")
  return
}
