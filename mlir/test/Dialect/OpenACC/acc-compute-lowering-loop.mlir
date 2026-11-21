// RUN: mlir-opt %s -acc-compute-lowering | FileCheck %s

// CHECK: [[UNROLL:#.*]] = #llvm.loop_unroll<disable = false, full = true>
// CHECK: #loop_annotation = #llvm.loop_annotation<unroll = [[UNROLL]]>

// CHECK-LABEL: func.func @parallel_independent_loop
func.func @parallel_independent_loop(%buf: memref<16xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index

  %dev = acc.copyin varPtr(%buf : memref<16xi32>) -> memref<16xi32>
  // CHECK-NOT: acc.parallel
  // CHECK: acc.kernel_environment
  // CHECK-NOT: acc.par_width
  // CHECK: acc.compute_region
  // CHECK: scf.parallel
  acc.parallel dataOperands(%dev : memref<16xi32>) {
    acc.loop control(%i : index) = (%c0 : index) to (%c16 : index) step (%c1 : index) {
      %vi = arith.index_cast %i : index to i32
      memref.store %vi, %dev[%i] : memref<16xi32>
      acc.yield
    } independent
    acc.yield
  }
  acc.copyout accPtr(%dev : memref<16xi32>) to varPtr(%buf : memref<16xi32>)
  return
}

// -----

// CHECK-LABEL: func.func @parallel_loop_multi_block_body
func.func @parallel_loop_multi_block_body(%buf: memref<4xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index

  %dev = acc.copyin varPtr(%buf : memref<4xi32>) -> memref<4xi32>
  // CHECK-NOT: acc.parallel
  // CHECK: acc.kernel_environment
  // CHECK-NOT: acc.par_width
  // CHECK: acc.compute_region
  // CHECK: scf.parallel
  // CHECK: scf.execute_region
  acc.parallel dataOperands(%dev : memref<4xi32>) {
    acc.loop control(%i : index) = (%c0 : index) to (%c4 : index) step (%c1 : index) {
      %vi = arith.index_cast %i : index to i32
      memref.store %vi, %dev[%i] : memref<4xi32>
      cf.br ^bb1
    ^bb1:
      acc.yield
    } independent
    acc.yield
  }
  acc.copyout accPtr(%dev : memref<4xi32>) to varPtr(%buf : memref<4xi32>)
  return
}

// -----

// CHECK-LABEL: func.func @parallel_loop_auto_collapse
func.func @parallel_loop_auto_collapse(%buf: memref<1xi32>, %lb0 : index, %ub0 : index, %lb1 : index, %ub1 : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %dev = acc.copyin varPtr(%buf : memref<1xi32>) -> memref<1xi32>
  // CHECK-NOT: acc.parallel
  // CHECK: acc.kernel_environment
  // CHECK-NOT: acc.par_width
  // CHECK: acc.compute_region
  // CHECK: scf.for
  // CHECK-NOT: scf.for
  // CHECK-NOT: scf.parallel
  // CHECK: acc.collapse_count = 2 : i64
  acc.parallel dataOperands(%dev : memref<1xi32>) {
    acc.loop control(%i : index, %j : index) = (%lb0, %lb1 : index, index) to (%ub0, %ub1 : index, index) step (%c1, %c1 : index, index) {
      %vi = arith.index_cast %i : index to i32
      memref.store %vi, %dev[%c0] : memref<1xi32>
      acc.yield
    } auto_
    acc.yield
  }
  acc.copyout accPtr(%dev : memref<1xi32>) to varPtr(%buf : memref<1xi32>)
  return
}

// -----

// CHECK-LABEL: func.func @parallel_loop_collapse
func.func @parallel_loop_collapse(%buf: memref<1xi32>, %lb0 : index, %ub0 : index, %lb1 : index, %ub1 : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %dev = acc.copyin varPtr(%buf : memref<1xi32>) -> memref<1xi32>
  // CHECK-NOT: acc.parallel
  // CHECK: acc.kernel_environment
  // CHECK-NOT: acc.par_width
  // CHECK: acc.compute_region
  // CHECK: scf.parallel (%{{.*}}, %{{.*}})
  // CHECK-NOT: scf.for
  // CHECK-NOT: scf.parallel
  // CHECK: acc.collapse_count = 2 : i64
  acc.parallel dataOperands(%dev : memref<1xi32>) {
    acc.loop control(%i : index, %j : index) = (%lb0, %lb1 : index, index) to (%ub0, %ub1 : index, index) step (%c1, %c1 : index, index) {
      %vi = arith.index_cast %i : index to i32
      memref.store %vi, %dev[%c0] : memref<1xi32>
      acc.yield
    } independent
    acc.yield
  }
  acc.copyout accPtr(%dev : memref<1xi32>) to varPtr(%buf : memref<1xi32>)
  return
}

// -----

// CHECK-LABEL: func.func @serial_loop_normalized
func.func @serial_loop_normalized(%buf: memref<1xi32>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index
  %c9 = arith.constant 9 : index

  %dev = acc.copyin varPtr(%buf : memref<1xi32>) -> memref<1xi32>
  // CHECK-NOT: acc.serial
  // CHECK: acc.kernel_environment
  // CHECK: acc.par_width par_dim(#acc.par_dim<sequential>)
  // CHECK: acc.compute_region launch(
  // CHECK: scf.for
  // CHECK-DAG: arith.muli
  // CHECK-DAG: arith.addi
  // CHECK: acc.par_dims = #acc<par_dims[sequential]>
  acc.serial dataOperands(%dev : memref<1xi32>) {
    acc.loop control(%i : index) = (%c5 : index) to (%c9 : index) step (%c2 : index) {
      %vi = arith.index_cast %i : index to i32
      memref.store %vi, %dev[%c0] : memref<1xi32>
      acc.yield
    } independent
    acc.yield
  }
  acc.copyout accPtr(%dev : memref<1xi32>) to varPtr(%buf : memref<1xi32>)
  return
}

// -----

// CHECK-LABEL: func.func @orphan_loop
func.func @orphan_loop(%buf: memref<8xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c0_i32 = arith.constant 0 : i32

  // CHECK-NOT: acc.loop
  // CHECK: scf.for
  // CHECK-NOT: scf.parallel
  acc.loop control(%i : index) = (%c0 : index) to (%c8 : index) step (%c1 : index) {
    memref.store %c0_i32, %buf[%i] : memref<8xi32>
    acc.yield
  } independent
  return
}

// -----

// Loop in specialized seq acc routine: not treated as orphan (scf.for).
acc.routine @routine_with_loop func(@device_routine_with_loop) seq
// CHECK-LABEL: func.func @device_routine_with_loop
// CHECK: attributes {acc.specialized_routine = #acc.specialized_routine<@routine_with_loop, <seq>, "host_routine_with_loop">}
// CHECK-NOT: acc.loop
// CHECK: scf.for
// CHECK: acc.par_dims = #acc<par_dims[sequential]>
// CHECK-NOT: scf.parallel
func.func @device_routine_with_loop(%buf: memref<8xi32>) attributes {acc.specialized_routine = #acc.specialized_routine<@routine_with_loop, <seq>, "host_routine_with_loop">} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c0_i32 = arith.constant 0 : i32

  acc.loop control(%i : index) = (%c0 : index) to (%c8 : index) step (%c1 : index) {
    memref.store %c0_i32, %buf[%i] : memref<8xi32>
    acc.yield
  } independent
  return
}

// -----

// Loop in specialized vector acc routine with vector loop.
acc.routine @routine_vector_with_loop func(@device_routine_vector_with_loop) vector
// CHECK-LABEL: func.func @device_routine_vector_with_loop
// CHECK: attributes {acc.specialized_routine = #acc.specialized_routine<@routine_vector_with_loop, <vector>, "host_routine_vector_with_loop">}
// CHECK-NOT: acc.loop
// CHECK: scf.parallel
// CHECK: acc.par_dims = #acc<par_dims[thread_x]>
// CHECK-NOT: scf.for
func.func @device_routine_vector_with_loop(%buf: memref<8xi32>) attributes {acc.specialized_routine = #acc.specialized_routine<@routine_vector_with_loop, <vector>, "host_routine_vector_with_loop">} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c0_i32 = arith.constant 0 : i32

  acc.loop vector control(%i : index) = (%c0 : index) to (%c8 : index) step (%c1 : index) {
    memref.store %c0_i32, %buf[%i] : memref<8xi32>
    acc.yield
  } independent
  return
}

// -----

// Auto loop with gang: lowered to scf.for with predetermined par_dims.
// CHECK-LABEL: func.func @parallel_loop_auto_gang
func.func @parallel_loop_auto_gang(%buf: memref<1xi32>) {
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  %c100_i32 = arith.constant 100 : i32

  %dev = acc.copyin varPtr(%buf : memref<1xi32>) -> memref<1xi32>
  // CHECK-NOT: acc.parallel
  // CHECK: acc.kernel_environment
  // CHECK: acc.par_width {{.*}} par_dim(#acc.par_dim<block_x>)
  // CHECK: acc.compute_region launch(
  // CHECK: scf.for
  // CHECK-NOT: scf.parallel
  // CHECK: acc.par_dims = #acc<par_dims[block_x]>
  acc.parallel num_gangs({%c10_i32 : i32}) dataOperands(%dev : memref<1xi32>) {
    acc.loop gang control(%arg0 : i32) = (%c1_i32 : i32) to (%c100_i32 : i32) step (%c1_i32 : i32) {
      memref.store %arg0, %dev[%c0] : memref<1xi32>
      acc.yield
    } auto_
    acc.yield
  }
  acc.copyout accPtr(%dev : memref<1xi32>) to varPtr(%buf : memref<1xi32>)
  return
}

// -----

// Preserve llvm.loop_annotation (e.g. from !dir$ unroll) when lowering acc.loop.
// CHECK-LABEL: func.func @orphan_loop_unroll_annotation
// CHECK-NOT: acc.loop
// CHECK: scf.for {{.*}} {
// CHECK: } {llvm.loop_annotation = #loop_annotation}
func.func @orphan_loop_unroll_annotation(%buf: memref<8xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c0_i32 = arith.constant 0 : i32

  acc.loop control(%i : index) = (%c0 : index) to (%c8 : index) step (%c1 : index) {
    memref.store %c0_i32, %buf[%i] : memref<8xi32>
    acc.yield
  } independent attributes {llvm.loop_annotation = #llvm.loop_annotation<unroll = <disable = false, full = true>>} 
  return
}

// -----

// Independent loop with gang(static:N): chunk size and gang par_dims on scf.parallel.
// CHECK-LABEL: func.func @parallel_loop_gang_static
func.func @parallel_loop_gang_static(%buf: memref<1xi32>) {
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32
  %c10_i32 = arith.constant 10 : i32
  %c100_i32 = arith.constant 100 : i32

  %dev = acc.copyin varPtr(%buf : memref<1xi32>) -> memref<1xi32>
  // CHECK-NOT: acc.parallel
  // CHECK: acc.kernel_environment
  // CHECK: acc.compute_region
  // CHECK: scf.parallel
  // CHECK: acc.chunk_size = #acc.chunk_size<32>
  // CHECK-SAME: acc.par_dims = #acc<par_dims[block_x]>
  acc.parallel num_gangs({%c10_i32 : i32}) dataOperands(%dev : memref<1xi32>) {
    acc.loop gang({static=%c32_i32 : i32}) control(%arg0 : i32) = (%c1_i32 : i32) to (%c100_i32 : i32) step (%c1_i32 : i32) {
      memref.store %arg0, %dev[%c0] : memref<1xi32>
      acc.yield
    } independent
    acc.yield
  }
  acc.copyout accPtr(%dev : memref<1xi32>) to varPtr(%buf : memref<1xi32>)
  return
}

// -----

// gang(static:*) is encoded as chunk size -1.
// CHECK-LABEL: func.func @parallel_loop_gang_static_star
func.func @parallel_loop_gang_static_star(%buf: memref<1xi32>) {
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %c_m1 = arith.constant -1 : index
  %c10_i32 = arith.constant 10 : i32
  %c100_i32 = arith.constant 100 : i32

  %dev = acc.copyin varPtr(%buf : memref<1xi32>) -> memref<1xi32>
  // CHECK: scf.parallel
  // CHECK: acc.chunk_size = #acc.chunk_size<-1>
  // CHECK-SAME: acc.par_dims = #acc<par_dims[block_x]>
  acc.parallel num_gangs({%c10_i32 : i32}) dataOperands(%dev : memref<1xi32>) {
    acc.loop gang({static=%c_m1 : index}) control(%arg0 : i32) = (%c1_i32 : i32) to (%c100_i32 : i32) step (%c1_i32 : i32) {
      memref.store %arg0, %dev[%c0] : memref<1xi32>
      acc.yield
    } independent
    acc.yield
  }
  acc.copyout accPtr(%dev : memref<1xi32>) to varPtr(%buf : memref<1xi32>)
  return
}

// -----

// The exclusive upper bound of a descending loop lies below the last
// iteration: 10, 7 and 4 are iterated, so the normalized loop has three trips.
// CHECK-LABEL: func.func @descending_loop_exclusive_bound
func.func @descending_loop_exclusive_bound(%buf: memref<16xi32>) {
  %lb = arith.constant 10 : index
  %ub = arith.constant 1 : index
  %step = arith.constant -3 : index
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[C3:.*]] = arith.constant 3 : index
  // CHECK: scf.for %{{.*}} = %[[C0]] to %[[C3]] step %[[C1]]
  acc.loop control(%i : index) = (%lb : index) to (%ub : index) step (%step : index) {
    %vi = arith.index_cast %i : index to i32
    memref.store %vi, %buf[%i] : memref<16xi32>
    acc.yield
  } independent
  return
}
