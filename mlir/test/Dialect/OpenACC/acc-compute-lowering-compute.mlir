// RUN: mlir-opt %s -acc-compute-lowering | FileCheck %s

// CHECK-LABEL: func.func @parallel_gang_loop
func.func @parallel_gang_loop(%buf: memref<1xi32>) {
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  %c100_i32 = arith.constant 100 : i32

  %dev = acc.copyin varPtr(%buf : memref<1xi32>) -> memref<1xi32>
  // CHECK-NOT: acc.parallel
  // CHECK: acc.kernel_environment
  // CHECK: acc.par_width {{.*}} {par_dim = #acc.par_dim<block_x>}
  // CHECK: acc.compute_region launch(
  // CHECK: scf.parallel
  // CHECK: acc.par_dims = #acc<par_dims[block_x]>
  acc.parallel num_gangs({%c10_i32 : i32}) dataOperands(%dev : memref<1xi32>) {
    acc.loop gang control(%arg0 : i32) = (%c1_i32 : i32) to (%c100_i32 : i32) step (%c1_i32 : i32) {
      memref.store %arg0, %dev[%c0] : memref<1xi32>
      acc.yield
    } attributes {independent = [#acc.device_type<none>]}
    acc.yield
  }
  acc.copyout accPtr(%dev : memref<1xi32>) to varPtr(%buf : memref<1xi32>)
  return
}

// -----

// CHECK-LABEL: func.func @parallel_seq_loop
func.func @parallel_seq_loop(%buf: memref<4xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c10_i32 = arith.constant 10 : i32

  %dev = acc.copyin varPtr(%buf : memref<4xi32>) -> memref<4xi32>
  // CHECK-NOT: acc.parallel
  // CHECK: acc.kernel_environment
  // CHECK: acc.par_width {{.*}} {par_dim = #acc.par_dim<block_x>}
  // CHECK: acc.compute_region launch(
  // CHECK: scf.parallel
  // CHECK: acc.par_dims = #acc<par_dims[sequential]>
  acc.parallel num_gangs({%c10_i32 : i32}) dataOperands(%dev : memref<4xi32>) {
    acc.loop control(%i : index) = (%c0 : index) to (%c4 : index) step (%c1 : index) {
      %vi = arith.index_cast %i : index to i32
      memref.store %vi, %dev[%i] : memref<4xi32>
      acc.yield
    } attributes {seq = [#acc.device_type<none>]}
    acc.yield
  }
  acc.copyout accPtr(%dev : memref<4xi32>) to varPtr(%buf : memref<4xi32>)
  return
}

// -----

// CHECK-LABEL: func.func @serial_loop
func.func @serial_loop(%buf: memref<4xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index

  %dev = acc.copyin varPtr(%buf : memref<4xi32>) -> memref<4xi32>
  // CHECK-NOT: acc.serial
  // CHECK: acc.kernel_environment
  // CHECK: acc.par_width {par_dim = #acc.par_dim<sequential>}
  // CHECK: acc.compute_region launch(
  // CHECK: scf.parallel
  // CHECK: acc.par_dims = #acc<par_dims[sequential]>
  acc.serial dataOperands(%dev : memref<4xi32>) {
    acc.loop control(%i : index) = (%c0 : index) to (%c4 : index) step (%c1 : index) {
      %vi = arith.index_cast %i : index to i32
      memref.store %vi, %dev[%i] : memref<4xi32>
      acc.yield
    } attributes {independent = [#acc.device_type<none>]}
    acc.yield
  }
  acc.copyout accPtr(%dev : memref<4xi32>) to varPtr(%buf : memref<4xi32>)
  return
}

// -----

// CHECK-LABEL: func.func @kernels_loop
func.func @kernels_loop(%buf: memref<8xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index

  %dev = acc.copyin varPtr(%buf : memref<8xi32>) -> memref<8xi32>
  // CHECK-NOT: acc.kernels
  // CHECK: acc.kernel_environment
  // CHECK-NOT: acc.par_width
  // CHECK: acc.compute_region
  // CHECK: scf.parallel
  acc.kernels dataOperands(%dev : memref<8xi32>) {
    acc.loop control(%i : index) = (%c0 : index) to (%c8 : index) step (%c1 : index) {
      %vi = arith.index_cast %i : index to i32
      memref.store %vi, %dev[%i] : memref<8xi32>
      acc.yield
    } attributes {independent = [#acc.device_type<none>]}
    acc.terminator
  }
  acc.copyout accPtr(%dev : memref<8xi32>) to varPtr(%buf : memref<8xi32>)
  return
}

// -----

// Constant live-ins are cloned into the compute region body so they are not
// passed through `acc.compute_region` arguments.

// CHECK-LABEL: func.func @constant_livein_materialized_into_compute_region
func.func @constant_livein_materialized_into_compute_region(%buf: memref<1xi32>) {
  %c0 = arith.constant 0 : index
  %c42 = arith.constant 42 : i32
  %dev = acc.copyin varPtr(%buf : memref<1xi32>) -> memref<1xi32>
  // CHECK: acc.kernel_environment
  // CHECK: acc.par_width {par_dim = #acc.par_dim<sequential>}
  // CHECK: acc.compute_region launch(
  // CHECK-SAME: ins({{.*}}) : (memref<1xi32>) {
  // CHECK-DAG: arith.constant 42 : i32
  // CHECK-DAG: arith.constant 0 : index
  // CHECK: memref.store
  // CHECK: acc.yield
  acc.serial dataOperands(%dev : memref<1xi32>) {
    memref.store %c42, %dev[%c0] : memref<1xi32>
    acc.yield
  }
  acc.copyout accPtr(%dev : memref<1xi32>) to varPtr(%buf : memref<1xi32>)
  return
}

// -----

// acc.parallel with num_gangs(1), num_workers(1), and vector_length(1) is
// treated like acc.serial: sequential acc.par_width launch args and sequential
// par_dims on lowered loops.

// CHECK-LABEL: func.func @parallel_unit_launch_serial_loops
func.func @parallel_unit_launch_serial_loops(%buf: memref<4xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c1_i32 = arith.constant 1 : i32

  %dev = acc.copyin varPtr(%buf : memref<4xi32>) -> memref<4xi32>
  // CHECK-NOT: acc.parallel
  // CHECK: acc.kernel_environment
  // CHECK: acc.par_width {par_dim = #acc.par_dim<sequential>}
  // CHECK: acc.compute_region launch(
  // CHECK: scf.parallel
  // CHECK: acc.par_dims = #acc<par_dims[sequential]>
  acc.parallel num_gangs({%c1_i32 : i32}) num_workers(%c1_i32 : i32) vector_length(%c1_i32 : i32) dataOperands(%dev : memref<4xi32>) {
    acc.loop control(%i : index) = (%c0 : index) to (%c4 : index) step (%c1 : index) {
      %vi = arith.index_cast %i : index to i32
      memref.store %vi, %dev[%i] : memref<4xi32>
      acc.yield
    } attributes {independent = [#acc.device_type<none>]}
    acc.yield
  }
  acc.copyout accPtr(%dev : memref<4xi32>) to varPtr(%buf : memref<4xi32>)
  return
}

// -----

// acc.kernels with num_gangs(1), num_workers(1), and vector_length(1) is
// treated like acc.serial: sequential acc.par_width launch args and sequential
// par_dims on lowered loops.

// CHECK-LABEL: func.func @kernels_unit_launch_serial_loops
func.func @kernels_unit_launch_serial_loops(%buf: memref<4xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c1_i32 = arith.constant 1 : i32

  %dev = acc.copyin varPtr(%buf : memref<4xi32>) -> memref<4xi32>
  // CHECK-NOT: acc.kernels
  // CHECK: acc.kernel_environment
  // CHECK: acc.par_width {par_dim = #acc.par_dim<sequential>}
  // CHECK: acc.compute_region launch(
  // CHECK: scf.parallel
  // CHECK: acc.par_dims = #acc<par_dims[sequential]>
  acc.kernels num_gangs({%c1_i32 : i32}) num_workers(%c1_i32 : i32) vector_length(%c1_i32 : i32) dataOperands(%dev : memref<4xi32>) {
    acc.loop control(%i : index) = (%c0 : index) to (%c4 : index) step (%c1 : index) {
      %vi = arith.index_cast %i : index to i32
      memref.store %vi, %dev[%i] : memref<4xi32>
      acc.yield
    } attributes {independent = [#acc.device_type<none>]}
    acc.terminator
  }
  acc.copyout accPtr(%dev : memref<4xi32>) to varPtr(%buf : memref<4xi32>)
  return
}

// -----

// CHECK-LABEL: func.func @parallel_vector_length32_independent
func.func @parallel_vector_length32_independent(%buf: memref<4xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32

  %dev = acc.copyin varPtr(%buf : memref<4xi32>) -> memref<4xi32>
  // CHECK-NOT: acc.par_dims = #acc<par_dims[sequential]>
  // CHECK: acc.par_dims = #acc<par_dims[thread_x]>
  acc.parallel num_gangs({%c1_i32 : i32}) num_workers(%c1_i32 : i32) vector_length(%c32_i32 : i32) dataOperands(%dev : memref<4xi32>) {
    acc.loop control(%i : index) = (%c0 : index) to (%c4 : index) step (%c1 : index) {
      %vi = arith.index_cast %i : index to i32
      memref.store %vi, %dev[%i] : memref<4xi32>
      acc.yield
    } attributes {independent = [#acc.device_type<none>], vector = [#acc.device_type<none>]}
    acc.yield
  }
  acc.copyout accPtr(%dev : memref<4xi32>) to varPtr(%buf : memref<4xi32>)
  return
}

// -----

// CHECK-LABEL: func.func @kernels_num_gangs4_independent
func.func @kernels_num_gangs4_independent(%buf: memref<4xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32

  %dev = acc.copyin varPtr(%buf : memref<4xi32>) -> memref<4xi32>
  // CHECK-NOT: acc.par_dims = #acc<par_dims[sequential]>
  // CHECK: acc.par_dims = #acc<par_dims[thread_x]>
  acc.kernels num_gangs({%c4_i32 : i32}) num_workers(%c1_i32 : i32) vector_length(%c1_i32 : i32) dataOperands(%dev : memref<4xi32>) {
    acc.loop control(%i : index) = (%c0 : index) to (%c4 : index) step (%c1 : index) {
      %vi = arith.index_cast %i : index to i32
      memref.store %vi, %dev[%i] : memref<4xi32>
      acc.yield
    } attributes {independent = [#acc.device_type<none>], vector = [#acc.device_type<none>]}
    acc.terminator
  }
  acc.copyout accPtr(%dev : memref<4xi32>) to varPtr(%buf : memref<4xi32>)
  return
}

// -----

// CHECK-LABEL: func.func @parallel_num_gangs_1_2_independent
func.func @parallel_num_gangs_1_2_independent(%buf: memref<4xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32

  %dev = acc.copyin varPtr(%buf : memref<4xi32>) -> memref<4xi32>
  // CHECK-NOT: acc.par_dims = #acc<par_dims[sequential]>
  // CHECK: acc.par_dims = #acc<par_dims[thread_x]>
  acc.parallel num_gangs({%c1_i32 : i32, %c2_i32 : i32}) num_workers(%c1_i32 : i32) vector_length(%c1_i32 : i32) dataOperands(%dev : memref<4xi32>) {
    acc.loop control(%i : index) = (%c0 : index) to (%c4 : index) step (%c1 : index) {
      %vi = arith.index_cast %i : index to i32
      memref.store %vi, %dev[%i] : memref<4xi32>
      acc.yield
    } attributes {independent = [#acc.device_type<none>], vector = [#acc.device_type<none>]}
    acc.yield
  }
  acc.copyout accPtr(%dev : memref<4xi32>) to varPtr(%buf : memref<4xi32>)
  return
}
