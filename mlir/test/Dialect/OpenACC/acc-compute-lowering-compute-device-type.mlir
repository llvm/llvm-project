// RUN: mlir-opt %s -acc-compute-lowering=device-type=nvidia | FileCheck %s

// Default num_gangs, nvidia vector_length: with device-type=nvidia only vector applies.
// CHECK-LABEL: func.func @parallel_default_gangs_nvidia_vector_length
func.func @parallel_default_gangs_nvidia_vector_length(%buf: memref<4xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c4_i32 = arith.constant 4 : i32
  %c32_i32 = arith.constant 32 : i32

  %dev = acc.copyin varPtr(%buf : memref<4xi32>) -> memref<4xi32>
  // CHECK-NOT: acc.par_width {{.*}} {par_dim = #acc.par_dim<block_x>}
  // CHECK: acc.par_width {{.*}} {par_dim = #acc.par_dim<thread_x>}
  acc.parallel num_gangs({%c4_i32 : i32}) vector_length(%c32_i32 : i32 [#acc.device_type<nvidia>]) dataOperands(%dev : memref<4xi32>) {
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
