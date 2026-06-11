// RUN: mlir-opt %s -acc-compute-lowering=device-type=nvidia | FileCheck %s

// Gang on default, vector on nvidia: with device-type=nvidia only vector applies.
// CHECK-LABEL: func.func @parallel_loop_gang_default_vector_nvidia
func.func @parallel_loop_gang_default_vector_nvidia(%buf: memref<1xi32>) {
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  %c100_i32 = arith.constant 100 : i32

  %dev = acc.copyin varPtr(%buf : memref<1xi32>) -> memref<1xi32>
  // CHECK-NOT: acc.par_dims = #acc<par_dims[block_x]>
  // CHECK: acc.par_dims = #acc<par_dims[thread_x]>
  acc.parallel num_gangs({%c10_i32 : i32}) dataOperands(%dev : memref<1xi32>) {
    acc.loop gang control(%arg0 : i32) = (%c1_i32 : i32) to (%c100_i32 : i32) step (%c1_i32 : i32) {
      memref.store %arg0, %dev[%c0] : memref<1xi32>
      acc.yield
    } attributes {auto_ = [#acc.device_type<none>], gang = [#acc.device_type<none>], vector = [#acc.device_type<nvidia>]}
    acc.yield
  }
  acc.copyout accPtr(%dev : memref<1xi32>) to varPtr(%buf : memref<1xi32>)
  return
}
