// RUN: mlir-opt %s -split-input-file -acc-compute-lowering -acc-emit-remarks-loop --remarks-filter="(open)?acc.*" 2>&1 | FileCheck %s

// CHECK: remark: [Passed] openacc | Category:acc-emit-remarks-loop | Function=parallel_gang_loop | Remark="!$acc loop gang(10) ! blockidx.x"
func.func @parallel_gang_loop(%buf: memref<1xi32>) {
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  %c100_i32 = arith.constant 100 : i32

  %dev = acc.copyin varPtr(%buf : memref<1xi32>) -> memref<1xi32>
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

// CHECK: remark: [Passed] openacc | Category:acc-emit-remarks-loop | Function=parallel_loop_auto_collapse | Remark="!$acc loop sequential collapse(2)"
func.func @parallel_loop_auto_collapse(%buf: memref<1xi32>, %lb0 : index, %ub0 : index, %lb1 : index, %ub1 : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %dev = acc.copyin varPtr(%buf : memref<1xi32>) -> memref<1xi32>
  acc.parallel dataOperands(%dev : memref<1xi32>) {
    acc.loop control(%i : index, %j : index) = (%lb0, %lb1 : index, index) to (%ub0, %ub1 : index, index) step (%c1, %c1 : index, index) {
      %vi = arith.index_cast %i : index to i32
      memref.store %vi, %dev[%c0] : memref<1xi32>
      acc.yield
    } attributes {auto_ = [#acc.device_type<none>]}
    acc.yield
  }
  acc.copyout accPtr(%dev : memref<1xi32>) to varPtr(%buf : memref<1xi32>)
  return
}
