// RUN: mlir-opt %s -acc-compute-lowering | FileCheck %s

// Test that a collapsed acc.loop with 2 IVs converts correctly to
// scf.parallel with 2 dimensions and no redundant inner loops.

// CHECK-LABEL: func.func @collapse_two_ivs
// CHECK:       acc.compute_region
// CHECK:       scf.parallel
// CHECK-NOT:   scf.for
// CHECK:       scf.reduce
func.func @collapse_two_ivs(%buf: memref<1xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index

  %dev = acc.copyin varPtr(%buf : memref<1xf32>) -> memref<1xf32>
  acc.parallel dataOperands(%dev : memref<1xf32>) {
    acc.loop combined(parallel) control(%i : index, %j : index) = (%c1, %c1 : index, index) to (%c100, %c200 : index, index) step (%c1, %c1 : index, index) {
      %vi = arith.index_cast %i : index to i32
      %vj = arith.index_cast %j : index to i32
      %sum = arith.addi %vi, %vj : i32
      %valf = arith.sitofp %sum : i32 to f32
      memref.store %valf, %dev[%c0] : memref<1xf32>
      acc.yield
    } attributes {collapse = [2], collapseDeviceType = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true, true>, independent = [#acc.device_type<none>]}
    acc.yield
  }
  acc.copyout accPtr(%dev : memref<1xf32>) to varPtr(%buf : memref<1xf32>)
  return
}
