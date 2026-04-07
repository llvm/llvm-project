// RUN: mlir-opt %s -acc-compute-lowering | FileCheck %s

// Test that collapse(force:2) with a redundant inner acc.loop eliminates
// the inner loop.  The outer acc.loop has 2 control variables (for the
// collapsed dimensions) but the body contains an inner acc.loop that
// re-iterates the second dimension.  The inner loop should be eliminated
// and its IV replaced with the outer's corresponding IV.

// CHECK-LABEL: func.func @collapse_force_with_redundant_inner
// CHECK:       acc.compute_region
// After conversion, the scf.parallel body should NOT contain an scf.for
// CHECK:       scf.parallel
// CHECK-NOT:   scf.for
// CHECK:       scf.reduce
func.func @collapse_force_with_redundant_inner(%buf: memref<1xf32>) {
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
      acc.loop control(%j_inner : index) = (%c1 : index) to (%c200 : index) step (%c1 : index) {
        %vj2 = arith.index_cast %j_inner : index to i32
        %valf2 = arith.sitofp %vj2 : i32 to f32
        memref.store %valf2, %dev[%c0] : memref<1xf32>
        acc.yield
      } attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}
      acc.yield
    } attributes {collapse = [2], collapseDeviceType = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true, true>, independent = [#acc.device_type<none>]}
    acc.yield
  }
  acc.copyout accPtr(%dev : memref<1xf32>) to varPtr(%buf : memref<1xf32>)
  return
}

// -----

// Verify that a collapsed loop without a redundant inner loop still works.
// CHECK-LABEL: func.func @collapse_no_redundant_inner
// CHECK:       acc.compute_region
// CHECK:       scf.parallel
// CHECK-NOT:   scf.for
// CHECK:       scf.reduce
func.func @collapse_no_redundant_inner(%buf: memref<1xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index

  %dev = acc.copyin varPtr(%buf : memref<1xf32>) -> memref<1xf32>
  acc.parallel dataOperands(%dev : memref<1xf32>) {
    acc.loop combined(parallel) control(%i : index, %j : index) = (%c1, %c1 : index, index) to (%c10, %c20 : index, index) step (%c1, %c1 : index, index) {
      %vj = arith.index_cast %j : index to i32
      %valf = arith.sitofp %vj : i32 to f32
      memref.store %valf, %dev[%c0] : memref<1xf32>
      acc.yield
    } attributes {collapse = [2], collapseDeviceType = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true, true>, independent = [#acc.device_type<none>]}
    acc.yield
  }
  acc.copyout accPtr(%dev : memref<1xf32>) to varPtr(%buf : memref<1xf32>)
  return
}
