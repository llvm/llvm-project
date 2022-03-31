// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='func.func(scf-parallel-loop-collapsing{collapsed-indices-0=0,3 collapsed-indices-1=1,4 collapsed-indices-2=2}, canonicalize)' | FileCheck %s

// CHECK-LABEL: func @parallel_many_dims() {
func @parallel_many_dims() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %c8 = arith.constant 8 : index
  %c9 = arith.constant 9 : index
  %c10 = arith.constant 10 : index
  %c11 = arith.constant 11 : index
  %c12 = arith.constant 12 : index
  %c13 = arith.constant 13 : index
  %c14 = arith.constant 14 : index
  %c15 = arith.constant 15 : index
  %c26 = arith.constant 26 : index

  scf.parallel (%i0, %i1, %i2, %i3, %i4) = (%c0, %c3, %c6, %c9, %c12)
    to (%c2, %c5, %c8, %c26, %c14) step (%c1, %c4, %c7, %c10, %c13) {
    %result = "magic.op"(%i0, %i1, %i2, %i3, %i4)
        : (index, index, index, index, index) -> index
  }
  return
}

// CHECK-DAG: [[C12:%.*]] = arith.constant 12 : index
// CHECK-DAG: [[C10:%.*]] = arith.constant 10 : index
// CHECK-DAG: [[C9:%.*]] = arith.constant 9 : index
// CHECK-DAG: [[C6:%.*]] = arith.constant 6 : index
// CHECK-DAG: [[C4:%.*]] = arith.constant 4 : index
// CHECK-DAG: [[C3:%.*]] = arith.constant 3 : index
// CHECK-DAG: [[C2:%.*]] = arith.constant 2 : index
// CHECK-DAG: [[C1:%.*]] = arith.constant 1 : index
// CHECK-DAG: [[C0:%.*]] = arith.constant 0 : index
// CHECK: scf.parallel ([[NEW_I0:%.*]]) = ([[C0]]) to ([[C4]]) step ([[C1]]) {
// CHECK:   [[V0:%.*]] = arith.remsi [[NEW_I0]], [[C2]] : index
// CHECK:   [[I0:%.*]] = arith.divsi [[NEW_I0]], [[C2]] : index
// CHECK:   [[V2:%.*]] = arith.muli [[V0]], [[C10]] : index
// CHECK:   [[I3:%.*]] = arith.addi [[V2]], [[C9]] : index
// CHECK:   "magic.op"([[I0]], [[C3]], [[C6]], [[I3]], [[C12]]) : (index, index, index, index, index) -> index
// CHECK:   scf.yield
