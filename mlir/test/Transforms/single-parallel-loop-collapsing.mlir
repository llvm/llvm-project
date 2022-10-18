// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(scf-parallel-loop-collapsing{collapsed-indices-0=0,1}, canonicalize))' | FileCheck %s

func.func @collapse_to_single() {
  %c0 = arith.constant 3 : index
  %c1 = arith.constant 7 : index
  %c2 = arith.constant 11 : index
  %c3 = arith.constant 29 : index
  %c4 = arith.constant 3 : index
  %c5 = arith.constant 4 : index
  scf.parallel (%i0, %i1) = (%c0, %c1) to (%c2, %c3) step (%c4, %c5) {
    %result = "magic.op"(%i0, %i1): (index, index) -> index
  }
  return
}

// CHECK-LABEL: func @collapse_to_single() {
// CHECK-DAG:         [[C18:%.*]] = arith.constant 18 : index
// CHECK-DAG:         [[C6:%.*]] = arith.constant 6 : index
// CHECK-DAG:         [[C3:%.*]] = arith.constant 3 : index
// CHECK-DAG:         [[C7:%.*]] = arith.constant 7 : index
// CHECK-DAG:         [[C4:%.*]] = arith.constant 4 : index
// CHECK-DAG:         [[C1:%.*]] = arith.constant 1 : index
// CHECK-DAG:         [[C0:%.*]] = arith.constant 0 : index
// CHECK:         scf.parallel ([[NEW_I:%.*]]) = ([[C0]]) to ([[C18]]) step ([[C1]]) {
// CHECK:           [[I0_COUNT:%.*]] = arith.remsi [[NEW_I]], [[C6]] : index
// CHECK:           [[I1_COUNT:%.*]] = arith.divsi [[NEW_I]], [[C6]] : index
// CHECK:           [[V0:%.*]] = arith.muli [[I0_COUNT]], [[C4]] : index
// CHECK:           [[I1:%.*]] = arith.addi [[V0]], [[C7]] : index
// CHECK:           [[V1:%.*]] = arith.muli [[I1_COUNT]], [[C3]] : index
// CHECK:           [[I0:%.*]] = arith.addi [[V1]], [[C3]] : index
// CHECK:           "magic.op"([[I0]], [[I1]]) : (index, index) -> index
// CHECK:           scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    return
