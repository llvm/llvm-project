// RUN: mlir-opt -allow-unregistered-dialect -pass-pipeline='builtin.module(func.func(test-scf-parallel-loop-collapsing{collapsed-indices-0=0,1}, canonicalize))' --mlir-print-local-scope %s | FileCheck %s

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

// CHECK: func @collapse_to_single() {
// CHECK-DAG:         %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG:         %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:         %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:         %[[C18:.*]] = arith.constant 18 : index
// CHECK:         scf.parallel (%[[NEW_I:.*]]) = (%[[C0]]) to (%[[C18]]) step (%[[C1]]) {
// CHECK:           %[[I0_COUNT:.*]] = arith.remsi %[[NEW_I]], %[[C6]] : index
// CHECK:           %[[I1_COUNT:.*]] = arith.divsi %[[NEW_I]], %[[C6]] : index
// CHECK:           %[[I1:.*]] = affine.apply affine_map<(d0) -> (d0 * 4 + 7)>(%[[I0_COUNT]])
// CHECK:           %[[I0:.*]] = affine.apply affine_map<(d0) -> (d0 * 3 + 3)>(%[[I1_COUNT]])
// CHECK:           "magic.op"(%[[I0]], %[[I1]]) : (index, index) -> index
// CHECK:           scf.reduce
// CHECK-NEXT:    }
// CHECK-NEXT:    return
