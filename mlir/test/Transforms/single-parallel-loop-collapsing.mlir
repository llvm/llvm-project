// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='func(parallel-loop-collapsing{collapsed-indices-0=0,1}, canonicalize)' | FileCheck %s

func @collapse_to_single() {
  %c0 = constant 3 : index
  %c1 = constant 7 : index
  %c2 = constant 11 : index
  %c3 = constant 29 : index
  %c4 = constant 3 : index
  %c5 = constant 4 : index
  scf.parallel (%i0, %i1) = (%c0, %c1) to (%c2, %c3) step (%c4, %c5) {
    %result = "magic.op"(%i0, %i1): (index, index) -> index
  }
  return
}

// CHECK-LABEL: func @collapse_to_single() {
// CHECK:         [[C7:%.*]] = constant 7 : index
// CHECK:         [[C4:%.*]] = constant 4 : index
// CHECK:         [[C18:%.*]] = constant 18 : index
// CHECK:         [[C3:%.*]] = constant 3 : index
// CHECK:         [[C0:%.*]] = constant 0 : index
// CHECK:         [[C1:%.*]] = constant 1 : index
// CHECK:         scf.parallel ([[NEW_I:%.*]]) = ([[C0]]) to ([[C18]]) step ([[C1]]) {
// CHECK:           [[I0_COUNT:%.*]] = remi_signed [[NEW_I]], [[C3]] : index
// CHECK:           [[I1_COUNT:%.*]] = divi_signed [[NEW_I]], [[C3]] : index
// CHECK:           [[V0:%.*]] = muli [[I1_COUNT]], [[C4]] : index
// CHECK:           [[I1:%.*]] = addi [[V0]], [[C7]] : index
// CHECK:           [[V1:%.*]] = muli [[I0_COUNT]], [[C3]] : index
// CHECK:           [[I0:%.*]] = addi [[V1]], [[C3]] : index
// CHECK:           "magic.op"([[I0]], [[I1]]) : (index, index) -> index
// CHECK:           scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    return
