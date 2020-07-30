// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='func(parallel-loop-collapsing{collapsed-indices-0=0,3 collapsed-indices-1=1,4 collapsed-indices-2=2}, canonicalize)' | FileCheck %s

func @parallel_many_dims() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %c5 = constant 5 : index
  %c6 = constant 6 : index
  %c7 = constant 7 : index
  %c8 = constant 8 : index
  %c9 = constant 9 : index
  %c10 = constant 10 : index
  %c11 = constant 11 : index
  %c12 = constant 12 : index
  %c13 = constant 13 : index
  %c14 = constant 14 : index
  %c15 = constant 15 : index
  %c26 = constant 26 : index

  scf.parallel (%i0, %i1, %i2, %i3, %i4) = (%c0, %c3, %c6, %c9, %c12) to (%c2, %c5, %c8, %c11, %c14)
                                          step (%c1, %c4, %c7, %c10, %c13) {
    %result = "magic.op"(%i0, %i1, %i2, %i3, %i4): (index, index, index, index, index) -> index
  }
  return
}

// CHECK-LABEL: func @parallel_many_dims() {
// CHECK:         [[C6:%.*]] = constant 6 : index
// CHECK:         [[C9:%.*]] = constant 9 : index
// CHECK:         [[C10:%.*]] = constant 10 : index
// CHECK:         [[C0:%.*]] = constant 0 : index
// CHECK:         [[C1:%.*]] = constant 1 : index
// CHECK:         [[C2:%.*]] = constant 2 : index
// CHECK:         [[C3:%.*]] = constant 3 : index
// CHECK:         [[C12:%.*]] = constant 12 : index
// CHECK:         scf.parallel ([[NEW_I0:%.*]]) = ([[C0]]) to ([[C2]]) step ([[C1]]) {
// CHECK:           [[I0:%.*]] = remi_signed [[NEW_I0]], [[C2]] : index
// CHECK:           [[V0:%.*]] = divi_signed [[NEW_I0]], [[C2]] : index
// CHECK:           [[V2:%.*]] = muli [[V0]], [[C10]] : index
// CHECK:           [[I3:%.*]] = addi [[V2]], [[C9]] : index
// CHECK:           "magic.op"([[I0]], [[C3]], [[C6]], [[I3]], [[C12]]) : (index, index, index, index, index) -> index
// CHECK:           scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    return
