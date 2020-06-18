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

  scf.parallel (%i0, %i1, %i2, %i3, %i4) = (%c0, %c3, %c6, %c9, %c12) to (%c2, %c5, %c8, %c11, %c14)
                                          step (%c1, %c4, %c7, %c10, %c13) {
    %result = "magic.op"(%i0, %i1, %i2, %i3, %i4): (index, index, index, index, index) -> index
  }
  return
}

// CHECK-LABEL: func @parallel_many_dims() {
// CHECK:         [[C6:%.*]] = constant 6 : index
// CHECK:         [[C7:%.*]] = constant 7 : index
// CHECK:         [[C9:%.*]] = constant 9 : index
// CHECK:         [[C10:%.*]] = constant 10 : index
// CHECK:         [[C12:%.*]] = constant 12 : index
// CHECK:         [[C13:%.*]] = constant 13 : index
// CHECK:         [[C3:%.*]] = constant 3 : index
// CHECK:         [[C0:%.*]] = constant 0 : index
// CHECK:         [[C1:%.*]] = constant 1 : index
// CHECK:         [[C2:%.*]] = constant 2 : index
// CHECK:         scf.parallel ([[NEW_I0:%.*]], [[NEW_I1:%.*]], [[NEW_I2:%.*]]) = ([[C0]], [[C0]], [[C0]]) to ([[C2]], [[C1]], [[C1]]) step ([[C1]], [[C1]], [[C1]]) {
// CHECK:           [[I0:%.*]] = remi_signed [[NEW_I0]], [[C2]] : index
// CHECK:           [[VAL_16:%.*]] = muli [[NEW_I1]], [[C13]] : index
// CHECK:           [[I4:%.*]] = addi [[VAL_16]], [[C12]] : index
// CHECK:           [[VAL_18:%.*]] = muli [[NEW_I0]], [[C10]] : index
// CHECK:           [[I3:%.*]] = addi [[VAL_18]], [[C9]] : index
// CHECK:           [[VAL_20:%.*]] = muli [[NEW_I2]], [[C7]] : index
// CHECK:           [[I2:%.*]] = addi [[VAL_20]], [[C6]] : index
// CHECK:           "magic.op"([[I0]], [[C3]], [[I2]], [[I3]], [[I4]]) : (index, index, index, index, index) -> index
// CHECK:           scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    return
