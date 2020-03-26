// RUN: mlir-opt %s -pass-pipeline='func(parallel-loop-collapsing{collapsed-indices-0=0,3 collapsed-indices-1=1,4 collapsed-indices-2=2}, canonicalize)' | FileCheck %s

// CHECK-LABEL:   func @parallel_many_dims() {
func @parallel_many_dims() {
  // CHECK:           [[VAL_0:%.*]] = constant 6 : index
  // CHECK:           [[VAL_1:%.*]] = constant 7 : index
  // CHECK:           [[VAL_2:%.*]] = constant 9 : index
  // CHECK:           [[VAL_3:%.*]] = constant 10 : index
  // CHECK:           [[VAL_4:%.*]] = constant 12 : index
  // CHECK:           [[VAL_5:%.*]] = constant 13 : index
  // CHECK:           [[VAL_6:%.*]] = constant 3 : index
  // CHECK:           [[VAL_7:%.*]] = constant 0 : index
  // CHECK:           [[VAL_8:%.*]] = constant 1 : index
  // CHECK:           [[VAL_9:%.*]] = constant 2 : index
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

  // CHECK:           loop.parallel ([[VAL_10:%.*]], [[VAL_11:%.*]], [[VAL_12:%.*]]) = ([[VAL_7]], [[VAL_7]], [[VAL_7]]) to ([[VAL_9]], [[VAL_8]], [[VAL_8]]) step ([[VAL_8]], [[VAL_8]], [[VAL_8]]) {
  loop.parallel (%i0, %i1, %i2, %i3, %i4) = (%c0, %c3, %c6, %c9, %c12) to (%c2, %c5, %c8, %c11, %c14)
                                          step (%c1, %c4, %c7, %c10, %c13) {
    // CHECK:             [[VAL_13:%.*]] = remi_signed [[VAL_10]], [[VAL_9]] : index
    // CHECK:             [[VAL_14:%.*]] = divi_signed [[VAL_10]], [[VAL_8]] : index
    // CHECK:             [[VAL_15:%.*]] = divi_signed [[VAL_11]], [[VAL_8]] : index
    // CHECK:             [[VAL_16:%.*]] = muli [[VAL_15]], [[VAL_5]] : index
    // CHECK:             [[VAL_17:%.*]] = addi [[VAL_16]], [[VAL_4]] : index
    // CHECK:             [[VAL_18:%.*]] = muli [[VAL_14]], [[VAL_3]] : index
    // CHECK:             [[VAL_19:%.*]] = addi [[VAL_18]], [[VAL_2]] : index
    // CHECK:             [[VAL_20:%.*]] = muli [[VAL_12]], [[VAL_1]] : index
    // CHECK:             [[VAL_21:%.*]] = addi [[VAL_20]], [[VAL_0]] : index
    // CHECK:             [[VAL_22:%.*]] = "magic.op"([[VAL_13]], [[VAL_6]], [[VAL_21]], [[VAL_19]], [[VAL_17]]) : (index, index, index, index, index) -> index
    %result = "magic.op"(%i0, %i1, %i2, %i3, %i4): (index, index, index, index, index) -> index
  }
  return
}
// CHECK:             loop.yield
// CHECK:           }
// CHECK:           return
// CHECK:         }

