// RUN: mlir-opt %s -pass-pipeline='func(parallel-loop-collapsing{collapsed-indices-0=0,1}, canonicalize)' | FileCheck %s

// CHECK-LABEL:   func @collapse_to_single() {
func @collapse_to_single() {
  // CHECK:           [[VAL_0:%.*]] = constant 7 : index
  // CHECK:           [[VAL_1:%.*]] = constant 4 : index
  // CHECK:           [[VAL_2:%.*]] = constant 18 : index
  // CHECK:           [[VAL_3:%.*]] = constant 3 : index
  // CHECK:           [[VAL_4:%.*]] = constant 6 : index
  // CHECK:           [[VAL_5:%.*]] = constant 0 : index
  // CHECK:           [[VAL_6:%.*]] = constant 1 : index
  %c0 = constant 3 : index
  %c1 = constant 7 : index
  %c2 = constant 11 : index
  %c3 = constant 29 : index
  %c4 = constant 3 : index
  %c5 = constant 4 : index
  // CHECK:           loop.parallel ([[VAL_7:%.*]]) = ([[VAL_5]]) to ([[VAL_2]]) step ([[VAL_6]]) {
  loop.parallel (%i0, %i1) = (%c0, %c1) to (%c2, %c3) step (%c4, %c5) {
    // CHECK:             [[VAL_8:%.*]] = remi_signed [[VAL_7]], [[VAL_3]] : index
    // CHECK:             [[VAL_9:%.*]] = divi_signed [[VAL_7]], [[VAL_4]] : index
    // CHECK:             [[VAL_10:%.*]] = muli [[VAL_9]], [[VAL_1]] : index
    // CHECK:             [[VAL_11:%.*]] = addi [[VAL_10]], [[VAL_0]] : index
    // CHECK:             [[VAL_12:%.*]] = muli [[VAL_8]], [[VAL_3]] : index
    // CHECK:             [[VAL_13:%.*]] = addi [[VAL_12]], [[VAL_3]] : index
    // CHECK:             [[VAL_14:%.*]] = "magic.op"([[VAL_13]], [[VAL_11]]) : (index, index) -> index
    %result = "magic.op"(%i0, %i1): (index, index) -> index
  }
  return
}
// CHECK:             loop.yield
// CHECK:           }
// CHECK:           return
// CHECK:         }


