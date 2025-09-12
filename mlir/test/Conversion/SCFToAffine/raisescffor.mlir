// RUN: mlir-opt --raise-scf-to-affine %s | FileCheck %s

module {
  func.func private @_Z12kernel5_initPc(%0: index, %arg0: memref<index>) {
    %c10 = arith.constant 10 : index
    %c0 = arith.constant 0 : index
    scf.for %arg1 = %c0 to %c10 step %0 {
      memref.store %c10, %arg0[] : memref<index>
    }
    return
  }
}

// CHECK-LABEL:   func.func private @_Z12kernel5_initPc(
// CHECK-SAME:                                          %[[VAL_0:.*]]: index,
// CHECK-SAME:                                          %[[VAL_1:.*]]: memref<index>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.subi %[[VAL_0]], %[[VAL_2]] : index
// CHECK:           %[[VAL_5:.*]] = arith.addi %[[VAL_4]], %[[VAL_3]] : index
// CHECK:           %[[VAL_6:.*]] = arith.divui %[[VAL_5]], %[[VAL_0]] : index
// CHECK:           affine.for %[[VAL_7:.*]] = 0 to %[[VAL_6]] {
// CHECK:             memref.store %[[VAL_3]], %[[VAL_1]][] : memref<index>
// CHECK:           }

