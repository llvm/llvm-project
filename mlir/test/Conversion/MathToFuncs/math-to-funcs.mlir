// RUN: mlir-opt %s -split-input-file -pass-pipeline="convert-math-to-funcs" | FileCheck %s

// -----

// CHECK-LABEL: func @ipowi(
// CHECK-SAME: %[[ARG0:.+]]: i64,
// CHECK-SAME: %[[ARG1:.+]]: i64)
func.func @ipowi(%arg0: i64, %arg1: i64) {
  // CHECK: call @__mlir_math_ipowi_i64(%[[ARG0]], %[[ARG1]]) : (i64, i64) -> i64
  %0 = math.ipowi %arg0, %arg1 : i64
  func.return
}

// CHECK-LABEL:   func.func private @__mlir_math_ipowi_i64(
// CHECK-SAME:      %[[VAL_0:.*]]: i64,
// CHECK-SAME:      %[[VAL_1:.*]]: i64) -> i64
// CHECK-SAME:        attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_4:.*]] = arith.constant -1 : i64
// CHECK:           %[[VAL_5:.*]] = arith.cmpi eq, %[[VAL_1]], %[[VAL_2]] : i64
// CHECK:           cf.cond_br %[[VAL_5]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           return %[[VAL_3]] : i64
// CHECK:         ^bb2:
// CHECK:           %[[VAL_6:.*]] = arith.cmpi sle, %[[VAL_1]], %[[VAL_2]] : i64
// CHECK:           cf.cond_br %[[VAL_6]], ^bb3, ^bb12(%[[VAL_3]], %[[VAL_0]], %[[VAL_1]] : i64, i64, i64)
// CHECK:         ^bb3:
// CHECK:           %[[VAL_7:.*]] = arith.cmpi eq, %[[VAL_0]], %[[VAL_2]] : i64
// CHECK:           cf.cond_br %[[VAL_7]], ^bb4, ^bb5
// CHECK:         ^bb4:
// CHECK:           %[[VAL_8:.*]] = arith.divsi %[[VAL_3]], %[[VAL_2]]  : i64
// CHECK:           return %[[VAL_8]] : i64
// CHECK:         ^bb5:
// CHECK:           %[[VAL_9:.*]] = arith.cmpi eq, %[[VAL_0]], %[[VAL_3]] : i64
// CHECK:           cf.cond_br %[[VAL_9]], ^bb6, ^bb7
// CHECK:         ^bb6:
// CHECK:           return %[[VAL_3]] : i64
// CHECK:         ^bb7:
// CHECK:           %[[VAL_10:.*]] = arith.cmpi eq, %[[VAL_0]], %[[VAL_4]] : i64
// CHECK:           cf.cond_br %[[VAL_10]], ^bb8, ^bb11
// CHECK:         ^bb8:
// CHECK:           %[[VAL_11:.*]] = arith.andi %[[VAL_1]], %[[VAL_3]]  : i64
// CHECK:           %[[VAL_12:.*]] = arith.cmpi ne, %[[VAL_11]], %[[VAL_2]] : i64
// CHECK:           cf.cond_br %[[VAL_12]], ^bb9, ^bb10
// CHECK:         ^bb9:
// CHECK:           return %[[VAL_4]] : i64
// CHECK:         ^bb10:
// CHECK:           return %[[VAL_3]] : i64
// CHECK:         ^bb11:
// CHECK:           return %[[VAL_2]] : i64
// CHECK:         ^bb12(%[[VAL_13:.*]]: i64, %[[VAL_14:.*]]: i64, %[[VAL_15:.*]]: i64):
// CHECK:           %[[VAL_16:.*]] = arith.andi %[[VAL_15]], %[[VAL_3]]  : i64
// CHECK:           %[[VAL_17:.*]] = arith.cmpi ne, %[[VAL_16]], %[[VAL_2]] : i64
// CHECK:           cf.cond_br %[[VAL_17]], ^bb13, ^bb14(%[[VAL_13]] : i64)
// CHECK:         ^bb13:
// CHECK:           %[[VAL_18:.*]] = arith.muli %[[VAL_13]], %[[VAL_14]]  : i64
// CHECK:           cf.br ^bb14(%[[VAL_18]] : i64)
// CHECK:         ^bb14(%[[VAL_19:.*]]: i64):
// CHECK:           %[[VAL_20:.*]] = arith.shrui %[[VAL_15]], %[[VAL_3]]  : i64
// CHECK:           %[[VAL_21:.*]] = arith.cmpi eq, %[[VAL_20]], %[[VAL_2]] : i64
// CHECK:           cf.cond_br %[[VAL_21]], ^bb15, ^bb16
// CHECK:         ^bb15:
// CHECK:           return %[[VAL_19]] : i64
// CHECK:         ^bb16:
// CHECK:           %[[VAL_22:.*]] = arith.muli %[[VAL_14]], %[[VAL_14]]  : i64
// CHECK:           cf.br ^bb12(%[[VAL_19]], %[[VAL_22]], %[[VAL_20]] : i64, i64, i64)
// CHECK:         }

// -----

// CHECK-LABEL: func @ipowi(
// CHECK-SAME: %[[ARG0:.+]]: i8,
// CHECK-SAME: %[[ARG1:.+]]: i8)
  // CHECK: call @__mlir_math_ipowi_i8(%[[ARG0]], %[[ARG1]]) : (i8, i8) -> i8
func.func @ipowi(%arg0: i8, %arg1: i8) {
  %0 = math.ipowi %arg0, %arg1 : i8
  func.return
}

// CHECK-LABEL:   func.func private @__mlir_math_ipowi_i8(
// CHECK-SAME:      %[[VAL_0:.*]]: i8,
// CHECK-SAME:      %[[VAL_1:.*]]: i8) -> i8
// CHECK-SAME:        attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i8
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i8
// CHECK:           %[[VAL_4:.*]] = arith.constant -1 : i8
// CHECK:           %[[VAL_5:.*]] = arith.cmpi eq, %[[VAL_1]], %[[VAL_2]] : i8
// CHECK:           cf.cond_br %[[VAL_5]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           return %[[VAL_3]] : i8
// CHECK:         ^bb2:
// CHECK:           %[[VAL_6:.*]] = arith.cmpi sle, %[[VAL_1]], %[[VAL_2]] : i8
// CHECK:           cf.cond_br %[[VAL_6]], ^bb3, ^bb12(%[[VAL_3]], %[[VAL_0]], %[[VAL_1]] : i8, i8, i8)
// CHECK:         ^bb3:
// CHECK:           %[[VAL_7:.*]] = arith.cmpi eq, %[[VAL_0]], %[[VAL_2]] : i8
// CHECK:           cf.cond_br %[[VAL_7]], ^bb4, ^bb5
// CHECK:         ^bb4:
// CHECK:           %[[VAL_8:.*]] = arith.divsi %[[VAL_3]], %[[VAL_2]]  : i8
// CHECK:           return %[[VAL_8]] : i8
// CHECK:         ^bb5:
// CHECK:           %[[VAL_9:.*]] = arith.cmpi eq, %[[VAL_0]], %[[VAL_3]] : i8
// CHECK:           cf.cond_br %[[VAL_9]], ^bb6, ^bb7
// CHECK:         ^bb6:
// CHECK:           return %[[VAL_3]] : i8
// CHECK:         ^bb7:
// CHECK:           %[[VAL_10:.*]] = arith.cmpi eq, %[[VAL_0]], %[[VAL_4]] : i8
// CHECK:           cf.cond_br %[[VAL_10]], ^bb8, ^bb11
// CHECK:         ^bb8:
// CHECK:           %[[VAL_11:.*]] = arith.andi %[[VAL_1]], %[[VAL_3]]  : i8
// CHECK:           %[[VAL_12:.*]] = arith.cmpi ne, %[[VAL_11]], %[[VAL_2]] : i8
// CHECK:           cf.cond_br %[[VAL_12]], ^bb9, ^bb10
// CHECK:         ^bb9:
// CHECK:           return %[[VAL_4]] : i8
// CHECK:         ^bb10:
// CHECK:           return %[[VAL_3]] : i8
// CHECK:         ^bb11:
// CHECK:           return %[[VAL_2]] : i8
// CHECK:         ^bb12(%[[VAL_13:.*]]: i8, %[[VAL_14:.*]]: i8, %[[VAL_15:.*]]: i8):
// CHECK:           %[[VAL_16:.*]] = arith.andi %[[VAL_15]], %[[VAL_3]]  : i8
// CHECK:           %[[VAL_17:.*]] = arith.cmpi ne, %[[VAL_16]], %[[VAL_2]] : i8
// CHECK:           cf.cond_br %[[VAL_17]], ^bb13, ^bb14(%[[VAL_13]] : i8)
// CHECK:         ^bb13:
// CHECK:           %[[VAL_18:.*]] = arith.muli %[[VAL_13]], %[[VAL_14]]  : i8
// CHECK:           cf.br ^bb14(%[[VAL_18]] : i8)
// CHECK:         ^bb14(%[[VAL_19:.*]]: i8):
// CHECK:           %[[VAL_20:.*]] = arith.shrui %[[VAL_15]], %[[VAL_3]]  : i8
// CHECK:           %[[VAL_21:.*]] = arith.cmpi eq, %[[VAL_20]], %[[VAL_2]] : i8
// CHECK:           cf.cond_br %[[VAL_21]], ^bb15, ^bb16
// CHECK:         ^bb15:
// CHECK:           return %[[VAL_19]] : i8
// CHECK:         ^bb16:
// CHECK:           %[[VAL_22:.*]] = arith.muli %[[VAL_14]], %[[VAL_14]]  : i8
// CHECK:           cf.br ^bb12(%[[VAL_19]], %[[VAL_22]], %[[VAL_20]] : i8, i8, i8)
// CHECK:         }

// -----

// CHECK-LABEL:   func.func @ipowi_vec(
// CHECK-SAME:                          %[[VAL_0:.*]]: vector<2x3xi64>,
// CHECK-SAME:                          %[[VAL_1:.*]]: vector<2x3xi64>) {
func.func @ipowi_vec(%arg0: vector<2x3xi64>, %arg1: vector<2x3xi64>) {
// CHECK:   %[[CST:.*]] = arith.constant dense<0> : vector<2x3xi64>
// CHECK:   %[[B00:.*]] = vector.extract %[[VAL_0]][0, 0] : vector<2x3xi64>
// CHECK:   %[[E00:.*]] = vector.extract %[[VAL_1]][0, 0] : vector<2x3xi64>
// CHECK:   %[[R00:.*]] = call @__mlir_math_ipowi_i64(%[[B00]], %[[E00]]) : (i64, i64) -> i64
// CHECK:   %[[TMP00:.*]] = vector.insert %[[R00]], %[[CST]] [0, 0] : i64 into vector<2x3xi64>
// CHECK:   %[[B01:.*]] = vector.extract %[[VAL_0]][0, 1] : vector<2x3xi64>
// CHECK:   %[[E01:.*]] = vector.extract %[[VAL_1]][0, 1] : vector<2x3xi64>
// CHECK:   %[[R01:.*]] = call @__mlir_math_ipowi_i64(%[[B01]], %[[E01]]) : (i64, i64) -> i64
// CHECK:   %[[TMP01:.*]] = vector.insert %[[R01]], %[[TMP00]] [0, 1] : i64 into vector<2x3xi64>
// CHECK:   %[[B02:.*]] = vector.extract %[[VAL_0]][0, 2] : vector<2x3xi64>
// CHECK:   %[[E02:.*]] = vector.extract %[[VAL_1]][0, 2] : vector<2x3xi64>
// CHECK:   %[[R02:.*]] = call @__mlir_math_ipowi_i64(%[[B02]], %[[E02]]) : (i64, i64) -> i64
// CHECK:   %[[TMP02:.*]] = vector.insert %[[R02]], %[[TMP01]] [0, 2] : i64 into vector<2x3xi64>
// CHECK:   %[[B10:.*]] = vector.extract %[[VAL_0]][1, 0] : vector<2x3xi64>
// CHECK:   %[[E10:.*]] = vector.extract %[[VAL_1]][1, 0] : vector<2x3xi64>
// CHECK:   %[[R10:.*]] = call @__mlir_math_ipowi_i64(%[[B10]], %[[E10]]) : (i64, i64) -> i64
// CHECK:   %[[TMP10:.*]] = vector.insert %[[R10]], %[[TMP02]] [1, 0] : i64 into vector<2x3xi64>
// CHECK:   %[[B11:.*]] = vector.extract %[[VAL_0]][1, 1] : vector<2x3xi64>
// CHECK:   %[[E11:.*]] = vector.extract %[[VAL_1]][1, 1] : vector<2x3xi64>
// CHECK:   %[[R11:.*]] = call @__mlir_math_ipowi_i64(%[[B11]], %[[E11]]) : (i64, i64) -> i64
// CHECK:   %[[TMP11:.*]] = vector.insert %[[R11]], %[[TMP10]] [1, 1] : i64 into vector<2x3xi64>
// CHECK:   %[[B12:.*]] = vector.extract %[[VAL_0]][1, 2] : vector<2x3xi64>
// CHECK:   %[[E12:.*]] = vector.extract %[[VAL_1]][1, 2] : vector<2x3xi64>
// CHECK:   %[[R12:.*]] = call @__mlir_math_ipowi_i64(%[[B12]], %[[E12]]) : (i64, i64) -> i64
// CHECK:   %[[TMP12:.*]] = vector.insert %[[R12]], %[[TMP11]] [1, 2] : i64 into vector<2x3xi64>
// CHECK:   return
// CHECK: }
  %0 = math.ipowi %arg0, %arg1 : vector<2x3xi64>
  func.return
}
