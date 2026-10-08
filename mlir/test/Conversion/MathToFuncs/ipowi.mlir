// RUN: mlir-opt %s -split-input-file -pass-pipeline="builtin.module(convert-math-to-funcs)" | FileCheck %s
// RUN: mlir-opt %s -split-input-file -pass-pipeline="builtin.module(convert-math-to-funcs)" | FileCheck %s --check-prefix=NODIV

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
// CHECK:           cf.cond_br %[[VAL_6]], ^bb3, ^bb10(%[[VAL_3]], %[[VAL_0]], %[[VAL_1]] : i64, i64, i64)
// CHECK:         ^bb3:
// CHECK:           %[[VAL_7:.*]] = arith.cmpi eq, %[[VAL_0]], %[[VAL_3]] : i64
// CHECK:           cf.cond_br %[[VAL_7]], ^bb4, ^bb5
// CHECK:         ^bb4:
// CHECK:           return %[[VAL_3]] : i64
// CHECK:         ^bb5:
// CHECK:           %[[VAL_8:.*]] = arith.cmpi eq, %[[VAL_0]], %[[VAL_4]] : i64
// CHECK:           cf.cond_br %[[VAL_8]], ^bb6, ^bb9
// CHECK:         ^bb6:
// CHECK:           %[[VAL_9:.*]] = arith.andi %[[VAL_1]], %[[VAL_3]]  : i64
// CHECK:           %[[VAL_10:.*]] = arith.cmpi ne, %[[VAL_9]], %[[VAL_2]] : i64
// CHECK:           cf.cond_br %[[VAL_10]], ^bb7, ^bb8
// CHECK:         ^bb7:
// CHECK:           return %[[VAL_4]] : i64
// CHECK:         ^bb8:
// CHECK:           return %[[VAL_3]] : i64
// CHECK:         ^bb9:
// CHECK:           return %[[VAL_2]] : i64
// CHECK:         ^bb10(%[[VAL_11:.*]]: i64, %[[VAL_12:.*]]: i64, %[[VAL_13:.*]]: i64):
// CHECK:           %[[VAL_14:.*]] = arith.andi %[[VAL_13]], %[[VAL_3]]  : i64
// CHECK:           %[[VAL_15:.*]] = arith.cmpi ne, %[[VAL_14]], %[[VAL_2]] : i64
// CHECK:           cf.cond_br %[[VAL_15]], ^bb11, ^bb12(%[[VAL_11]] : i64)
// CHECK:         ^bb11:
// CHECK:           %[[VAL_16:.*]] = arith.muli %[[VAL_11]], %[[VAL_12]]  : i64
// CHECK:           cf.br ^bb12(%[[VAL_16]] : i64)
// CHECK:         ^bb12(%[[VAL_17:.*]]: i64):
// CHECK:           %[[VAL_18:.*]] = arith.shrui %[[VAL_13]], %[[VAL_3]]  : i64
// CHECK:           %[[VAL_19:.*]] = arith.cmpi eq, %[[VAL_18]], %[[VAL_2]] : i64
// CHECK:           cf.cond_br %[[VAL_19]], ^bb13, ^bb14
// CHECK:         ^bb13:
// CHECK:           return %[[VAL_17]] : i64
// CHECK:         ^bb14:
// CHECK:           %[[VAL_20:.*]] = arith.muli %[[VAL_12]], %[[VAL_12]]  : i64
// CHECK:           cf.br ^bb10(%[[VAL_17]], %[[VAL_20]], %[[VAL_18]] : i64, i64, i64)
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
// CHECK:           cf.cond_br %[[VAL_6]], ^bb3, ^bb10(%[[VAL_3]], %[[VAL_0]], %[[VAL_1]] : i8, i8, i8)
// CHECK:         ^bb3:
// CHECK:           %[[VAL_7:.*]] = arith.cmpi eq, %[[VAL_0]], %[[VAL_3]] : i8
// CHECK:           cf.cond_br %[[VAL_7]], ^bb4, ^bb5
// CHECK:         ^bb4:
// CHECK:           return %[[VAL_3]] : i8
// CHECK:         ^bb5:
// CHECK:           %[[VAL_8:.*]] = arith.cmpi eq, %[[VAL_0]], %[[VAL_4]] : i8
// CHECK:           cf.cond_br %[[VAL_8]], ^bb6, ^bb9
// CHECK:         ^bb6:
// CHECK:           %[[VAL_9:.*]] = arith.andi %[[VAL_1]], %[[VAL_3]]  : i8
// CHECK:           %[[VAL_10:.*]] = arith.cmpi ne, %[[VAL_9]], %[[VAL_2]] : i8
// CHECK:           cf.cond_br %[[VAL_10]], ^bb7, ^bb8
// CHECK:         ^bb7:
// CHECK:           return %[[VAL_4]] : i8
// CHECK:         ^bb8:
// CHECK:           return %[[VAL_3]] : i8
// CHECK:         ^bb9:
// CHECK:           return %[[VAL_2]] : i8
// CHECK:         ^bb10(%[[VAL_11:.*]]: i8, %[[VAL_12:.*]]: i8, %[[VAL_13:.*]]: i8):
// CHECK:           %[[VAL_14:.*]] = arith.andi %[[VAL_13]], %[[VAL_3]]  : i8
// CHECK:           %[[VAL_15:.*]] = arith.cmpi ne, %[[VAL_14]], %[[VAL_2]] : i8
// CHECK:           cf.cond_br %[[VAL_15]], ^bb11, ^bb12(%[[VAL_11]] : i8)
// CHECK:         ^bb11:
// CHECK:           %[[VAL_16:.*]] = arith.muli %[[VAL_11]], %[[VAL_12]]  : i8
// CHECK:           cf.br ^bb12(%[[VAL_16]] : i8)
// CHECK:         ^bb12(%[[VAL_17:.*]]: i8):
// CHECK:           %[[VAL_18:.*]] = arith.shrui %[[VAL_13]], %[[VAL_3]]  : i8
// CHECK:           %[[VAL_19:.*]] = arith.cmpi eq, %[[VAL_18]], %[[VAL_2]] : i8
// CHECK:           cf.cond_br %[[VAL_19]], ^bb13, ^bb14
// CHECK:         ^bb13:
// CHECK:           return %[[VAL_17]] : i8
// CHECK:         ^bb14:
// CHECK:           %[[VAL_20:.*]] = arith.muli %[[VAL_12]], %[[VAL_12]]  : i8
// CHECK:           cf.br ^bb10(%[[VAL_17]], %[[VAL_20]], %[[VAL_18]] : i8, i8, i8)
// CHECK:         }

// -----

// CHECK-LABEL:   func.func @ipowi_vec(
// CHECK-SAME:                          %[[VAL_0:.*]]: vector<2x3xi64>,
// CHECK-SAME:                          %[[VAL_1:.*]]: vector<2x3xi64>) {
func.func @ipowi_vec(%arg0: vector<2x3xi64>, %arg1: vector<2x3xi64>) {
// CHECK:   %[[CST:.*]] = arith.constant dense<0> : vector<2x3xi64>
// CHECK:   %[[B00:.*]] = vector.extract %[[VAL_0]][0, 0] : i64 from vector<2x3xi64>
// CHECK:   %[[E00:.*]] = vector.extract %[[VAL_1]][0, 0] : i64 from vector<2x3xi64>
// CHECK:   %[[R00:.*]] = call @__mlir_math_ipowi_i64(%[[B00]], %[[E00]]) : (i64, i64) -> i64
// CHECK:   %[[TMP00:.*]] = vector.insert %[[R00]], %[[CST]] [0, 0] : i64 into vector<2x3xi64>
// CHECK:   %[[B01:.*]] = vector.extract %[[VAL_0]][0, 1] : i64 from vector<2x3xi64>
// CHECK:   %[[E01:.*]] = vector.extract %[[VAL_1]][0, 1] : i64 from vector<2x3xi64>
// CHECK:   %[[R01:.*]] = call @__mlir_math_ipowi_i64(%[[B01]], %[[E01]]) : (i64, i64) -> i64
// CHECK:   %[[TMP01:.*]] = vector.insert %[[R01]], %[[TMP00]] [0, 1] : i64 into vector<2x3xi64>
// CHECK:   %[[B02:.*]] = vector.extract %[[VAL_0]][0, 2] : i64 from vector<2x3xi64>
// CHECK:   %[[E02:.*]] = vector.extract %[[VAL_1]][0, 2] : i64 from vector<2x3xi64>
// CHECK:   %[[R02:.*]] = call @__mlir_math_ipowi_i64(%[[B02]], %[[E02]]) : (i64, i64) -> i64
// CHECK:   %[[TMP02:.*]] = vector.insert %[[R02]], %[[TMP01]] [0, 2] : i64 into vector<2x3xi64>
// CHECK:   %[[B10:.*]] = vector.extract %[[VAL_0]][1, 0] : i64 from vector<2x3xi64>
// CHECK:   %[[E10:.*]] = vector.extract %[[VAL_1]][1, 0] : i64 from vector<2x3xi64>
// CHECK:   %[[R10:.*]] = call @__mlir_math_ipowi_i64(%[[B10]], %[[E10]]) : (i64, i64) -> i64
// CHECK:   %[[TMP10:.*]] = vector.insert %[[R10]], %[[TMP02]] [1, 0] : i64 into vector<2x3xi64>
// CHECK:   %[[B11:.*]] = vector.extract %[[VAL_0]][1, 1] : i64 from vector<2x3xi64>
// CHECK:   %[[E11:.*]] = vector.extract %[[VAL_1]][1, 1] : i64 from vector<2x3xi64>
// CHECK:   %[[R11:.*]] = call @__mlir_math_ipowi_i64(%[[B11]], %[[E11]]) : (i64, i64) -> i64
// CHECK:   %[[TMP11:.*]] = vector.insert %[[R11]], %[[TMP10]] [1, 1] : i64 into vector<2x3xi64>
// CHECK:   %[[B12:.*]] = vector.extract %[[VAL_0]][1, 2] : i64 from vector<2x3xi64>
// CHECK:   %[[E12:.*]] = vector.extract %[[VAL_1]][1, 2] : i64 from vector<2x3xi64>
// CHECK:   %[[R12:.*]] = call @__mlir_math_ipowi_i64(%[[B12]], %[[E12]]) : (i64, i64) -> i64
// CHECK:   %[[TMP12:.*]] = vector.insert %[[R12]], %[[TMP11]] [1, 2] : i64 into vector<2x3xi64>
// CHECK:   return
// CHECK: }
  %0 = math.ipowi %arg0, %arg1 : vector<2x3xi64>
  func.return
}

// -----

// Check that index is not converted

// CHECK-LABEL: func.func @ipowi_index
// CHECK:         math.ipowi
func.func @ipowi_index(%arg0: index, %arg1: index) {
  %0 = math.ipowi %arg0, %arg1 : index
  func.return
}

// -----

// Zero base raised to a negative exponent is zero. The expansion gets there
// through the same final return of the zero constant as every |b| > 1 base,
// and must not get there by dividing by zero: that is undefined behaviour, a
// trap on targets whose integer division faults and a wrong value on the
// ones where the optimizer exploits it.

// CHECK-LABEL: func @ipowi_zero_base_negative_exponent(
// CHECK-SAME: %[[ARG0:.+]]: i32)
func.func @ipowi_zero_base_negative_exponent(%arg0: i32) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  // CHECK: call @__mlir_math_ipowi_i32(%{{.*}}, %[[ARG0]]) : (i32, i32) -> i32
  %0 = math.ipowi %c0_i32, %arg0 : i32
  func.return %0 : i32
}

// CHECK-LABEL:   func.func private @__mlir_math_ipowi_i32(
// CHECK-SAME:      %[[VAL_0:.*]]: i32,
// CHECK-SAME:      %[[VAL_1:.*]]: i32) -> i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant -1 : i32
// CHECK:           %[[VAL_5:.*]] = arith.cmpi eq, %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:           cf.cond_br %[[VAL_5]], ^bb1, ^bb2
// CHECK:         ^bb2:
// CHECK:           %[[VAL_6:.*]] = arith.cmpi sle, %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:           cf.cond_br %[[VAL_6]], ^bb3, ^bb10

// No block of the helper may divide. The end anchor matches the closing brace
// of the module, which bounds the NODIV-NOT to the whole helper (a bare `}`
// would match the brace of the attribute dictionary on the label line).

// NODIV-LABEL: func.func private @__mlir_math_ipowi_i32
// NODIV-NOT:     arith.divsi
// NODIV:       {{^\}$}}
