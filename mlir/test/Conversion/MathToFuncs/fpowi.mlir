// RUN: mlir-opt %s -split-input-file -pass-pipeline="builtin.module(convert-math-to-funcs{min-width-of-fpowi-exponent=33})" | FileCheck %s

// -----

// Check that i32 exponent case is not converted
// due to {min-width-of-fpowi-exponent=33}:

// CHECK-LABEL:   func.func @fpowi32(
// CHECK-SAME:                       %[[VAL_0:.*]]: f64,
// CHECK-SAME:                       %[[VAL_1:.*]]: i32) {
// CHECK:           %[[VAL_2:.*]] = math.fpowi %[[VAL_0]], %[[VAL_1]] : f64, i32
// CHECK:           return
// CHECK:         }
func.func @fpowi32(%arg0: f64, %arg1: i32) {
  %0 = math.fpowi %arg0, %arg1 : f64, i32
  func.return
}

// -----

// CHECK-LABEL:   func.func @fpowi64(
// CHECK-SAME:                       %[[VAL_0:.*]]: f64,
// CHECK-SAME:                       %[[VAL_1:.*]]: i64) {
// CHECK:           %[[VAL_2:.*]] = call @__mlir_math_fpowi_f64_i64(%[[VAL_0]], %[[VAL_1]]) : (f64, i64) -> f64
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func private @__mlir_math_fpowi_f64_i64(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: f64,
// CHECK-SAME:                                                 %[[VAL_1:.*]]: i64) -> f64 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1.000000e+00 : f64
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_5:.*]] = arith.constant -9223372036854775808 : i64
// CHECK:           %[[VAL_6:.*]] = arith.constant 9223372036854775807 : i64
// CHECK:           %[[VAL_7:.*]] = arith.cmpi eq, %[[VAL_1]], %[[VAL_3]] : i64
// CHECK:           cf.cond_br %[[VAL_7]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           return %[[VAL_2]] : f64
// CHECK:         ^bb2:
// CHECK:           %[[VAL_8:.*]] = arith.cmpi sle, %[[VAL_1]], %[[VAL_3]] : i64
// CHECK:           %[[VAL_9:.*]] = arith.cmpi eq, %[[VAL_1]], %[[VAL_5]] : i64
// CHECK:           %[[VAL_10:.*]] = arith.subi %[[VAL_3]], %[[VAL_1]] : i64
// CHECK:           %[[VAL_11:.*]] = arith.select %[[VAL_8]], %[[VAL_10]], %[[VAL_1]] : i64
// CHECK:           %[[VAL_12:.*]] = arith.select %[[VAL_9]], %[[VAL_6]], %[[VAL_11]] : i64
// CHECK:           cf.br ^bb3(%[[VAL_2]], %[[VAL_0]], %[[VAL_12]] : f64, f64, i64)
// CHECK:         ^bb3(%[[VAL_13:.*]]: f64, %[[VAL_14:.*]]: f64, %[[VAL_15:.*]]: i64):
// CHECK:           %[[VAL_16:.*]] = arith.andi %[[VAL_15]], %[[VAL_4]] : i64
// CHECK:           %[[VAL_17:.*]] = arith.cmpi ne, %[[VAL_16]], %[[VAL_3]] : i64
// CHECK:           cf.cond_br %[[VAL_17]], ^bb4, ^bb5(%[[VAL_13]] : f64)
// CHECK:         ^bb4:
// CHECK:           %[[VAL_18:.*]] = arith.mulf %[[VAL_13]], %[[VAL_14]] : f64
// CHECK:           cf.br ^bb5(%[[VAL_18]] : f64)
// CHECK:         ^bb5(%[[VAL_19:.*]]: f64):
// CHECK:           %[[VAL_20:.*]] = arith.shrui %[[VAL_15]], %[[VAL_4]] : i64
// CHECK:           %[[VAL_21:.*]] = arith.cmpi eq, %[[VAL_20]], %[[VAL_3]] : i64
// CHECK:           cf.cond_br %[[VAL_21]], ^bb7(%[[VAL_19]] : f64), ^bb6
// CHECK:         ^bb6:
// CHECK:           %[[VAL_22:.*]] = arith.mulf %[[VAL_14]], %[[VAL_14]] : f64
// CHECK:           cf.br ^bb3(%[[VAL_19]], %[[VAL_22]], %[[VAL_20]] : f64, f64, i64)
// CHECK:         ^bb7(%[[VAL_23:.*]]: f64):
// CHECK:           cf.cond_br %[[VAL_9]], ^bb8, ^bb9(%[[VAL_23]] : f64)
// CHECK:         ^bb8:
// CHECK:           %[[VAL_24:.*]] = arith.mulf %[[VAL_23]], %[[VAL_0]] : f64
// CHECK:           cf.br ^bb9(%[[VAL_24]] : f64)
// CHECK:         ^bb9(%[[VAL_25:.*]]: f64):
// CHECK:           cf.cond_br %[[VAL_8]], ^bb10, ^bb11(%[[VAL_25]] : f64)
// CHECK:         ^bb10:
// CHECK:           %[[VAL_26:.*]] = arith.divf %[[VAL_2]], %[[VAL_25]] : f64
// CHECK:           cf.br ^bb11(%[[VAL_26]] : f64)
// CHECK:         ^bb11(%[[VAL_27:.*]]: f64):
// CHECK:           return %[[VAL_27]] : f64
// CHECK:         }
func.func @fpowi64(%arg0: f64, %arg1: i64) {
  %0 = math.fpowi %arg0, %arg1 : f64, i64
  func.return
}

// -----

// CHECK-LABEL:   func.func @fpowi64_vec(
// CHECK-SAME:                           %[[VAL_0:.*]]: vector<2x3xf32>,
// CHECK-SAME:                           %[[VAL_1:.*]]: vector<2x3xi64>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant dense<0.000000e+00> : vector<2x3xf32>
// CHECK:           %[[VAL_3:.*]] = vector.extract %[[VAL_0]][0, 0] : f32 from vector<2x3xf32>
// CHECK:           %[[VAL_4:.*]] = vector.extract %[[VAL_1]][0, 0] : i64 from vector<2x3xi64>
// CHECK:           %[[VAL_5:.*]] = call @__mlir_math_fpowi_f32_i64(%[[VAL_3]], %[[VAL_4]]) : (f32, i64) -> f32
// CHECK:           %[[VAL_6:.*]] = vector.insert %[[VAL_5]], %[[VAL_2]] [0, 0] : f32 into vector<2x3xf32>
// CHECK:           %[[VAL_7:.*]] = vector.extract %[[VAL_0]][0, 1] : f32 from vector<2x3xf32>
// CHECK:           %[[VAL_8:.*]] = vector.extract %[[VAL_1]][0, 1] : i64 from vector<2x3xi64>
// CHECK:           %[[VAL_9:.*]] = call @__mlir_math_fpowi_f32_i64(%[[VAL_7]], %[[VAL_8]]) : (f32, i64) -> f32
// CHECK:           %[[VAL_10:.*]] = vector.insert %[[VAL_9]], %[[VAL_6]] [0, 1] : f32 into vector<2x3xf32>
// CHECK:           %[[VAL_11:.*]] = vector.extract %[[VAL_0]][0, 2] : f32 from vector<2x3xf32>
// CHECK:           %[[VAL_12:.*]] = vector.extract %[[VAL_1]][0, 2] : i64 from vector<2x3xi64>
// CHECK:           %[[VAL_13:.*]] = call @__mlir_math_fpowi_f32_i64(%[[VAL_11]], %[[VAL_12]]) : (f32, i64) -> f32
// CHECK:           %[[VAL_14:.*]] = vector.insert %[[VAL_13]], %[[VAL_10]] [0, 2] : f32 into vector<2x3xf32>
// CHECK:           %[[VAL_15:.*]] = vector.extract %[[VAL_0]][1, 0] : f32 from vector<2x3xf32>
// CHECK:           %[[VAL_16:.*]] = vector.extract %[[VAL_1]][1, 0] : i64 from vector<2x3xi64>
// CHECK:           %[[VAL_17:.*]] = call @__mlir_math_fpowi_f32_i64(%[[VAL_15]], %[[VAL_16]]) : (f32, i64) -> f32
// CHECK:           %[[VAL_18:.*]] = vector.insert %[[VAL_17]], %[[VAL_14]] [1, 0] : f32 into vector<2x3xf32>
// CHECK:           %[[VAL_19:.*]] = vector.extract %[[VAL_0]][1, 1] : f32 from vector<2x3xf32>
// CHECK:           %[[VAL_20:.*]] = vector.extract %[[VAL_1]][1, 1] : i64 from vector<2x3xi64>
// CHECK:           %[[VAL_21:.*]] = call @__mlir_math_fpowi_f32_i64(%[[VAL_19]], %[[VAL_20]]) : (f32, i64) -> f32
// CHECK:           %[[VAL_22:.*]] = vector.insert %[[VAL_21]], %[[VAL_18]] [1, 1] : f32 into vector<2x3xf32>
// CHECK:           %[[VAL_23:.*]] = vector.extract %[[VAL_0]][1, 2] : f32 from vector<2x3xf32>
// CHECK:           %[[VAL_24:.*]] = vector.extract %[[VAL_1]][1, 2] : i64 from vector<2x3xi64>
// CHECK:           %[[VAL_25:.*]] = call @__mlir_math_fpowi_f32_i64(%[[VAL_23]], %[[VAL_24]]) : (f32, i64) -> f32
// CHECK:           %[[VAL_26:.*]] = vector.insert %[[VAL_25]], %[[VAL_22]] [1, 2] : f32 into vector<2x3xf32>
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func private @__mlir_math_fpowi_f32_i64(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: f32,
// CHECK-SAME:                                                 %[[VAL_1:.*]]: i64) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_5:.*]] = arith.constant -9223372036854775808 : i64
// CHECK:           %[[VAL_6:.*]] = arith.constant 9223372036854775807 : i64
// CHECK:           %[[VAL_7:.*]] = arith.cmpi eq, %[[VAL_1]], %[[VAL_3]] : i64
// CHECK:           cf.cond_br %[[VAL_7]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           return %[[VAL_2]] : f32
// CHECK:         ^bb2:
// CHECK:           %[[VAL_8:.*]] = arith.cmpi sle, %[[VAL_1]], %[[VAL_3]] : i64
// CHECK:           %[[VAL_9:.*]] = arith.cmpi eq, %[[VAL_1]], %[[VAL_5]] : i64
// CHECK:           %[[VAL_10:.*]] = arith.subi %[[VAL_3]], %[[VAL_1]] : i64
// CHECK:           %[[VAL_11:.*]] = arith.select %[[VAL_8]], %[[VAL_10]], %[[VAL_1]] : i64
// CHECK:           %[[VAL_12:.*]] = arith.select %[[VAL_9]], %[[VAL_6]], %[[VAL_11]] : i64
// CHECK:           cf.br ^bb3(%[[VAL_2]], %[[VAL_0]], %[[VAL_12]] : f32, f32, i64)
// CHECK:         ^bb3(%[[VAL_13:.*]]: f32, %[[VAL_14:.*]]: f32, %[[VAL_15:.*]]: i64):
// CHECK:           %[[VAL_16:.*]] = arith.andi %[[VAL_15]], %[[VAL_4]] : i64
// CHECK:           %[[VAL_17:.*]] = arith.cmpi ne, %[[VAL_16]], %[[VAL_3]] : i64
// CHECK:           cf.cond_br %[[VAL_17]], ^bb4, ^bb5(%[[VAL_13]] : f32)
// CHECK:         ^bb4:
// CHECK:           %[[VAL_18:.*]] = arith.mulf %[[VAL_13]], %[[VAL_14]] : f32
// CHECK:           cf.br ^bb5(%[[VAL_18]] : f32)
// CHECK:         ^bb5(%[[VAL_19:.*]]: f32):
// CHECK:           %[[VAL_20:.*]] = arith.shrui %[[VAL_15]], %[[VAL_4]] : i64
// CHECK:           %[[VAL_21:.*]] = arith.cmpi eq, %[[VAL_20]], %[[VAL_3]] : i64
// CHECK:           cf.cond_br %[[VAL_21]], ^bb7(%[[VAL_19]] : f32), ^bb6
// CHECK:         ^bb6:
// CHECK:           %[[VAL_22:.*]] = arith.mulf %[[VAL_14]], %[[VAL_14]] : f32
// CHECK:           cf.br ^bb3(%[[VAL_19]], %[[VAL_22]], %[[VAL_20]] : f32, f32, i64)
// CHECK:         ^bb7(%[[VAL_23:.*]]: f32):
// CHECK:           cf.cond_br %[[VAL_9]], ^bb8, ^bb9(%[[VAL_23]] : f32)
// CHECK:         ^bb8:
// CHECK:           %[[VAL_24:.*]] = arith.mulf %[[VAL_23]], %[[VAL_0]] : f32
// CHECK:           cf.br ^bb9(%[[VAL_24]] : f32)
// CHECK:         ^bb9(%[[VAL_25:.*]]: f32):
// CHECK:           cf.cond_br %[[VAL_8]], ^bb10, ^bb11(%[[VAL_25]] : f32)
// CHECK:         ^bb10:
// CHECK:           %[[VAL_26:.*]] = arith.divf %[[VAL_2]], %[[VAL_25]] : f32
// CHECK:           cf.br ^bb11(%[[VAL_26]] : f32)
// CHECK:         ^bb11(%[[VAL_27:.*]]: f32):
// CHECK:           return %[[VAL_27]] : f32
// CHECK:         }
func.func @fpowi64_vec(%arg0: vector<2x3xf32>, %arg1: vector<2x3xi64>) {
  %0 = math.fpowi %arg0, %arg1 : vector<2x3xf32>, vector<2x3xi64>
  func.return
}
