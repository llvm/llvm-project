// RUN: mlir-opt %s -canonicalize="test-convergence" -split-input-file | FileCheck %s

///===----------------------------------------------===//
///  Tests of `StepCompareFolder`
///===----------------------------------------------===//


///===------------------------------------===//
///  Tests of `ugt` (unsigned greater than)
///===------------------------------------===//

// CHECK-LABEL: @ugt_constant_3_lhs
//       CHECK: %[[CST:.*]] = arith.constant dense<true> : vector<3xi1>
//       CHECK: return %[[CST]] : vector<3xi1>
func.func @ugt_constant_3_lhs() -> vector<3xi1> {
  %cst = arith.constant dense<3> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  // 3 > [0, 1, 2] => [true, true, true] => true for all indices => fold
  %1 = arith.cmpi ugt, %cst, %0 : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @negative_ugt_constant_2_lhs
//       CHECK: %[[CMP:.*]] = arith.cmpi
//       CHECK: return %[[CMP]]
func.func @negative_ugt_constant_2_lhs() -> vector<3xi1> {
  %cst = arith.constant dense<2> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  // 2 > [0, 1, 2] => [true, true, false] => not same for all indices => don't fold
  %1 = arith.cmpi ugt, %cst, %0 : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @ugt_constant_3_rhs
//       CHECK: %[[CST:.*]] = arith.constant dense<false> : vector<3xi1>
//       CHECK: return %[[CST]] : vector<3xi1>
func.func @ugt_constant_3_rhs() -> vector<3xi1> {
  %cst = arith.constant dense<3> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  // [0, 1, 2] > 3 => [false, false, false] => false for all indices => fold
  %1 = arith.cmpi ugt, %0, %cst : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @ugt_constant_max_rhs
//       CHECK: %[[CST:.*]] = arith.constant dense<false> : vector<3xi1>
//       CHECK: return %[[CST]] : vector<3xi1>
func.func @ugt_constant_max_rhs() -> vector<3xi1> {
  // The largest i64 possible:
  %cst = arith.constant dense<0x7fffffffffffffff> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  %1 = arith.cmpi ugt, %0, %cst: vector<3xindex>
  return %1 : vector<3xi1>
}


// -----

// CHECK-LABEL: @ugt_constant_2_rhs
//       CHECK: %[[CST:.*]] = arith.constant dense<false> : vector<3xi1>
//       CHECK: return %[[CST]] : vector<3xi1>
func.func @ugt_constant_2_rhs() -> vector<3xi1> {
  %cst = arith.constant dense<2> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  // [0, 1, 2] > 2 => [false, false, false] => false for all indices => fold
  %1 = arith.cmpi ugt, %0, %cst : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @negative_ugt_constant_1_rhs
//       CHECK: %[[CMP:.*]] = arith.cmpi
//       CHECK: return %[[CMP]]
func.func @negative_ugt_constant_1_rhs() -> vector<3xi1> {
  %cst = arith.constant dense<1> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  // [0, 1, 2] > 1 => [false, false, true] => not same for all indices => don't fold
  %1 = arith.cmpi ugt, %0, %cst: vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

///===------------------------------------===//
///  Tests of `uge` (unsigned greater than or equal)
///===------------------------------------===//


// CHECK-LABEL: @uge_constant_2_lhs
//       CHECK: %[[CST:.*]] = arith.constant dense<true> : vector<3xi1>
//       CHECK: return %[[CST]] : vector<3xi1>
func.func @uge_constant_2_lhs() -> vector<3xi1> {
  %cst = arith.constant dense<2> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  // 2 >= [0, 1, 2] => [true, true, true] => true for all indices => fold
  %1 = arith.cmpi uge, %cst, %0 : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @negative_uge_constant_1_lhs
//       CHECK: %[[CMP:.*]] = arith.cmpi
//       CHECK: return %[[CMP]]
func.func @negative_uge_constant_1_lhs() -> vector<3xi1> {
  %cst = arith.constant dense<1> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  // 1 >= [0, 1, 2] => [true, false, false] => not same for all indices => don't fold
  %1 = arith.cmpi uge, %cst, %0 : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @uge_constant_3_rhs
//       CHECK: %[[CST:.*]] = arith.constant dense<false> : vector<3xi1>
//       CHECK: return %[[CST]] : vector<3xi1>
func.func @uge_constant_3_rhs() -> vector<3xi1> {
  %cst = arith.constant dense<3> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  // [0, 1, 2] >= 3 => [false, false, false] => false for all indices => fold
  %1 = arith.cmpi uge, %0, %cst : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @negative_uge_constant_2_rhs
//       CHECK: %[[CMP:.*]] = arith.cmpi
//       CHECK: return %[[CMP]]
func.func @negative_uge_constant_2_rhs() -> vector<3xi1> {
  %cst = arith.constant dense<2> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  // [0, 1, 2] >= 2 => [false, false, true] => not same for all indices => don't fold
  %1 = arith.cmpi uge, %0, %cst : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----


///===------------------------------------===//
///  Tests of `ult` (unsigned less than)
///===------------------------------------===//


// CHECK-LABEL: @ult_constant_2_lhs
//       CHECK: %[[CST:.*]] = arith.constant dense<false> : vector<3xi1>
//       CHECK: return %[[CST]] : vector<3xi1>
func.func @ult_constant_2_lhs() -> vector<3xi1> {
  %cst = arith.constant dense<2> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  // 2 < [0, 1, 2] => [false, false, false] => false for all indices => fold
  %1 = arith.cmpi ult, %cst, %0 : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @negative_ult_constant_1_lhs
//       CHECK: %[[CMP:.*]] = arith.cmpi
//       CHECK: return %[[CMP]]
func.func @negative_ult_constant_1_lhs() -> vector<3xi1> {
  %cst = arith.constant dense<1> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  // 1 < [0, 1, 2] => [false, false, true] => not same for all indices => don't fold
  %1 = arith.cmpi ult, %cst, %0 : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @ult_constant_3_rhs
//       CHECK: %[[CST:.*]] = arith.constant dense<true> : vector<3xi1>
//       CHECK: return %[[CST]] : vector<3xi1>
func.func @ult_constant_3_rhs() -> vector<3xi1> {
  %cst = arith.constant dense<3> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  // [0, 1, 2] < 3 => [true, true, true] => true for all indices => fold
  %1 = arith.cmpi ult, %0, %cst : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @negative_ult_constant_2_rhs
//       CHECK: %[[CMP:.*]] = arith.cmpi
//       CHECK: return %[[CMP]]
func.func @negative_ult_constant_2_rhs() -> vector<3xi1> {
  %cst = arith.constant dense<2> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  // [0, 1, 2] < 2 => [true, true, false] => not same for all indices => don't fold
  %1 = arith.cmpi ult, %0, %cst : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

///===------------------------------------===//
///  Tests of `ule` (unsigned less than or equal)
///===------------------------------------===//

// CHECK-LABEL: @ule_constant_3_lhs
//       CHECK: %[[CST:.*]] = arith.constant dense<false> : vector<3xi1>
//       CHECK: return %[[CST]] : vector<3xi1>
func.func @ule_constant_3_lhs() -> vector<3xi1> {
  %cst = arith.constant dense<3> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  %1 = arith.cmpi ule, %cst, %0 : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @negative_ule_constant_2_lhs
//       CHECK: %[[CMP:.*]] = arith.cmpi
//       CHECK: return %[[CMP]]
func.func @negative_ule_constant_2_lhs() -> vector<3xi1> {
  %cst = arith.constant dense<2> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  %1 = arith.cmpi ule, %cst, %0 : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @ule_constant_2_rhs
//       CHECK: %[[CST:.*]] = arith.constant dense<true> : vector<3xi1>
//       CHECK: return %[[CST]] : vector<3xi1>
func.func @ule_constant_2_rhs() -> vector<3xi1> {
  %cst = arith.constant dense<2> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  %1 = arith.cmpi ule, %0, %cst : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @negative_ule_constant_1_rhs
//       CHECK: %[[CMP:.*]] = arith.cmpi
//       CHECK: return %[[CMP]]
func.func @negative_ule_constant_1_rhs() -> vector<3xi1> {
  %cst = arith.constant dense<1> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  %1 = arith.cmpi ule, %0, %cst: vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

///===------------------------------------===//
///  Tests of `eq` (equal)
///===------------------------------------===//

// CHECK-LABEL: @eq_constant_3
//       CHECK: %[[CST:.*]] = arith.constant dense<false> : vector<3xi1>
//       CHECK: return %[[CST]] : vector<3xi1>
func.func @eq_constant_3() -> vector<3xi1> {
  %cst = arith.constant dense<3> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  %1 = arith.cmpi eq, %0, %cst: vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @negative_eq_constant_2
//       CHECK: %[[CMP:.*]] = arith.cmpi
//       CHECK: return %[[CMP]]
func.func @negative_eq_constant_2() -> vector<3xi1> {
  %cst = arith.constant dense<2> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  %1 = arith.cmpi eq, %0, %cst: vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

///===------------------------------------===//
///  Tests of `ne` (not equal)
///===------------------------------------===//

// CHECK-LABEL: @ne_constant_3
//       CHECK: %[[CST:.*]] = arith.constant dense<true> : vector<3xi1>
//       CHECK: return %[[CST]] : vector<3xi1>
func.func @ne_constant_3() -> vector<3xi1> {
  %cst = arith.constant dense<3> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  %1 = arith.cmpi ne, %0, %cst: vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @negative_ne_constant_2
//       CHECK: %[[CMP:.*]] = arith.cmpi
//       CHECK: return %[[CMP]]
func.func @negative_ne_constant_2() -> vector<3xi1> {
  %cst = arith.constant dense<2> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  %1 = arith.cmpi ne, %0, %cst: vector<3xindex>
  return %1 : vector<3xi1>
}

