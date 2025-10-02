// RUN: mlir-opt %s -canonicalize="test-convergence" -split-input-file -allow-unregistered-dialect | FileCheck %s

///===----------------------------------------------===//
///  Tests of `StepCompareFolder`
///===----------------------------------------------===//


///===--------------===//
///  Tests of `ugt` (unsigned greater than)
///===--------------===//

// CHECK-LABEL: @check_ugt_constant_3_lhs
//       CHECK: %[[CST:.*]] = arith.constant dense<true> : vector<3xi1>
//       CHECK: return %[[CST]] : vector<3xi1>
func.func @check_ugt_constant_3_lhs() -> vector<3xi1> {
  %cst = arith.constant dense<3> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  // 3 > [0, 1, 2] => true
  %1 = arith.cmpi ugt, %cst, %0 : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @check_ugt_constant_2_lhs
//       CHECK: %[[CMP:.*]] = arith.cmpi
//       CHECK: return %[[CMP]]
func.func @check_ugt_constant_2_lhs() -> vector<3xi1> {
  %cst = arith.constant dense<2> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  // 2 > [0, 1, 2] => not constant
  %1 = arith.cmpi ugt, %cst, %0 : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @check_ugt_constant_1_lhs
//       CHECK: %[[CMP:.*]] = arith.cmpi
//       CHECK: return %[[CMP]]
func.func @check_ugt_constant_1_lhs() -> vector<3xi1> {
  %cst = arith.constant dense<1> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  // 1 > [0, 1, 2] => not constant
  %1 = arith.cmpi ugt, %cst, %0 : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @check_ugt_constant_3_rhs
//       CHECK: %[[CST:.*]] = arith.constant dense<false> : vector<3xi1>
//       CHECK: return %[[CST]] : vector<3xi1>
func.func @check_ugt_constant_3_rhs() -> vector<3xi1> {
  %cst = arith.constant dense<3> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  // [0, 1, 2] > 3 => false
  %1 = arith.cmpi ugt, %0, %cst : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @check_ugt_constant_2_rhs
//       CHECK: %[[CST:.*]] = arith.constant dense<false> : vector<3xi1>
//       CHECK: return %[[CST]] : vector<3xi1>
func.func @check_ugt_constant_2_rhs() -> vector<3xi1> {
  %cst = arith.constant dense<2> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  // [0, 1, 2] > 2 => false
  %1 = arith.cmpi ugt, %0, %cst : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @check_ugt_constant_1_rhs
//       CHECK: %[[CMP:.*]] = arith.cmpi
//       CHECK: return %[[CMP]]
func.func @check_ugt_constant_1_rhs() -> vector<3xi1> {
  %cst = arith.constant dense<1> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  // [0, 1, 2] > 1 => not constant
  %1 = arith.cmpi ugt, %0, %cst: vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

///===--------------===//
///  Tests of `uge` (unsigned greater than or equal)
///===--------------===//

// CHECK-LABEL: @check_uge_constant_3_lhs
//       CHECK: %[[CST:.*]] = arith.constant dense<true> : vector<3xi1>
//       CHECK: return %[[CST]] : vector<3xi1>
func.func @check_uge_constant_3_lhs() -> vector<3xi1> {
  %cst = arith.constant dense<3> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  // 3 >= [0, 1, 2] => true
  %1 = arith.cmpi uge, %cst, %0 : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @check_uge_constant_2_lhs
//       CHECK: %[[CST:.*]] = arith.constant dense<true> : vector<3xi1>
//       CHECK: return %[[CST]] : vector<3xi1>
func.func @check_uge_constant_2_lhs() -> vector<3xi1> {
  %cst = arith.constant dense<2> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  // 2 >= [0, 1, 2] => true
  %1 = arith.cmpi uge, %cst, %0 : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @check_uge_constant_1_lhs
//       CHECK: %[[CMP:.*]] = arith.cmpi
//       CHECK: return %[[CMP]]
func.func @check_uge_constant_1_lhs() -> vector<3xi1> {
  %cst = arith.constant dense<1> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  // 1 >= [0, 1, 2] => not constant
  %1 = arith.cmpi uge, %cst, %0 : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @check_uge_constant_3_rhs
//       CHECK: %[[CST:.*]] = arith.constant dense<false> : vector<3xi1>
//       CHECK: return %[[CST]] : vector<3xi1>
func.func @check_uge_constant_3_rhs() -> vector<3xi1> {
  %cst = arith.constant dense<3> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  // [0, 1, 2] >= 3 => false
  %1 = arith.cmpi uge, %0, %cst : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @check_uge_constant_2_rhs
//       CHECK: %[[CMP:.*]] = arith.cmpi
//       CHECK: return %[[CMP]]
func.func @check_uge_constant_2_rhs() -> vector<3xi1> {
  %cst = arith.constant dense<2> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  // [0, 1, 2] >= 2 => not constant
  %1 = arith.cmpi uge, %0, %cst : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @check_uge_constant_1_rhs
//       CHECK: %[[CMP:.*]] = arith.cmpi
//       CHECK: return %[[CMP]]
func.func @check_uge_constant_1_rhs() -> vector<3xi1> {
  %cst = arith.constant dense<1> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  // [0, 1, 2] >= 1 => not constant
  %1 = arith.cmpi uge, %0, %cst: vector<3xindex>
  return %1 : vector<3xi1>
}

// -----



///===--------------===//
///  Tests of `ult` (unsigned less than)
///===--------------===//

// CHECK-LABEL: @check_ult_constant_3_lhs
//       CHECK: %[[CST:.*]] = arith.constant dense<false> : vector<3xi1>
//       CHECK: return %[[CST]] : vector<3xi1>
func.func @check_ult_constant_3_lhs() -> vector<3xi1> {
  %cst = arith.constant dense<3> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  %1 = arith.cmpi ult, %cst, %0 : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @check_ult_constant_2_lhs
//       CHECK: %[[CST:.*]] = arith.constant dense<false> : vector<3xi1>
//       CHECK: return %[[CST]] : vector<3xi1>
func.func @check_ult_constant_2_lhs() -> vector<3xi1> {
  %cst = arith.constant dense<2> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  %1 = arith.cmpi ult, %cst, %0 : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @check_ult_constant_1_lhs
//       CHECK: %[[CMP:.*]] = arith.cmpi
//       CHECK: return %[[CMP]]
func.func @check_ult_constant_1_lhs() -> vector<3xi1> {
  %cst = arith.constant dense<1> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  %1 = arith.cmpi ult, %cst, %0 : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @check_ult_constant_3_rhs
//       CHECK: %[[CST:.*]] = arith.constant dense<true> : vector<3xi1>
//       CHECK: return %[[CST]] : vector<3xi1>
func.func @check_ult_constant_3_rhs() -> vector<3xi1> {
  %cst = arith.constant dense<3> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  %1 = arith.cmpi ult, %0, %cst : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @check_ult_constant_2_rhs
//       CHECK: %[[CMP:.*]] = arith.cmpi
//       CHECK: return %[[CMP]]
func.func @check_ult_constant_2_rhs() -> vector<3xi1> {
  %cst = arith.constant dense<2> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  %1 = arith.cmpi ult, %0, %cst : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @check_ult_constant_1_rhs
//       CHECK: %[[CMP:.*]] = arith.cmpi
//       CHECK: return %[[CMP]]
func.func @check_ult_constant_1_rhs() -> vector<3xi1> {
  %cst = arith.constant dense<1> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  %1 = arith.cmpi ult, %0, %cst: vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

///===--------------===//
///  Tests of `ule` (unsigned less than or equal)
///===--------------===//

// CHECK-LABEL: @check_ule_constant_3_lhs
//       CHECK: %[[CST:.*]] = arith.constant dense<false> : vector<3xi1>
//       CHECK: return %[[CST]] : vector<3xi1>
func.func @check_ule_constant_3_lhs() -> vector<3xi1> {
  %cst = arith.constant dense<3> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  %1 = arith.cmpi ule, %cst, %0 : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @check_ule_constant_2_lhs
//       CHECK: %[[CMP:.*]] = arith.cmpi
//       CHECK: return %[[CMP]]
func.func @check_ule_constant_2_lhs() -> vector<3xi1> {
  %cst = arith.constant dense<2> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  %1 = arith.cmpi ule, %cst, %0 : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @check_ule_constant_1_lhs
//       CHECK: %[[CMP:.*]] = arith.cmpi
//       CHECK: return %[[CMP]]
func.func @check_ule_constant_1_lhs() -> vector<3xi1> {
  %cst = arith.constant dense<1> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  %1 = arith.cmpi ule, %cst, %0 : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @check_ule_constant_3_rhs
//       CHECK: %[[CST:.*]] = arith.constant dense<true> : vector<3xi1>
//       CHECK: return %[[CST]] : vector<3xi1>
func.func @check_ule_constant_3_rhs() -> vector<3xi1> {
  %cst = arith.constant dense<3> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  %1 = arith.cmpi ule, %0, %cst : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @check_ule_constant_2_rhs
//       CHECK: %[[CST:.*]] = arith.constant dense<true> : vector<3xi1>
//       CHECK: return %[[CST]] : vector<3xi1>
func.func @check_ule_constant_2_rhs() -> vector<3xi1> {
  %cst = arith.constant dense<2> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  %1 = arith.cmpi ule, %0, %cst : vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @check_ule_constant_1_rhs
//       CHECK: %[[CMP:.*]] = arith.cmpi
//       CHECK: return %[[CMP]]
func.func @check_ule_constant_1_rhs() -> vector<3xi1> {
  %cst = arith.constant dense<1> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  %1 = arith.cmpi ule, %0, %cst: vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

///===--------------===//
///  Tests of `eq` (equal)
///===--------------===//

// CHECK-LABEL: @check_eq_constant_3
//       CHECK: %[[CST:.*]] = arith.constant dense<false> : vector<3xi1>
//       CHECK: return %[[CST]] : vector<3xi1>
func.func @check_eq_constant_3() -> vector<3xi1> {
  %cst = arith.constant dense<3> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  %1 = arith.cmpi eq, %0, %cst: vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @check_eq_constant_2
//       CHECK: %[[CMP:.*]] = arith.cmpi
//       CHECK: return %[[CMP]]
func.func @check_eq_constant_2() -> vector<3xi1> {
  %cst = arith.constant dense<2> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  %1 = arith.cmpi eq, %0, %cst: vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

///===--------------===//
///  Tests of `ne` (not equal)
///===--------------===//

// CHECK-LABEL: @check_ne_constant_3
//       CHECK: %[[CST:.*]] = arith.constant dense<true> : vector<3xi1>
//       CHECK: return %[[CST]] : vector<3xi1>
func.func @check_ne_constant_3() -> vector<3xi1> {
  %cst = arith.constant dense<3> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  %1 = arith.cmpi ne, %0, %cst: vector<3xindex>
  return %1 : vector<3xi1>
}

// -----

// CHECK-LABEL: @check_ne_constant_2
//       CHECK: %[[CMP:.*]] = arith.cmpi
//       CHECK: return %[[CMP]]
func.func @check_ne_constant_2() -> vector<3xi1> {
  %cst = arith.constant dense<2> : vector<3xindex>
  %0 = vector.step : vector<3xindex>
  %1 = arith.cmpi ne, %0, %cst: vector<3xindex>
  return %1 : vector<3xi1>
}

