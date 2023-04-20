// RUN: mlir-opt %s --irdl-file=%S/testd.irdl.mlir -split-input-file -verify-diagnostics | FileCheck %s

//===----------------------------------------------------------------------===//
// Type or attribute constraint
//===----------------------------------------------------------------------===//

func.func @typeFitsType() {
  // CHECK: "testd.any"() : () -> !testd.parametric<i32>
  "testd.any"() : () -> !testd.parametric<i32>
  return
}

// -----

func.func @attrDoesntFitType() {
  "testd.any"() : () -> !testd.parametric<"foo">
  return
}

// -----

func.func @attrFitsAttr() {
  // CHECK: "testd.any"() : () -> !testd.attr_in_type_out<"foo">
  "testd.any"() : () -> !testd.attr_in_type_out<"foo">
  return
}

// -----

func.func @typeFitsAttr() {
  // CHECK: "testd.any"() : () -> !testd.attr_in_type_out<i32>
  "testd.any"() : () -> !testd.attr_in_type_out<i32>
  return
}

// -----

//===----------------------------------------------------------------------===//
// Equality constraint
//===----------------------------------------------------------------------===//

func.func @succeededEqConstraint() {
  // CHECK: "testd.eq"() : () -> i32
  "testd.eq"() : () -> i32
  return
}


// -----

//===----------------------------------------------------------------------===//
// Any constraint
//===----------------------------------------------------------------------===//

func.func @succeededAnyConstraint() {
  // CHECK: "testd.any"() : () -> i32
  "testd.any"() : () -> i32
  // CHECK: "testd.any"() : () -> i64
  "testd.any"() : () -> i64
  return
}

// -----

//===----------------------------------------------------------------------===//
// Dynamic base constraint
//===----------------------------------------------------------------------===//

func.func @succeededDynBaseConstraint() {
  // CHECK: "testd.dynbase"() : () -> !testd.parametric<i32>
  "testd.dynbase"() : () -> !testd.parametric<i32>
  // CHECK: "testd.dynbase"() : () -> !testd.parametric<!testd.parametric<i32>>
  "testd.dynbase"() : () -> !testd.parametric<!testd.parametric<i32>>
  return
}


// -----

//===----------------------------------------------------------------------===//
// Dynamic parameters constraint
//===----------------------------------------------------------------------===//

func.func @succeededDynParamsConstraint() {
  // CHECK: "testd.dynparams"() : () -> !testd.parametric<i32>
  "testd.dynparams"() : () -> !testd.parametric<i32>
  return
}

// -----

//===----------------------------------------------------------------------===//
// Constraint variables
//===----------------------------------------------------------------------===//

func.func @succeededConstraintVars() {
  // CHECK: "testd.constraint_vars"() : () -> (i32, i32)
  "testd.constraint_vars"() : () -> (i32, i32)
  return
}

// -----

func.func @succeededConstraintVars2() {
  // CHECK: "testd.constraint_vars"() : () -> (i64, i64)
  "testd.constraint_vars"() : () -> (i64, i64)
  return
}
