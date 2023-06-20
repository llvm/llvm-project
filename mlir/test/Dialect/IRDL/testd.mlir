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

func.func @failedEqConstraint() {
  // expected-error@+1 {{expected 'i32' but got 'i64'}}
  "testd.eq"() : () -> i64
  return
}

// -----

//===----------------------------------------------------------------------===//
// AnyOf constraint
//===----------------------------------------------------------------------===//

func.func @succeededAnyOfConstraint() {
  // CHECK: "testd.anyof"() : () -> i32
  "testd.anyof"() : () -> i32
  // CHECK: "testd.anyof"() : () -> i64
  "testd.anyof"() : () -> i64
  return
}

// -----

func.func @failedAnyOfConstraint() {
  // expected-error@+1 {{'i1' does not satisfy the constraint}}
  "testd.anyof"() : () -> i1
  return
}

// -----

//===----------------------------------------------------------------------===//
// AllOf constraint
//===----------------------------------------------------------------------===//

func.func @succeededAllOfConstraint() {
  // CHECK: "testd.all_of"() : () -> i64
  "testd.all_of"() : () -> i64
  return
}

// -----

func.func @failedAllOfConstraint1() {
  // expected-error@+1 {{'i1' does not satisfy the constraint}}
  "testd.all_of"() : () -> i1
  return
}

// -----

func.func @failedAllOfConstraint2() {
  // expected-error@+1 {{expected 'i64' but got 'i32'}}
  "testd.all_of"() : () -> i32
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
  // CHECK: "testd.dynbase"() : () -> !testd.parametric<i64>
  "testd.dynbase"() : () -> !testd.parametric<i64>
  // CHECK: "testd.dynbase"() : () -> !testd.parametric<!testd.parametric<i64>>
  "testd.dynbase"() : () -> !testd.parametric<!testd.parametric<i64>>
  return
}

// -----

func.func @failedDynBaseConstraint() {
  // expected-error@+1 {{expected base type 'testd.parametric' but got 'i32'}}
  "testd.dynbase"() : () -> i32
  return
}

// -----

//===----------------------------------------------------------------------===//
// Dynamic parameters constraint
//===----------------------------------------------------------------------===//

func.func @succeededDynParamsConstraint() {
  // CHECK: "testd.dynparams"() : () -> !testd.parametric<i32>
  "testd.dynparams"() : () -> !testd.parametric<i32>
  // CHECK: "testd.dynparams"() : () -> !testd.parametric<i64>
  "testd.dynparams"() : () -> !testd.parametric<i64>
  return
}

// -----

func.func @failedDynParamsConstraintBase() {
  // expected-error@+1 {{expected base type 'testd.parametric' but got 'i32'}}
  "testd.dynparams"() : () -> i32
  return
}

// -----

func.func @failedDynParamsConstraintParam() {
  // expected-error@+1 {{'i1' does not satisfy the constraint}}
  "testd.dynparams"() : () -> !testd.parametric<i1>
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

// -----

func.func @failedConstraintVars() {
  // expected-error@+1 {{expected 'i64' but got 'i32'}}
  "testd.constraint_vars"() : () -> (i64, i32)
  return
}

// -----

//===----------------------------------------------------------------------===//
// Constraint attributes
//===----------------------------------------------------------------------===//

func.func @succeededAttrs() {
  // CHECK: "testd.attrs"() {attr1 = i32, attr2 = i64} : () -> ()
  "testd.attrs"() {attr1 = i32, attr2 = i64} : () -> ()
  return
}

// -----

func.func @failedAttrsMissingAttr() {
  // expected-error@+1 {{attribute "attr2" is expected but not provided}}
  "testd.attrs"() {attr1 = i32} : () -> ()
  return
}

// -----

func.func @failedAttrsConstraint() {
  // expected-error@+1 {{expected 'i32' but got 'i64'}}
  "testd.attrs"() {attr1 = i64, attr2 = i64} : () -> ()
  return
}

// -----

func.func @failedAttrsConstraint2() {
  // expected-error@+1 {{expected 'i64' but got 'i32'}}
  "testd.attrs"() {attr1 = i32, attr2 = i32} : () -> ()
  return
}
