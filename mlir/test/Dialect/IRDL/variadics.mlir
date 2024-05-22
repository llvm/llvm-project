// RUN: mlir-opt %s --irdl-file=%S/variadics.irdl.mlir -split-input-file -verify-diagnostics | FileCheck %s

//===----------------------------------------------------------------------===//
// Single operand
//===----------------------------------------------------------------------===//

// Test an operation with a single operand.
func.func @testSingleOperand(%x: i32) {
  "testvar.single_operand"(%x) : (i32) -> ()
  // CHECK: "testvar.single_operand"(%{{.*}}) : (i32) -> ()
  return
}

// -----

// Test an operation with a single operand definition and a wrong number of operands.
func.func @testSingleOperandFail(%x: i32) {
  // expected-error@+1 {{op expects exactly 1 operands, but got 2}}
  "testvar.single_operand"(%x, %x) : (i32, i32) -> ()  
  return
}

// -----

// Test an operation with a single operand definition and a wrong number of operands.
func.func @testSingleOperandFail() {
  // expected-error@+1 {{op expects exactly 1 operands, but got 0}}
  "testvar.single_operand"() : () -> ()  
  return
}

// -----


//===----------------------------------------------------------------------===//
// Variadic operand
//===----------------------------------------------------------------------===//

// Test an operation with a single variadic operand.
func.func @testVarOperand(%x: i16, %y: i32, %z: i64) {
  "testvar.var_operand"(%x, %z) : (i16, i64) -> ()
  // CHECK: "testvar.var_operand"(%{{.*}}, %{{.*}}) : (i16, i64) -> ()
  "testvar.var_operand"(%x, %y, %z) : (i16, i32, i64) -> ()
  // CHECK-NEXT: "testvar.var_operand"(%{{.*}}, %{{.*}}, %{{.*}}) : (i16, i32, i64) -> ()
  "testvar.var_operand"(%x, %y, %y, %z) : (i16, i32, i32, i64) -> ()
  // CHECK-NEXT: "testvar.var_operand"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i16, i32, i32, i64) -> ()
  "testvar.var_operand"(%x, %y, %y, %y, %z) : (i16, i32, i32, i32, i64) -> ()
  // CHECK-NEXT: "testvar.var_operand"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i16, i32, i32, i32, i64) -> ()
  return
}

// -----

// Check that the verifier of a variadic operand  fails if the variadic is given
// a wrong type.
func.func @testVarOperandFail(%x: i16, %y: i64, %z: i64) {
  // expected-error@+1 {{expected 'i32' but got 'i64'}}
  "testvar.var_operand"(%x, %y, %z) : (i16, i64, i64) -> ()
  return
}

// -----

// Check that the verifier of a variadic operand fails if the variadic is given
// a wrong type on the second value.
func.func @testVarOperandFail(%x: i16, %y1: i32, %y2: i64, %z: i64) {
  // expected-error@+1 {{expected 'i32' but got 'i64'}}
  "testvar.var_operand"(%x, %y1, %y2, %z) : (i16, i32, i64, i64) -> ()
  return
}

// -----

// Check that if we do not give enough operands, the verifier fails.
func.func @testVarOperandFail() {
  // expected-error@+1 {{op expects at least 2 operands, but got 0}}
  "testvar.var_operand"() : () -> ()
  return
}

// -----

//===----------------------------------------------------------------------===//
// Optional operand
//===----------------------------------------------------------------------===//


// Test an operation with a single optional operand.
func.func @testOptOperand(%x: i16, %y: i32, %z: i64) {
  "testvar.opt_operand"(%x, %z) : (i16, i64) -> ()
  // CHECK: "testvar.opt_operand"(%{{.*}}, %{{.*}}) : (i16, i64) -> ()
  "testvar.opt_operand"(%x, %y, %z) : (i16, i32, i64) -> ()
  // CHECK-NEXT: "testvar.opt_operand"(%{{.*}}, %{{.*}}, %{{.*}}) : (i16, i32, i64) -> ()
  return
}

// -----

// Check that the verifier of an optional operand fails if the variadic is
// given a wrong type.
func.func @testOptOperandFail(%x: i16, %y: i64, %z: i64) {
  // expected-error@+1 {{expected 'i32' but got 'i64'}}
  "testvar.opt_operand"(%x, %y, %z) : (i16, i64, i64) -> ()
  return
}

// -----

// Check that the verifier of an optional operand fails if there are too
// many operands.
func.func @testOptOperandFail(%x: i16, %y: i32, %z: i64) {
  // expected-error@+1 {{op expects at most 3 operands, but got 4}}
  "testvar.opt_operand"(%x, %y, %y, %z) : (i16, i32, i32, i64) -> ()
  return
}

// -----

// Check that the verifier of an optional operand fails if there are not
// enough operands.
func.func @testOptOperandFail(%x: i16) {
  // expected-error@+1 {{op expects at least 2 operands, but got 1}}
  "testvar.opt_operand"(%x) : (i16) -> ()
  return
}

// -----

//===----------------------------------------------------------------------===//
// Multiple variadic
//===----------------------------------------------------------------------===//

// Check that an operation with multiple variadics expects the segment size
// attribute
func.func @testMultOperandsMissingSegment(%x: i16, %z: i64) {
  // expected-error@+1 {{'operand_segment_sizes' attribute is expected but not provided}}
  "testvar.var_and_opt_operand"(%x, %x, %z) : (i16, i16, i64) -> ()
  return
}

// -----

// Check that an operation with multiple variadics expects the segment size
// attribute of the right type
func.func @testMultOperandsWrongSegmentType(%x: i16, %z: i64) {
  // expected-error@+1 {{'operand_segment_sizes' attribute is expected to be a dense i32 array}}
  "testvar.var_and_opt_operand"(%x, %x, %z) {operand_segment_sizes = i32} : (i16, i16, i64) -> ()
  return
}

// -----

// Check that an operation with multiple variadics with the right segment size
// verifies.
func.func @testMultOperands(%x: i16, %y: i32, %z: i64) {
  "testvar.var_and_opt_operand"(%x, %x, %z) {operand_segment_sizes = array<i32: 2, 0, 1>} : (i16, i16, i64) -> ()
  // CHECK: "testvar.var_and_opt_operand"(%{{.*}}, %{{.*}}, %{{.*}}) {operand_segment_sizes = array<i32: 2, 0, 1>} : (i16, i16, i64) -> ()
  "testvar.var_and_opt_operand"(%x, %x, %y, %z) {operand_segment_sizes = array<i32: 2, 1, 1>} : (i16, i16, i32, i64) -> ()
  // CHECK-NEXT: "testvar.var_and_opt_operand"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {operand_segment_sizes = array<i32: 2, 1, 1>} : (i16, i16, i32, i64) -> ()
  "testvar.var_and_opt_operand"(%y, %z) {operand_segment_sizes = array<i32: 0, 1, 1>} : (i32, i64) -> ()
  // CHECK-NEXT: "testvar.var_and_opt_operand"(%{{.*}}, %{{.*}}) {operand_segment_sizes = array<i32: 0, 1, 1>} : (i32, i64) -> ()
  return
}

// -----

// Check that the segment sizes expects non-negative values
func.func @testMultOperandsSegmentNegative() {
  // expected-error@+1 {{'operand_segment_sizes' attribute for specifying operand segments must have non-negative values}}
  "testvar.var_and_opt_operand"() {operand_segment_sizes = array<i32: 2, -1, 1>} : () -> ()
  return
}

// -----

// Check that the segment sizes expects 1 for single values
func.func @testMultOperandsSegmentWrongSingle() {
  // expected-error@+1 {{element 2 in 'operand_segment_sizes' attribute must be equal to 1}}
  "testvar.var_and_opt_operand"() {operand_segment_sizes = array<i32: 0, 0, 0>} : () -> ()
  return
}

// -----

// Check that the segment sizes expects not more than 1 for optional values
func.func @testMultOperandsSegmentWrongOptional() {
  // expected-error@+1 {{element 1 in 'operand_segment_sizes' attribute must be equal to 0 or 1}}
  "testvar.var_and_opt_operand"() {operand_segment_sizes = array<i32: 0, 2, 0>} : () -> ()
  return
}

// -----

// Check that the sum of the segment sizes should be equal to the number of operands
func.func @testMultOperandsSegmentWrongOptional(%y: i32, %z: i64) {
  // expected-error@+1 {{sum of elements in 'operand_segment_sizes' attribute must be equal to the number of operands}}
  "testvar.var_and_opt_operand"(%y, %z) {operand_segment_sizes = array<i32: 0, 0, 1>} : (i32, i64) -> ()
  return
}

// -----

//===----------------------------------------------------------------------===//
// Single result
//===----------------------------------------------------------------------===//

// Test an operation with a single result.
func.func @testSingleResult() {
  %x = "testvar.single_result"() : () -> i32
  // CHECK: %{{.*}} = "testvar.single_result"() : () -> i32
  return
}

// -----

// Test an operation with a single result definition and a wrong number of results.
func.func @testSingleResultFail() {
  // expected-error@+1 {{op expects exactly 1 results, but got 2}}
  %x, %y = "testvar.single_result"() : () -> (i32, i32)  
  return
}

// -----

// Test an operation with a single result definition and a wrong number of results.
func.func @testSingleResultFail() {
  // expected-error@+1 {{op expects exactly 1 results, but got 0}}
  "testvar.single_result"() : () -> ()  
  return
}

// -----


//===----------------------------------------------------------------------===//
// Variadic result
//===----------------------------------------------------------------------===//


// Test an operation with a single variadic result.
func.func @testVarResult() {
  "testvar.var_result"() : () -> (i16, i64)
  // CHECK: "testvar.var_result"() : () -> (i16, i64)
  "testvar.var_result"() : () -> (i16, i32, i64)
  // CHECK-NEXT: "testvar.var_result"() : () -> (i16, i32, i64)
  "testvar.var_result"() : () -> (i16, i32, i32, i64)
  // CHECK-NEXT: "testvar.var_result"() : () -> (i16, i32, i32, i64)
  "testvar.var_result"() : () -> (i16, i32, i32, i32, i64)
  // CHECK-NEXT: "testvar.var_result"() : () -> (i16, i32, i32, i32, i64)
  return
}

// -----

// Check that the verifier of a variadic result  fails if the variadic is given
// a wrong type.
func.func @testVarResultFail() {
  // expected-error@+1 {{expected 'i32' but got 'i64'}}
  "testvar.var_result"() : () -> (i16, i64, i64)
  return
}

// -----

// Check that the verifier of a variadic result fails if the variadic is given
// a wrong type on the second value.
func.func @testVarResultFail() {
  // expected-error@+1 {{expected 'i32' but got 'i64'}}
  "testvar.var_result"() : () -> (i16, i32, i64, i64)
  return
}

// -----

// Check that if we do not give enough results, the verifier fails.
func.func @testVarResultFail() {
  // expected-error@+1 {{op expects at least 2 results, but got 0}}
  "testvar.var_result"() : () -> ()
  return
}

// -----

//===----------------------------------------------------------------------===//
// Optional result
//===----------------------------------------------------------------------===//


// Test an operation with a single optional result.
func.func @testOptResult() {
  "testvar.opt_result"() : () -> (i16, i64)
  // CHECK: "testvar.opt_result"() : () -> (i16, i64)
  "testvar.opt_result"() : () -> (i16, i32, i64)
  // CHECK-NEXT: "testvar.opt_result"() : () -> (i16, i32, i64)
  return
}

// -----

// Check that the verifier of an optional result fails if the variadic is
// given a wrong type.
func.func @testOptResultFail() {
  // expected-error@+1 {{expected 'i32' but got 'i64'}}
  "testvar.opt_result"() : () -> (i16, i64, i64)
  return
}

// -----

// Check that the verifier of an optional result fails if there are too
// many results.
func.func @testOptResultFail() {
  // expected-error@+1 {{op expects at most 3 results, but got 4}}
  "testvar.opt_result"() : () -> (i16, i32, i32, i64)
  return
}

// -----

// Check that the verifier of an optional result fails if there are not
// enough results.
func.func @testOptResultFail() {
  // expected-error@+1 {{op expects at least 2 results, but got 1}}
  "testvar.opt_result"() : () -> (i16)
  return
}

// -----

//===----------------------------------------------------------------------===//
// Multiple variadic
//===----------------------------------------------------------------------===//

// Check that an operation with multiple variadics expects the segment size
// attribute
func.func @testMultResultsMissingSegment() {
  // expected-error@+1 {{'result_segment_sizes' attribute is expected but not provided}}
  "testvar.var_and_opt_result"() : () -> (i16, i16, i64)
  return
}

// -----

// Check that an operation with multiple variadics expects the segment size
// attribute of the right type
func.func @testMultResultsWrongSegmentType() {
  // expected-error@+1 {{'result_segment_sizes' attribute is expected to be a dense i32 array}}
  "testvar.var_and_opt_result"() {result_segment_sizes = i32} : () -> (i16, i16, i64)
  return
}

// -----

// Check that an operation with multiple variadics with the right segment size
// verifies.
func.func @testMultResults() {
  "testvar.var_and_opt_result"() {result_segment_sizes = array<i32: 2, 0, 1>} : () -> (i16, i16, i64)
  // CHECK: "testvar.var_and_opt_result"() {result_segment_sizes = array<i32: 2, 0, 1>} : () -> (i16, i16, i64)
  "testvar.var_and_opt_result"() {result_segment_sizes = array<i32: 2, 1, 1>} : () -> (i16, i16, i32, i64)
  // CHECK-NEXT: "testvar.var_and_opt_result"() {result_segment_sizes = array<i32: 2, 1, 1>} : () -> (i16, i16, i32, i64)
  "testvar.var_and_opt_result"() {result_segment_sizes = array<i32: 0, 1, 1>} : () -> (i32, i64)
  // CHECK-NEXT: "testvar.var_and_opt_result"() {result_segment_sizes = array<i32: 0, 1, 1>} : () -> (i32, i64)
  return
}

// -----

// Check that the segment sizes expects non-negative values
func.func @testMultResultsSegmentNegative() {
  // expected-error@+1 {{'result_segment_sizes' attribute for specifying result segments must have non-negative values}}
  "testvar.var_and_opt_result"() {result_segment_sizes = array<i32: 2, -1, 1>} : () -> ()
  return
}

// -----

// Check that the segment sizes expects 1 for single values
func.func @testMultResultsSegmentWrongSingle() {
  // expected-error@+1 {{element 2 in 'result_segment_sizes' attribute must be equal to 1}}
  "testvar.var_and_opt_result"() {result_segment_sizes = array<i32: 0, 0, 0>} : () -> ()
  return
}

// -----

// Check that the segment sizes expects not more than 1 for optional values
func.func @testMultResultsSegmentWrongOptional() {
  // expected-error@+1 {{element 1 in 'result_segment_sizes' attribute must be equal to 0 or 1}}
  "testvar.var_and_opt_result"() {result_segment_sizes = array<i32: 0, 2, 0>} : () -> ()
  return
}

// -----

// Check that the sum of the segment sizes should be equal to the number of results
func.func @testMultResultsSegmentWrongOptional() {
  // expected-error@+1 {{sum of elements in 'result_segment_sizes' attribute must be equal to the number of results}}
  "testvar.var_and_opt_result"() {result_segment_sizes = array<i32: 0, 0, 1>} : () -> (i32, i64)
  return
}
