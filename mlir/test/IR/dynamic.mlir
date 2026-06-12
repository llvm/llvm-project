// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics | FileCheck %s
// Verify that extensible dialects can register dynamic operations and types.

//===----------------------------------------------------------------------===//
// Dynamic type
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @succeededDynamicTypeVerifier
func.func @succeededDynamicTypeVerifier() {
  // CHECK: %{{.*}} = "unregistered_op"() : () -> !test.dynamic_singleton
  "unregistered_op"() : () -> !test.dynamic_singleton
  // CHECK-NEXT: "unregistered_op"() : () -> !test.dynamic_pair<i32, f64>
  "unregistered_op"() : () -> !test.dynamic_pair<i32, f64>
  // CHECK-NEXT: %{{.*}} = "unregistered_op"() : () -> !test.dynamic_pair<!test.dynamic_pair<i32, f64>, !test.dynamic_singleton>
  "unregistered_op"() : () -> !test.dynamic_pair<!test.dynamic_pair<i32, f64>, !test.dynamic_singleton>
  return
}

// -----

func.func @failedDynamicTypeVerifier() {
  // expected-error@+1 {{expected 0 type arguments, but had 1}}
  "unregistered_op"() : () -> !test.dynamic_singleton<f64>
  return
}

// -----

func.func @failedDynamicTypeVerifier2() {
  // expected-error@+1 {{expected 2 type arguments, but had 1}}
  "unregistered_op"() : () -> !test.dynamic_pair<f64>
  return
}

// -----

// CHECK-LABEL: func @customTypeParserPrinter
func.func @customTypeParserPrinter() {
  // CHECK: "unregistered_op"() : () -> !test.dynamic_custom_assembly_format<f32:f64>
  "unregistered_op"() : () -> !test.dynamic_custom_assembly_format<f32 : f64>
  return
}

// -----

//===----------------------------------------------------------------------===//
// Dynamic attribute
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @succeededDynamicAttributeVerifier
func.func @succeededDynamicAttributeVerifier() {
  // CHECK: "unregistered_op"() {test_attr = #test.dynamic_singleton} : () -> ()
  "unregistered_op"() {test_attr = #test.dynamic_singleton} : () -> ()
  // CHECK-NEXT: "unregistered_op"() {test_attr = #test.dynamic_pair<3 : i32, 5 : i32>} : () -> ()
  "unregistered_op"() {test_attr = #test.dynamic_pair<3 : i32, 5 : i32>} : () -> ()
  // CHECK-NEXT: "unregistered_op"() {test_attr = #test.dynamic_pair<#test.dynamic_pair<3 : i32, 5 : i32>, f64>} : () -> ()
  "unregistered_op"() {test_attr = #test.dynamic_pair<#test.dynamic_pair<3 : i32, 5 : i32>, f64>} : () -> ()
  return
}

// -----

func.func @failedDynamicAttributeVerifier() {
  // expected-error@+1 {{expected 0 attribute arguments, but had 1}}
  "unregistered_op"() {test_attr = #test.dynamic_singleton<f64>} : () -> ()
  return
}

// -----

func.func @failedDynamicAttributeVerifier2() {
  // expected-error@+1 {{expected 2 attribute arguments, but had 1}}
  "unregistered_op"() {test_attr = #test.dynamic_pair<f64>} : () -> ()
  return
}

// -----

// CHECK-LABEL: func @customAttributeParserPrinter
func.func @customAttributeParserPrinter() {
  // CHECK: "unregistered_op"() {test_attr = #test.dynamic_custom_assembly_format<f32:f64>} : () -> ()
  "unregistered_op"() {test_attr = #test.dynamic_custom_assembly_format<f32:f64>} : () -> ()
  return
}

//===----------------------------------------------------------------------===//
// Dynamic op
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @succeededDynamicOpVerifier
func.func @succeededDynamicOpVerifier(%a: f32) {
  // CHECK: "test.dynamic_generic"() : () -> ()
  // CHECK-NEXT: %{{.*}} = "test.dynamic_generic"(%{{.*}}) : (f32) -> f64
  // CHECK-NEXT: %{{.*}}:2 = "test.dynamic_one_operand_two_results"(%{{.*}}) : (f32) -> (f64, f64)
  "test.dynamic_generic"() : () -> ()
  "test.dynamic_generic"(%a) : (f32) -> f64
  "test.dynamic_one_operand_two_results"(%a) : (f32) -> (f64, f64)
  return
}

// -----

func.func @failedDynamicOpVerifier() {
  // expected-error@+1 {{expected 1 operand, but had 0}}
  "test.dynamic_one_operand_two_results"() : () -> (f64, f64)
  return
}

// -----

func.func @failedDynamicOpVerifier2(%a: f32) {
  // expected-error@+1 {{expected 2 results, but had 0}}
  "test.dynamic_one_operand_two_results"(%a) : (f32) -> ()
  return
}

// -----

// CHECK-LABEL: func @customOpParserPrinter
func.func @customOpParserPrinter() {
  // CHECK: test.dynamic_custom_parser_printer custom_keyword
  test.dynamic_custom_parser_printer custom_keyword
  return
}

// -----

func.func @failedDynamicGenericOpNoTerminator() {
  // expected-error@+1 {{empty block: expect at least a terminator}}
  "test.dynamic_generic"() ({
    ^bb1:
  }) : () -> ()
  return
}

// -----

func.func @dynamicTerminatorOp() {
  // CHECK: "test.dynamic_generic"()
  "test.dynamic_generic"() ({
    ^bb1:
      // CHECK: test.dynamic_terminator"()
      "test.dynamic_terminator"() : () -> ()
  }) : () -> ()
  return
}

// -----

func.func @failedDynamicTerminatorOp() {
  "test.dynamic_generic"() ({
    ^bb1:
      // expected-error@+1 {{'test.dynamic_terminator' op must be the last operation in the parent block}}
      "test.dynamic_terminator"() : () -> ()
      "test.dynamic_generic"() : () -> ()
  }) : () -> ()
  return
}

// -----

func.func @dynamicNoTerminatorOp() {
  // CHECK: "test.dynamic_noterminator"()
  "test.dynamic_noterminator"() ({
    ^bb1:
  }) : () -> ()
  return
}

//===----------------------------------------------------------------------===//
// Dynamic dialect
//===----------------------------------------------------------------------===//

// -----

// Check that the verifier of a dynamic operation in a dynamic dialect
// can fail. This shows that the dialect is correctly registered.

func.func @failedDynamicDialectOpVerifier() {
  // expected-error@+1 {{expected a single result, no operands and no regions}}
  "test_dyn.one_result"() : () -> ()
  return
}
