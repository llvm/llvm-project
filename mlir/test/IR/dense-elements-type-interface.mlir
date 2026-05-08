// RUN: mlir-opt %s -verify-diagnostics -split-input-file | FileCheck %s

// Test dense elements attribute with custom element type using DenseElementTypeInterface.
// Uses the new type-first syntax: dense<TYPE : [ATTR, ...]>
// Note: The type is embedded in the attribute, so it's not printed again at the end.

// CHECK-LABEL: func @dense_custom_element_type
func.func @dense_custom_element_type() {
  // CHECK: "test.dummy"() {attr = dense<tensor<3x!test.dense_element> : [1 : i32, 2 : i32, 3 : i32]>}
  "test.dummy"() {attr = dense<tensor<3x!test.dense_element> : [1 : i32, 2 : i32, 3 : i32]>} : () -> ()
  return
}

// -----

// CHECK-LABEL: func @dense_custom_element_type_2d
func.func @dense_custom_element_type_2d() {
  // CHECK: "test.dummy"() {attr = dense<tensor<2x2x!test.dense_element> : {{\[}}{{\[}}1 : i32, 2 : i32], [3 : i32, 4 : i32]]>}
  "test.dummy"() {attr = dense<tensor<2x2x!test.dense_element> : [[1 : i32, 2 : i32], [3 : i32, 4 : i32]]>} : () -> ()
  return
}

// -----

// CHECK-LABEL: func @dense_custom_element_splat
func.func @dense_custom_element_splat() {
  // CHECK: "test.dummy"() {attr = dense<tensor<4x!test.dense_element> : 42 : i32>}
  "test.dummy"() {attr = dense<tensor<4x!test.dense_element> : 42 : i32>} : () -> ()
  return
}

// -----

// CHECK-LABEL func @dense_i32_1d
func.func @dense_i32_1d() {
  // The default assembly format for int, index, float, complex element types is
  // the literal-first syntax. Such a dense elements attribute can be parsed
  // with the type-first syntax, but it will come back with the literal-first
  // syntax.
  // CHECK: "test.dummy"() {attr = dense<[1, 2, 3]> : tensor<3xi32>} : () -> ()
  "test.dummy"() {attr = dense<tensor<3xi32> : [1 : i32, 2 : i32, 3 : i32]>} : () -> ()
  return
}

// -----

func.func @invalid_element() {
  // expected-error @+1 {{expected attribute value}}
  "test.dummy"() {attr = dense<tensor<3xi32> : [foo]>} : () -> ()
  return
}

// -----

func.func @incompatible_attribute() {
  // expected-error @+1 {{incompatible attribute for element type}}
  "test.dummy"() {attr = dense<tensor<3xi32> : ["foo"]>} : () -> ()
  return
}

// -----

func.func @shape_mismatch() {
  // expected-error @+1 {{expected 3 elements in dimension, got 2}}
  "test.dummy"() {attr = dense<tensor<3xi32> : [1 : i32, 2 : i32]>} : () -> ()
  return
}

// -----

func.func @dynamic_shape() {
  // expected-error @+1 {{dense elements type must have static shape}}
  "test.dummy"() {attr = dense<tensor<?xi32> : [1 : i32, 2 : i32, 3 : i32]>} : () -> ()
  return
}

// -----

func.func @invalid_type() {
  // expected-error @+1 {{expected a shaped type for dense elements}}
  "test.dummy"() {attr = dense<i32 : [1 : i32, 2 : i32, 3 : i32]>} : () -> ()
  return
}
