// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// Check that a typed attribute with a self type parameter round-trips through
// the custom assembly format, with its type inferred from the enclosing op.
// CHECK-LABEL: func.func @typed_attr_roundtrip
func.func @typed_attr_roundtrip() {
  // CHECK: test.typed_attr i32 = #test.attr_with_self_type_param
  test.typed_attr i32 = #test.attr_with_self_type_param

  // CHECK: test.typed_attr f32 = #test.attr_with_self_type_param
  test.typed_attr f32 = #test.attr_with_self_type_param
  return
}
