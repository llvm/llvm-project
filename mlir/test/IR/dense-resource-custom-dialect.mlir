// Verify text -> text roundtrip.
// RUN: mlir-opt %s -verify-diagnostics -split-input-file | FileCheck %s

// Test that dense_resource can reference blobs owned by a custom dialect using
// the "dialect::key" syntax.

// CHECK: attr = dense_resource<"test::blob1"> : tensor<3xi64>
"test.op_with_result_shape"() {attr = dense_resource<"test::blob1"> : tensor<3xi64> } : () -> ()

{-#
  dialect_resources: {
    test: {
      // CHECK: blob1: "0x08000000010000000000000002000000000000000300000000000000"
      blob1: "0x08000000010000000000000002000000000000000300000000000000"
    }
  }
#-}

// -----

// Verify that BuiltinDialect resources still work without a prefix.
// CHECK: attr = dense_resource<blob2> : tensor<2xi32>
"test.op_with_result_shape"() {attr = dense_resource<blob2> : tensor<2xi32> } : () -> ()

{-#
  dialect_resources: {
    builtin: {
      // CHECK: blob2: "0x0400000001000000020000000300000004000000"
      blob2: "0x0400000001000000020000000300000004000000"
    }
  }
#-}
