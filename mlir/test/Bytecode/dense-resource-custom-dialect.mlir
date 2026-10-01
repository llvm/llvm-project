// Verify text -> bytecode -> text roundtrip for dense_resource with custom
// dialect resource handles.

// RUN: mlir-opt -emit-bytecode %s | mlir-opt | FileCheck %s

// CHECK-LABEL: @CustomDialectResources
module @CustomDialectResources attributes {
  // Custom dialect resource uses "dialect::key" syntax.
  // CHECK: bytecode.test1 = dense_resource<"test::custom_blob"> : tensor<3xi64>
  bytecode.test1 = dense_resource<"test::custom_blob"> : tensor<3xi64>,

  // Builtin dialect resource uses bare key syntax (backward compatible).
  // CHECK: bytecode.test2 = dense_resource<builtin_blob> : tensor<2xi32>
  bytecode.test2 = dense_resource<builtin_blob> : tensor<2xi32>
} {}

// Verify resource data roundtrips correctly for both dialects.
// CHECK-DAG: custom_blob: "0x08000000010000000000000002000000000000000300000000000000"
// CHECK-DAG: builtin_blob: "0x04000000AABBCCDDAABBCCDD"

{-#
  dialect_resources: {
    test: {
      custom_blob: "0x08000000010000000000000002000000000000000300000000000000"
    },
    builtin: {
      builtin_blob: "0x04000000AABBCCDDAABBCCDD"
    }
  }
#-}
