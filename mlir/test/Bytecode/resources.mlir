// RUN: mlir-opt -emit-bytecode %s | mlir-opt | FileCheck %s

// Bytecode currently does not support big-endian platforms
// UNSUPPORTED: target=s390x-{{.*}}

// CHECK-LABEL: @TestDialectResources
module @TestDialectResources attributes {
  // CHECK: bytecode.test = dense_resource<decl_resource> : tensor<2xui32>
  // CHECK: bytecode.test2 = dense_resource<resource> : tensor<4xf64>
  // CHECK: bytecode.test3 = dense_resource<resource_2> : tensor<4xf64>
  bytecode.test = dense_resource<decl_resource> : tensor<2xui32>,
  bytecode.test2 = dense_resource<resource> : tensor<4xf64>,
  bytecode.test3 = dense_resource<resource_2> : tensor<4xf64>
} {}

// CHECK: builtin: {
// CHECK-NEXT: resource: "0x08000000010000000000000002000000000000000300000000000000"
// CHECK-NEXT: resource_2: "0x08000000010000000000000002000000000000000300000000000000"

{-#
  dialect_resources: {
    builtin: {
      resource: "0x08000000010000000000000002000000000000000300000000000000",
      resource_2: "0x08000000010000000000000002000000000000000300000000000000"
    }
  }
#-}
