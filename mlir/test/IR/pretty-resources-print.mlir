// Check printing with --mlir-elide-resource-strings-if-larger elides printing large resources

// RUN: mlir-opt %s --mlir-elide-resource-strings-if-larger=20| FileCheck %s

// To ensure we print the resource keys, have reference to them
// CHECK: attr = dense_resource<blob1> : tensor<3xi64>
"test.blob1op"() {attr = dense_resource<blob1> : tensor<3xi64> } : () -> ()

// CHECK-NEXT: attr = dense_resource<blob2> : tensor<3xi64>
"test.blob2op"() {attr = dense_resource<blob2> : tensor<3xi64> } : () -> ()

// CHECK:      {-#
// CHECK-NEXT:   external_resources: {
// CHECK-NEXT:     external: {
// CHECK-NEXT:       bool: true,
// CHECK-NEXT:       string: "\22string\22"
// CHECK-NEXT:     },
// CHECK-NEXT:     other_stuff: {
// CHECK-NEXT:       bool: true
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: #-}

{-#
  dialect_resources: {
    builtin: {
      blob1: "0x08000000010000000000000002000000000000000300000000000000",
      blob2: "0x08000000040000000000000005000000000000000600000000000000"
    }
  },
  external_resources: {
    external: {
      blob: "0x08000000010000000000000002000000000000000300000000000000",
      bool: true,
      string: "\"string\"" // with escape characters
    },
    other_stuff: {
      bool: true
    }
  }
#-}
