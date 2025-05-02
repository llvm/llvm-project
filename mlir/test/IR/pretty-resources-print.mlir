// Check printing with --mlir-elide-resource-strings-if-larger elides printing large resources

// RUN: mlir-opt %s --mlir-elide-resource-strings-if-larger=20| FileCheck %s

// RUN: mlir-opt %s --mlir-elide-resource-strings-if-larger=0| FileCheck %s --check-prefix=ZERO


// To ensure we print the resource keys, have reference to them
// CHECK: attr = dense_resource<blob1> : tensor<3xi64>
// ZERO: attr = dense_resource<blob1> : tensor<3xi64>
"test.blob1op"() {attr = dense_resource<blob1> : tensor<3xi64> } : () -> ()

// CHECK-NEXT: attr = dense_resource<blob2> : tensor<3xi64>
// ZERO-NEXT: attr = dense_resource<blob2> : tensor<3xi64>
"test.blob2op"() {attr = dense_resource<blob2> : tensor<3xi64> } : () -> ()

// CHECK:      {-#
// CHECK-NEXT:   external_resources: {
// CHECK-NEXT:     external: {
// CHECK-NEXT:       "backslash\\tab\09": true,
// CHECK-NEXT:       string: "\22string\22"
// CHECK-NEXT:     },
// CHECK-NEXT:     other_stuff: {
// CHECK-NEXT:       bool: true
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: #-}

// Make sure no external_resources are printed when --mlir-elide-resource-strings-if-larger=0
// ZERO:      {-#
// ZERO-EMPTY:
// ZERO-NEXT: #-}

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
      "backslash\\tab\09": true, // quoted key with escape characters
      string: "\"string\"" // string with escape characters
    },
    other_stuff: {
      bool: true
    }
  }
#-}
