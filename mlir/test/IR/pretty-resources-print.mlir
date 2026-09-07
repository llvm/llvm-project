// Check printing with --mlir-elide-resource-strings-if-larger elides printing large resources

// RUN: mlir-opt %s --mlir-elide-resource-strings-if-larger=20| FileCheck %s

// RUN: mlir-opt %s --mlir-elide-resource-strings-if-larger=0| FileCheck %s --check-prefix=ZERO

// blob3's exact serialized size (quotes + "0x" + hex(alignment) + hex(data)) is 14 chars;
// these two RUN lines check the off-by-one boundary of the sizeHint-based elision.
// RUN: mlir-opt %s --mlir-elide-resource-strings-if-larger=14| FileCheck %s --check-prefix=BOUND14
// RUN: mlir-opt %s --mlir-elide-resource-strings-if-larger=13| FileCheck %s --check-prefix=BOUND13

// To ensure we print the resource keys, have reference to them
// CHECK: attr = dense_resource<blob1> : tensor<3xi64>
// ZERO: attr = dense_resource<blob1> : tensor<3xi64>
// BOUND14: attr = dense_resource<blob1> : tensor<3xi64>
// BOUND13: attr = dense_resource<blob1> : tensor<3xi64>
"test.blob1op"() {attr = dense_resource<blob1> : tensor<3xi64> } : () -> ()

// CHECK-NEXT: attr = dense_resource<blob2> : tensor<3xi64>
// ZERO-NEXT: attr = dense_resource<blob2> : tensor<3xi64>
// BOUND14-NEXT: attr = dense_resource<blob2> : tensor<3xi64>
// BOUND13-NEXT: attr = dense_resource<blob2> : tensor<3xi64>
"test.blob2op"() {attr = dense_resource<blob2> : tensor<3xi64> } : () -> ()

// CHECK-NEXT: attr = dense_resource<blob3> : tensor<1xi8>
// ZERO-NEXT: attr = dense_resource<blob3> : tensor<1xi8>
// BOUND14-NEXT: attr = dense_resource<blob3> : tensor<1xi8>
// BOUND13-NEXT: attr = dense_resource<blob3> : tensor<1xi8>
"test.blob3op"() {attr = dense_resource<blob3> : tensor<1xi8> } : () -> ()

// CHECK:      {-#
// CHECK-NEXT:   dialect_resources: {
// CHECK-NEXT:     builtin: {
// CHECK-NEXT:       blob3: "0x0800000001"
// CHECK-NEXT:     }
// CHECK-NEXT:   },
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

// At the exact boundary (limit == blob3's exact size) blob3 must still be printed,
// since the sizeHint fast-path only elides when size is strictly greater than the limit.
// BOUND14:      {-#
// BOUND14-NEXT:   dialect_resources: {
// BOUND14-NEXT:     builtin: {
// BOUND14-NEXT:       blob3: "0x0800000001"
// BOUND14-NEXT:     }
// BOUND14-NEXT:   },
// BOUND14-NEXT:   external_resources: {
// BOUND14-NEXT:     external: {
// BOUND14-NEXT:       "backslash\\tab\09": true,
// BOUND14-NEXT:       string: "\22string\22"
// BOUND14-NEXT:     },
// BOUND14-NEXT:     other_stuff: {
// BOUND14-NEXT:       bool: true
// BOUND14-NEXT:     }
// BOUND14-NEXT:   }
// BOUND14-NEXT: #-}

// One below the boundary, blob3 is elided and no dialect_resources dict is emitted.
// Note: the escaped `string` entry is also exactly 14 chars, so it is elided here too.
// BOUND13:      {-#
// BOUND13-NEXT:   external_resources: {
// BOUND13-NEXT:     external: {
// BOUND13-NEXT:       "backslash\\tab\09": true
// BOUND13-NEXT:     },
// BOUND13-NEXT:     other_stuff: {
// BOUND13-NEXT:       bool: true
// BOUND13-NEXT:     }
// BOUND13-NEXT:   }
// BOUND13-NEXT: #-}

{-#
  dialect_resources: {
    builtin: {
      blob1: "0x08000000010000000000000002000000000000000300000000000000",
      blob2: "0x08000000040000000000000005000000000000000600000000000000",
      blob3: "0x0800000001"
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
