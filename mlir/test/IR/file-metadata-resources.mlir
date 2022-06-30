// RUN: mlir-opt %s -split-input-file | FileCheck %s

// Check that we only preserve the blob that got referenced.
// CHECK:      test: {
// CHECK-NEXT:   blob1: "0x08000000010000000000000002000000000000000300000000000000"
// CHECK-NEXT: }

module attributes { test.blob_ref = #test.e1di64_elements<blob1> } {}

{-#
  dialect_resources: {
    test: {
      blob1: "0x08000000010000000000000002000000000000000300000000000000",
      blob2: "0x08000000040000000000000005000000000000000600000000000000"
    }
  }
#-}
