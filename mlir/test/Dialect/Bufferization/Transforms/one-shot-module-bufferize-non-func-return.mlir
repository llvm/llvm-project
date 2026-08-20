// RUN: mlir-opt %s -one-shot-bufferize='bufferize-function-boundaries=1 copy-before-write=1' | FileCheck %s

// This function does not have a `func.return` terminator (it uses
// `spirv.Return` instead, which is a valid terminator for `func.func` since
// the verifier only requires the region to end in some terminator, not
// specifically a `func.return`). One-Shot Module Bufferize must not crash on
// such IR; it has no tensors to bufferize, so it should simply leave the
// function unchanged.

// CHECK-LABEL: func.func private @m(
// CHECK-SAME:      %{{.*}}: memref<4xi32>)
// CHECK-NEXT:    spirv.Return
module {
  func.func private @m(%arg0: memref<4xi32>) {
    spirv.Return
  }
}
