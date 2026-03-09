// RUN: not mlir-opt %s --test-bytecode-roundtrip="test-kind=2" 2>&1 | FileCheck %s

// Regression test: test-kind=2 replaces i32 with !test.i32 (a type that does
// not implement DenseElementTypeInterface). This should produce a proper error
// instead of an assertion failure when deserializing DenseIntOrFPElementsAttr.

// CHECK: DenseIntOrFPElementsAttr element type must implement DenseElementTypeInterface, but got: '!test.i32'
// CHECK: failed to read bytecode

module {
  func.func @test() -> tensor<10xi32> {
    %0 = arith.constant dense<42> : tensor<10xi32>
    return %0 : tensor<10xi32>
  }
}
