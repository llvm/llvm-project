// RUN: not mlir-opt %s --test-bytecode-roundtrip="test-kind=2" 2>&1 | FileCheck %s

// CHECK: failed to verify 'elementType': VectorElementTypeInterface instance
// CHECK: failed to read bytecode

// This test verifies that a proper error is emitted (rather than crashing with
// an assertion) when the type callback replaces a vector element type with one
// that does not implement VectorElementTypeInterface.
module {
  func.func @main() -> () {
    %cst = arith.constant dense<42> : vector<3xi32>
    return
  }
}
