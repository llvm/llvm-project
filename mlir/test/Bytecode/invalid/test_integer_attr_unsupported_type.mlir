// RUN: not mlir-opt %s --test-bytecode-roundtrip="test-kind=2" 2>&1 | FileCheck %s

// CHECK: expected integer or index type for IntegerAttr, but got: '!test.i32'
// CHECK: failed to read bytecode

// This test verifies that a proper error is emitted (rather than crashing with
// an APInt assertion) when the type callback replaces an integer type with one
// that does not implement IntegerType or IndexType.
module {
  func.func @combined_operations() -> () {
    %cst = arith.constant dense<[[true, true, true], [true, false, true]]> : tensor<2x3xi1>
    %reduction_output = tosa.reduce_all %cst {axis = 1 : i32} : (tensor<2x3xi1>) -> tensor<2x1xi1>
    return
  }
}
