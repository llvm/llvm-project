// RUN: not mlir-opt %s -split-input-file --test-bytecode-roundtrip="test-kind=2" 2>&1 | FileCheck %s

// Tests that proper errors are emitted (rather than crashes) when the type
// callback replaces types with ones that are incompatible with built-in types
// and attributes (test-kind=2 replaces i32 with !test.i32).

// CHECK: expected integer or index type for IntegerAttr, but got: '!test.i32'
// CHECK: failed to read bytecode
// IntegerAttr whose type is replaced by one that is neither IntegerType nor
// IndexType — previously crashed with an APInt assertion.
module {
  func.func @integer_attr_unsupported_type() {
    %c = arith.constant 1 : i32
    return
  }
}

// -----

// CHECK: failed to verify 'elementType': VectorElementTypeInterface instance
// CHECK: failed to read bytecode
// Fixed-size VectorType whose element type is replaced by one that does not
// implement VectorElementTypeInterface — previously crashed in VectorType::get.
module {
  func.func @vector_unsupported_elem_type() {
    %cst = arith.constant dense<42> : vector<3xi32>
    return
  }
}

// -----

// CHECK: failed to verify 'elementType': VectorElementTypeInterface instance
// CHECK: failed to read bytecode
// Scalable VectorType whose element type is replaced by one that does not
// implement VectorElementTypeInterface — exercises the VectorTypeWithScalableDims
// bytecode path.
module {
  func.func @scalable_vector_unsupported_elem_type(%v : vector<[3]xi32>) {
    return
  }
}

// -----

// CHECK: DenseTypedElementsAttr element type must implement DenseElementTypeInterface, but got: '!test.i32'
// CHECK: failed to read bytecode
// DenseTypedElementsAttr whose element type is replaced by one that does not
// implement DenseElementTypeInterface — previously crashed with an assertion.
module {
  func.func @dense_elem_unsupported_type() -> tensor<10xi32> {
    %0 = arith.constant dense<42> : tensor<10xi32>
    return %0 : tensor<10xi32>
  }
}
