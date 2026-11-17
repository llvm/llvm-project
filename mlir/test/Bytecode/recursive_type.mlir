// RUN: mlir-opt -emit-bytecode %s | mlir-opt | FileCheck %s

// Recursive types use their custom encoding rather than the textual fallback.
// RUN: mlir-opt -emit-bytecode %s | llvm-strings | FileCheck %s --check-prefix=BYTECODE --implicit-check-not=test_rec

// BYTECODE: dummy_op_for_roundtrip

// CHECK-LABEL: @recursive_types
func.func @recursive_types() {
  // CHECK: !test.test_rec<self, test_rec<self>>
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec<self, test_rec<self>>
  // CHECK: !test.test_rec<first, test_rec<second, test_rec<first>>>
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec<first, test_rec<second, test_rec<first>>>
  // CHECK: !test.test_rec<second, test_rec<first, test_rec<second>>>
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec<second, test_rec<first, test_rec<second>>>
  // A cycle deeper than the reader resolves eagerly, so it defers and retries
  // types it already bound.
  // CHECK: !test.test_rec<c0, test_rec<c1, test_rec<c2, test_rec<c3, test_rec<c4, test_rec<c5, test_rec<c6, test_rec<c7, test_rec<c8, test_rec<c9, test_rec<c0>>>>>>>>>>>
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec<c0, test_rec<c1, test_rec<c2, test_rec<c3, test_rec<c4, test_rec<c5, test_rec<c6, test_rec<c7, test_rec<c8, test_rec<c9, test_rec<c0>>>>>>>>>>>
  return
}
