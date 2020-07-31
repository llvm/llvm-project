// RUN: mlir-opt %s -test-recursive-types | FileCheck %s

// CHECK-LABEL: @roundtrip
func @roundtrip() {
  // CHECK: !test.test_rec<a, test_rec<b, test_type>>
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec<a, test_rec<b, test_type>>
  // CHECK: !test.test_rec<c, test_rec<c>>
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec<c, test_rec<c>>
  return
}

// CHECK-LABEL: @create
func @create() {
  // CHECK: !test.test_rec<some_long_and_unique_name, test_rec<some_long_and_unique_name>>
  return
}
