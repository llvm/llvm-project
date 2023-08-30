// RUN: mlir-opt %s -test-recursive-types | FileCheck %s

// CHECK: !testrec = !test.test_rec<type_to_alias, test_rec<type_to_alias>>
// CHECK: ![[$NAME:.*]] = !test.test_rec_alias<name, !test.test_rec_alias<name>>
// CHECK: ![[$NAME2:.*]] = !test.test_rec_alias<name2, tuple<!test.test_rec_alias<name2>, i32>>

// CHECK-LABEL: @roundtrip
func.func @roundtrip() {
  // CHECK: !test.test_rec<a, test_rec<b, test_type>>
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec<a, test_rec<b, test_type>>
  // CHECK: !test.test_rec<c, test_rec<c>>
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec<c, test_rec<c>>
  // Make sure walkSubElementType, which is used to generate aliases, doesn't go
  // into inifinite recursion.
  // CHECK: !testrec
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec<type_to_alias, test_rec<type_to_alias>>

  // CHECK: () -> ![[$NAME]]
  // CHECK: () -> ![[$NAME]]
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec_alias<name, !test.test_rec_alias<name>>
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec_alias<name, !test.test_rec_alias<name>>

  // CHECK: () -> ![[$NAME2]]
  // CHECK: () -> ![[$NAME2]]
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec_alias<name2, tuple<!test.test_rec_alias<name2>, i32>>
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec_alias<name2, tuple<!test.test_rec_alias<name2>, i32>>
  return
}

// CHECK-LABEL: @create
func.func @create() {
  // CHECK: !test.test_rec<some_long_and_unique_name, test_rec<some_long_and_unique_name>>
  return
}
