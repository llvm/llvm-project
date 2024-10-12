// RUN: mlir-opt %s -test-recursive-types | FileCheck %s

// CHECK: !testrec = !test.test_rec<type_to_alias, test_rec<type_to_alias>>
// CHECK: ![[$NAME:.*]] = !test.test_rec_alias<name, !test.test_rec_alias<name>>
// CHECK: ![[$NAME5:.*]] = !test.test_rec_alias<name5, !test.test_rec_alias<name3>>
// CHECK: ![[$NAME7:.*]] = !test.test_rec_alias<name7, !test.test_rec_alias<name6>>
// CHECK: ![[$NAME2:.*]] = !test.test_rec_alias<name2, tuple<!test.test_rec_alias<name2>, i32>>
// CHECK: ![[$NAME4:.*]] = !test.test_rec_alias<name4, ![[$NAME5]]>
// CHECK: ![[$NAME6:.*]] = !test.test_rec_alias<name6, ![[$NAME7]]>
// CHECK: ![[$NAME3:.*]] = !test.test_rec_alias<name3, ![[$NAME4]]>

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

  // Mutual recursion with types fully spelled out.
  // CHECK: () -> ![[$NAME3]]
  // CHECK: () -> ![[$NAME4]]
  // CHECK: () -> ![[$NAME5]]
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec_alias<name3, !test.test_rec_alias<name4, !test.test_rec_alias<name5, !test.test_rec_alias<name3>>>>
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec_alias<name4, !test.test_rec_alias<name5, !test.test_rec_alias<name3, !test.test_rec_alias<name4>>>>
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec_alias<name5, !test.test_rec_alias<name3, !test.test_rec_alias<name4, !test.test_rec_alias<name5>>>>

  // Mutual recursion with incomplete types.
  // CHECK: () -> ![[$NAME6]]
  // CHECK: () -> ![[$NAME7]]
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec_alias<name6, !test.test_rec_alias<name7>>
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec_alias<name7, !test.test_rec_alias<name6>>

  return
}

// CHECK-LABEL: @create
func.func @create() {
  // CHECK: !test.test_rec<some_long_and_unique_name, test_rec<some_long_and_unique_name>>
  return
}
