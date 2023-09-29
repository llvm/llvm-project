// RUN: mlir-opt %s -test-recursive-types | FileCheck %s

// CHECK-DAG: !testrec = !test.test_rec<type_to_alias, test_rec<type_to_alias>>
// CHECK-DAG: ![[$NAME:.*]] = !test.test_rec_alias<name, ![[$NAME]]>
// CHECK-DAG: ![[$NAME2:.*]] = !test.test_rec_alias<name2, tuple<![[$NAME2]], i32>>
// CHECK-DAG: #[[$ATTR:.*]] = #test.test_rec_alias<attr, #[[$ATTR]]>
// CHECK-DAG: #[[$ATTR2:.*]] = #test.test_rec_alias<attr2, [#[[$ATTR2]], 5]>


!name = !test.test_rec_alias<name, !name>
!name2 = !test.test_rec_alias<name2, tuple<!name2, i32>>

#attr = #test.test_rec_alias<attr, #attr>
#array = [#attr2, 5]
#attr2 = #test.test_rec_alias<attr2, #array>

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

  // Check that we can use these aliases, not just print them.
  // CHECK: value = #[[$ATTR]]
  // CHECK-SAME: () -> ![[$NAME]]
  // CHECK-NEXT: value = #[[$ATTR2]]
  // CHECK-SAME: () -> ![[$NAME2]]
  "test.dummy_op_for_roundtrip"() { value = #attr } : () -> !name
  "test.dummy_op_for_roundtrip"() { value = #attr2 } : () -> !name2
  return
}

// CHECK-LABEL: @create
func.func @create() {
  // CHECK: !test.test_rec<some_long_and_unique_name, test_rec<some_long_and_unique_name>>
  return
}
