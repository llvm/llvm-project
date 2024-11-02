// RUN: mlir-opt %s | mlir-opt -test-patterns | FileCheck %s

// CHECK-LABEL: @test_enum_attr_roundtrip
func.func @test_enum_attr_roundtrip() -> () {
  // CHECK: value = #test<"enum first">
  "test.op"() {value = #test<"enum first">} : () -> ()
  // CHECK: value = #test<"enum second">
  "test.op"() {value = #test<"enum second">} : () -> ()
  // CHECK: value = #test<"enum third">
  "test.op"() {value = #test<"enum third">} : () -> ()
  return
}

// CHECK-LABEL: @test_op_with_enum
func.func @test_op_with_enum() -> () {
  // CHECK: test.op_with_enum third
  test.op_with_enum third
  return
}

// CHECK-LABEL: @test_match_op_with_enum
func.func @test_match_op_with_enum() -> () {
  // CHECK: test.op_with_enum third tag 0 : i32
  test.op_with_enum third tag 0 : i32
  // CHECK: test.op_with_enum second tag 1 : i32
  test.op_with_enum first tag 0 : i32
  return
}
