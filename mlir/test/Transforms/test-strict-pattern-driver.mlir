// RUN: mlir-opt -allow-unregistered-dialect -test-strict-pattern-driver %s | FileCheck %s

// CHECK-LABEL: @test_erase
func.func @test_erase() {
  %0 = "test.arg0"() : () -> (i32)
  %1 = "test.arg1"() : () -> (i32)
  %erase = "test.erase_op"(%0, %1) : (i32, i32) -> (i32)
  return
}

// CHECK-LABEL: @test_insert_same_op
func.func @test_insert_same_op() {
  %0 = "test.insert_same_op"() : () -> (i32)
  return
}

// CHECK-LABEL: @test_replace_with_same_op
func.func @test_replace_with_same_op() {
  %0 = "test.replace_with_same_op"() : () -> (i32)
  %1 = "test.dummy_user"(%0) : (i32) -> (i32)
  %2 = "test.dummy_user"(%0) : (i32) -> (i32)
  return
}
