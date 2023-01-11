// RUN: mlir-opt -allow-unregistered-dialect -test-strict-pattern-driver %s | FileCheck %s

// CHECK-LABEL: func @test_erase
//       CHECK:   test.arg0
//       CHECK:   test.arg1
//   CHECK-NOT:   test.erase_op
func.func @test_erase() {
  %0 = "test.arg0"() : () -> (i32)
  %1 = "test.arg1"() : () -> (i32)
  %erase = "test.erase_op"(%0, %1) : (i32, i32) -> (i32)
  return
}

// CHECK-LABEL: func @test_insert_same_op
//       CHECK:   "test.insert_same_op"() {skip = true}
//       CHECK:   "test.insert_same_op"() {skip = true}
func.func @test_insert_same_op() {
  %0 = "test.insert_same_op"() : () -> (i32)
  return
}

// CHECK-LABEL: func @test_replace_with_new_op
//       CHECK:   %[[n:.*]] = "test.new_op"
//       CHECK:   "test.dummy_user"(%[[n]])
//       CHECK:   "test.dummy_user"(%[[n]])
func.func @test_replace_with_new_op() {
  %0 = "test.replace_with_new_op"() : () -> (i32)
  %1 = "test.dummy_user"(%0) : (i32) -> (i32)
  %2 = "test.dummy_user"(%0) : (i32) -> (i32)
  return
}

// CHECK-LABEL: func @test_replace_with_erase_op
//   CHECK-NOT:   test.replace_with_new_op
//   CHECK-NOT:   test.erase_op
func.func @test_replace_with_erase_op() {
  "test.replace_with_new_op"() {create_erase_op} : () -> ()
  return
}
