// RUN: mlir-opt \
// RUN:     -test-strict-pattern-driver="strictness=ExistingAndNewOps" \
// RUN:     --split-input-file %s | FileCheck %s --check-prefix=CHECK-EN

// RUN: mlir-opt \
// RUN:     -test-strict-pattern-driver="strictness=ExistingOps" \
// RUN:     --split-input-file %s | FileCheck %s --check-prefix=CHECK-EX

// CHECK-EN-LABEL: func @test_erase
//  CHECK-EN-SAME:     pattern_driver_all_erased = true, pattern_driver_changed = true}
//       CHECK-EN:   test.arg0
//       CHECK-EN:   test.arg1
//   CHECK-EN-NOT:   test.erase_op
func.func @test_erase() {
  %0 = "test.arg0"() : () -> (i32)
  %1 = "test.arg1"() : () -> (i32)
  %erase = "test.erase_op"(%0, %1) : (i32, i32) -> (i32)
  return
}

// -----

// CHECK-EN-LABEL: func @test_insert_same_op
//  CHECK-EN-SAME:     {pattern_driver_all_erased = false, pattern_driver_changed = true}
//       CHECK-EN:   "test.insert_same_op"() {skip = true}
//       CHECK-EN:   "test.insert_same_op"() {skip = true}
func.func @test_insert_same_op() {
  %0 = "test.insert_same_op"() : () -> (i32)
  return
}

// -----

// CHECK-EN-LABEL: func @test_replace_with_new_op
//  CHECK-EN-SAME:     {pattern_driver_all_erased = true, pattern_driver_changed = true}
//       CHECK-EN:   %[[n:.*]] = "test.new_op"
//       CHECK-EN:   "test.dummy_user"(%[[n]])
//       CHECK-EN:   "test.dummy_user"(%[[n]])
func.func @test_replace_with_new_op() {
  %0 = "test.replace_with_new_op"() : () -> (i32)
  %1 = "test.dummy_user"(%0) : (i32) -> (i32)
  %2 = "test.dummy_user"(%0) : (i32) -> (i32)
  return
}

// -----

// CHECK-EN-LABEL: func @test_replace_with_erase_op
//  CHECK-EN-SAME:     {pattern_driver_all_erased = true, pattern_driver_changed = true}
//   CHECK-EN-NOT:   test.replace_with_new_op
//   CHECK-EN-NOT:   test.erase_op

// CHECK-EX-LABEL: func @test_replace_with_erase_op
//  CHECK-EX-SAME:     {pattern_driver_all_erased = true, pattern_driver_changed = true}
//   CHECK-EX-NOT:   test.replace_with_new_op
//       CHECK-EX:   test.erase_op
func.func @test_replace_with_erase_op() {
  "test.replace_with_new_op"() {create_erase_op} : () -> ()
  return
}
