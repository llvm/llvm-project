// RUN: mlir-opt %s -test-greedy-patterns="dump-notifications top-down=true fold=false" -split-input-file | FileCheck %s

// Hoisting across regions must report the old position. Unifying the second
// constant must also report replacement, user modification, and erasure.
// CHECK-LABEL: notifyOperationInserted: test.constant, previous = test.constant
// CHECK-NEXT: notifyOperationReplaced: test.constant
// CHECK-NEXT: notifyOperationModified: test.use
// CHECK-NEXT: notifyOperationErased: test.constant
// CHECK: func.func @hoist_and_unique()
// CHECK-NEXT: %[[C:.*]] = "test.constant"
// CHECK: "test.use"(%[[C]], %[[C]])
func.func @hoist_and_unique() {
  %outer = "test.one_region_with_recursive_memory_effects"() ({
    %a = "test.constant"() {value = 1 : i32} : () -> i32
    %b = "test.constant"() {value = 1 : i32} : () -> i32
    "test.use"(%a, %b) : (i32, i32) -> ()
    "test.return"() : () -> ()
  }) : () -> i32
  return
}

// -----

// A constant behind an ordinary operation must report its move within the
// same block, even on its first insertion into the folder.
// CHECK-LABEL: notifyOperationInserted: test.constant, previous = func.return
// CHECK: func.func @hoist_in_block()
// CHECK-NEXT: %[[C:.*]] = "test.constant"
// CHECK-NEXT: {{.*}} = "test.side_effect_op"
// CHECK-NEXT: return %[[C]]
func.func @hoist_in_block() -> i32 {
  %unused = "test.side_effect_op"() {effects = [{effect = "write"}]} : () -> i32
  %c = "test.constant"() {value = 1 : i32} : () -> i32
  return %c : i32
}
