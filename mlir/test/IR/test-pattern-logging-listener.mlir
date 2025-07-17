// RUN: mlir-opt %s --test-walk-pattern-rewrite-driver \
// RUN:   --allow-unregistered-dialect --debug-only=pattern-logging-listener 2>&1 | FileCheck %s

// Check that when replacing an op with a new op, we get appropriate
// pattern-logging lines. The regex is because theanonymousis
// printed differently on different platforms.

// CHECK: [pattern-logging-listener] {anonymous}::ReplaceWithNewOp | notifyOperationInserted | test.new_op
// CHECK: [pattern-logging-listener] {anonymous}::ReplaceWithNewOp | notifyOperationReplaced (with values) | test.replace_with_new_op
// CHECK: [pattern-logging-listener] {anonymous}::ReplaceWithNewOp | notifyOperationModified | arith.addi
// CHECK: [pattern-logging-listener] {anonymous}::ReplaceWithNewOp | notifyOperationModified | arith.addi
// CHECK: [pattern-logging-listener] {anonymous}::ReplaceWithNewOp | notifyOperationErased | test.replace_with_new_op
func.func @replace_with_new_op() -> i32 {
  %a = "test.replace_with_new_op"() : () -> (i32)
  %res = arith.addi %a, %a : i32
  return %res : i32
}
