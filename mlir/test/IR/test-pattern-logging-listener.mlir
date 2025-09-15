// REQUIRES: asserts
// RUN: mlir-opt %s --test-walk-pattern-rewrite-driver \
// RUN:   --debug-only=pattern-logging-listener 2>&1 | FileCheck %s

// Check that when replacing an op with a new op, we get appropriate
// pattern-logging lines. The use of check same is to avoid the complexity of
// matching the anonymous namespace prefix, which can be one of {anonymous} vs
// {anonymous_namespace} vs `anonymous_namespace` (and maybe others?) on the
// various platforms.

// Changing the reference output here need to also plan on updating the post-processing
// script here:  https://github.com/llvm/mlir-www/blob/main/.github%2Fworkflows%2Fmain.yml#L71-L72

// CHECK: [pattern-logging-listener PatternLoggingListener.cpp:10 1]
// CHECK-SAME: ::ReplaceWithNewOp | notifyOperationInserted | test.new_op
// CHECK: [pattern-logging-listener PatternLoggingListener.cpp:31 1]
// CHECK-SAME: ::ReplaceWithNewOp | notifyOperationReplaced (with values) | test.replace_with_new_op
// CHECK: [pattern-logging-listener PatternLoggingListener.cpp:17 1]
// CHECK-SAME: ::ReplaceWithNewOp | notifyOperationModified | arith.addi
// CHECK: [pattern-logging-listener PatternLoggingListener.cpp:17 1]
// CHECK-SAME: ::ReplaceWithNewOp | notifyOperationModified | arith.addi
// CHECK: [pattern-logging-listener PatternLoggingListener.cpp:38 1]
// CHECK-SAME: ::ReplaceWithNewOp | notifyOperationErased | test.replace_with_new_op
func.func @replace_with_new_op() -> i32 {
  %a = "test.replace_with_new_op"() : () -> (i32)
  %res = arith.addi %a, %a : i32
  return %res : i32
}
