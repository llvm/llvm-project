// REQUIRES: asserts
// RUN: mlir-opt %s --test-walk-pattern-rewrite-driver --debug-only=pattern-application:1 -o /dev/null 2>&1 | FileCheck %s --check-prefix=LEVEL1 --implicit-check-not="IR Dump After Pattern Application"
// RUN: mlir-opt %s --test-walk-pattern-rewrite-driver --debug-only=pattern-application:2 -o /dev/null 2>&1 | FileCheck %s --check-prefix=LEVEL2

// The pattern erases its root. The dump must still show the enclosing function.
// LEVEL1: matchAndRewrite successful
// LEVEL2: IR Dump After Pattern Application
// LEVEL2: func.func @replace_with_new_op()
// LEVEL2: "test.new_op"
// LEVEL2-NOT: "test.replace_with_new_op"
// LEVEL2: return
func.func @replace_with_new_op() -> i32 {
  %a = "test.replace_with_new_op"() : () -> i32
  %res = arith.addi %a, %a : i32
  return %res : i32
}
