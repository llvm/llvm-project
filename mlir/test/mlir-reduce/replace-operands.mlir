// UNSUPPORTED: system-windows
// RUN: mlir-reduce %s -reduction-tree='traversal-mode=0 test=%S/failure-test.sh replace-operands=true' | FileCheck %s

// CHECK-LABEL: func.func @main
func.func @main() {
  // CHECK-NEXT: %[[RESULT:.*]] = arith.constant 3 : i32
  // CHECK-NEXT: {{.*}} = "test.op_crash"(%[[RESULT]], %[[RESULT]]) : (i32, i32) -> i32
  // CHECK-NEXT return

  %c1 = arith.constant 3 : i32
  %c2 = arith.constant 2 : i32
  %2 = "test.op_crash" (%c1, %c2) : (i32, i32) -> i32
  return
}
