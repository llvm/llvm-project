// UNSUPPORTED: system-windows
// RUN: mlir-reduce %s --no-implicit-module -opt-reduction-pass='opt-pass=cse test=%S/../failure-test.sh' | FileCheck %s

// CHECK-LABEL: func @cse_on_func
//  CHECK-SAME:   %[[ARG0:.*]]: i32,
//  CHECK-SAME:   %[[ARG1:.*]]: i32) {
func.func @cse_on_func(%arg0: i32, %arg1: i32) {
  %0 = arith.addi %arg0, %arg1 : i32
  %1 = arith.addi %arg0, %arg1 : i32 
  %2 = "test.op_crash_long" (%0, %0, %1) : (i32, i32, i32) -> i32
  return
}

// CHECK: %[[ADD:.*]] = arith.addi %[[ARG0]], %[[ARG1]]
// CHECK: %[[CRASH:.*]] = "test.op_crash_long"(%[[ADD]], %[[ADD]], %[[ADD]])
// CHECK: return
