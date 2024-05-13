// UNSUPPORTED: system-windows
// RUN: mlir-reduce %s -reduction-tree='traversal-mode=0 test=%S/failure-test.sh' | FileCheck %s
// "test.op_crash_long" should be replaced with a shorter form "test.op_crash_short".

// CHECK-NOT: func @simple1() {
func.func @simple1() {
  return
}

// CHECK-LABEL: func @simple2(%arg0: i32, %arg1: i32, %arg2: i32) {
func.func @simple2(%arg0: i32, %arg1: i32, %arg2: i32) {
  // CHECK-LABEL: %0 = "test.op_crash_short"() : () -> i32
  %0 = "test.op_crash_long" (%arg0, %arg1, %arg2) : (i32, i32, i32) -> i32
  return
}

// CHECK-NOT: func @simple5() {
func.func @simple5() {
  return
}
