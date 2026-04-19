// UNSUPPORTED: system-windows
// RUN: mlir-reduce %s -split-input-file -reduction-tree='traversal-mode=0 test=%S/failure-test.sh' | FileCheck %s
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

// -----

// This input should be reduced by the pass pipeline so that only
// the @simple5 function remains as this is the shortest function
// containing the interesting behavior.

// CHECK-NOT: func @simple1() {
func.func @simple1() {
  return
}

// CHECK-NOT: func @simple2() {
func.func @simple2() {
  return
}

// CHECK-LABEL: func @simple3() {
func.func @simple3() {
  "test.op_crash" () : () -> ()
  return
}

// CHECK-NOT: func @simple4() {
func.func @simple4(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  cf.br ^bb3(%arg1 : memref<2xf32>)
^bb2:
  %0 = memref.alloc() : memref<2xf32>
  cf.br ^bb3(%0 : memref<2xf32>)
^bb3(%1: memref<2xf32>):
  "test.op_crash"(%1, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
  return
}

// CHECK-NOT: func @simple5() {
func.func @simple5() {
  return
}

// -----

// This test case will be reduced by ReplaceOperandsPattern.
// crash's i32 operands should be replaced by %arg0.
// Extra dead operations should be removed.

// CHECK-LABEL: func.func @replace(%arg0: i32) -> i32 {
func.func @replace(%arg0: i32) -> i32 {
  // CHECK-NOT: {{.*}} = arith.constant 1 : i32
  %0 = arith.constant 1 : i32
  // CHECK: {{.*}} = arith.constant 2.000000e+00 : f32
  %1 = arith.constant 2.0 : f32
  // CHECK-NOT: {{.*}} = arith.constant 3.0 : i32
  %2 = arith.constant 3 : i32
  // CHECK: {{.*}} = "test.op_crash"(%arg0, {{.*}}, %arg0) : (i32, f32, i32) -> i32
  %crash = "test.op_crash"(%0, %1, %2) : (i32, f32, i32) -> i32
  return %crash : i32
}
