// UNSUPPORTED: system-windows
// RUN: mlir-reduce %s -test %S/failure-test.sh -pass-test function-reducer | FileCheck %s
// This input should be reduced by the pass pipeline so that only
// the @simple5 function remains as this is the shortest function
// containing the interesting behavior.

// CHECK-NOT: func @simple1() {
func @simple1() {
  return
}

// CHECK-NOT: func @simple2() {
func @simple2() {
  return
}

// CHECK-LABEL: func @simple3() {
func @simple3() {
  "test.crashOp" () : () -> ()
  return
}

// CHECK-NOT: func @simple4() {
func @simple4(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  cond_br %arg0, ^bb1, ^bb2
^bb1:
  br ^bb3(%arg1 : memref<2xf32>)
^bb2:
  %0 = memref.alloc() : memref<2xf32>
  br ^bb3(%0 : memref<2xf32>)
^bb3(%1: memref<2xf32>):
  "test.crashOp"(%1, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
  return
}

// CHECK-NOT: func @simple5() {
func @simple5() {
  return
}
