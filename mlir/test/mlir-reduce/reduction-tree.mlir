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

// CHECK-LABEL: func @br_reduction
//  CHECK-SAME:  %[[ARG0:.*]]: i1,
//  CHECK-SAME:  %[[ARG1:.*]]: memref<2xf32>,
//  CHECK-SAME:  %[[ARG2:.*]]: memref<2xf32>) {
func.func @br_reduction(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
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
// CHECK-NEXT: "test.op_crash"(%[[ARG1]], %[[ARG2]])

// -----

// CHECK-LABEL: func @br_reduction_loop
//  CHECK-SAME:   %[[ARG0:.*]]: i1,
//  CHECK-SAME:   %[[ARG1:.*]]: memref<2xf32>,
//  CHECK-SAME:   %[[ARG2:.*]]: memref<2xf32>) {
func.func @br_reduction_loop(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  cf.br ^bb3(%arg1 : memref<2xf32>)
^bb2:
  %0 = memref.alloc() : memref<2xf32>
  cf.br ^bb3(%0 : memref<2xf32>)
^bb3(%1: memref<2xf32>):
  "test.op_crash"(%1, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
  cf.cond_br %arg0, ^bb3(%1: memref<2xf32>), ^bb4
^bb4:
  return
}
// CHECK-NEXT:   cf.br ^bb1(%[[ARG1]] : memref<2xf32>)
// CHECK-NEXT: ^bb1(%[[VAL_0:.*]]: memref<2xf32>):
// CHECK-NEXT:   "test.op_crash"(%[[VAL_0]], %[[ARG2]])
// CHECK-NEXT:   cf.br ^bb1(%[[VAL_0]] : memref<2xf32>)

// -----

// CHECK-LABEL: func @switch_reduction
//  CHECK-SAME:   %[[ARG0:.*]]: i32,
//  CHECK-SAME:   %[[ARG1:.*]]: memref<2xf32>,
//  CHECK-SAME:   %[[ARG2:.*]]: memref<2xf32>) {
func.func @switch_reduction(%arg0: i32, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  cf.switch %arg0 : i32, [
    default: ^bb3(%arg1 : memref<2xf32>),
    0: ^bb1,
    1: ^bb2
  ]
^bb1:
  cf.br ^bb3(%arg1 : memref<2xf32>)
^bb2:
  %0 = memref.alloc() : memref<2xf32>
  cf.br ^bb3(%0 : memref<2xf32>)
^bb3(%1: memref<2xf32>):
  "test.op_crash"(%1, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
  return
}
// CHECK-NEXT:  "test.op_crash"(%[[ARG1]], %[[ARG2]])