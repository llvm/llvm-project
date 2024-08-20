// RUN: mlir-opt --canonicalize='enable-patterns=AnyPattern region-simplify=aggressive' %s | FileCheck %s

// Perform merge checks without performing canonicalization prior to simplification

// This test should not merge ^bb2 and ^bb5, despite the fact that they are
// identical because %4 is used outside of ^bb2.
func.func @nested_loop(%arg0 : i32, %arg1 : i32, %arg2 : i32, %arg3 : i32) -> i32 {
  cf.br ^bb1(%arg0, %arg0 : i32, i32)
^bb1(%0: i32, %1: i32):
  %2 = "test.foo"(%1, %arg2) : (i32, i32) -> i1
  cf.cond_br %2, ^bb2(%0, %1 : i32, i32), ^bb7(%0 : i32)
^bb2(%3: i32, %4: i32):
  %5 = "test.foo"(%4, %arg1) : (i32, i32) -> i32
  cf.br ^bb3(%3, %5 : i32, i32)
^bb3(%6: i32, %7: i32):
  %8 = "test.foo"(%7, %arg3) : (i32, i32) -> i1
  cf.cond_br %8, ^bb4(%6, %7 : i32, i32), ^bb6(%6, %4 : i32, i32)
^bb4(%9: i32, %10: i32):
  %11 = "test.foo"(%9, %arg1) : (i32, i32) -> i32
  cf.br ^bb5(%11, %10 : i32, i32)
^bb5(%12: i32, %13: i32):
  %14 = "test.foo"(%13, %arg1) : (i32, i32) -> i32
  cf.br ^bb3(%12, %14 : i32, i32)
^bb6(%15: i32, %16: i32):
  %17 = arith.addi %16, %arg1 : i32
  cf.br ^bb1(%15, %17 : i32, i32)
^bb7(%18: i32):
  return %18 : i32
}

// CHECK-LABEL:   func.func @nested_loop
// CHECK:         ^bb1(%{{.*}}: i32, %[[BB1_ARG1:.*]]: i32):
// CHECK:           arith.addi %[[BB1_ARG1]]
