// RUN: mlir-opt -convert-llvm-to-control-flow -split-input-file %s | FileCheck %s

// CHECK-LABEL: @branch
// CHECK-SAME: %[[VAL:[[:alnum:]]+]]: i32
llvm.func @branch(%val: i32) -> i32 {
  // CHECK: cf.br ^[[BB1:.*]](%[[VAL]] : i32)
  llvm.br ^bb1(%val : i32)
// CHECK: ^[[BB1]](%[[BB_ARG:.*]]: i32):
^bb1(%x: i32):
  // CHECK: llvm.return %[[BB_ARG]] : i32
  llvm.return %x : i32
}

// -----

// CHECK-LABEL: @cond_branch
// CHECK-SAME: %[[COND:[[:alnum:]]+]]: i1
// CHECK-SAME: %[[VAL1:[[:alnum:]]+]]: i32
// CHECK-SAME: %[[VAL2:[[:alnum:]]+]]: i32
llvm.func @cond_branch(%cond: i1, %val1: i32, %val2: i32) -> i32 {
  // CHECK: cf.cond_br %[[COND]], ^[[BB1:.*]](%[[VAL1]] : i32), ^[[BB2:.*]](%[[VAL2]] : i32)
  llvm.cond_br %cond, ^bb1(%val1 : i32), ^bb2(%val2 : i32)
// CHECK: ^[[BB1]](%[[BB1_ARG:.*]]: i32):
^bb1(%x: i32):
  // CHECK: llvm.return %[[BB1_ARG]] : i32
  llvm.return %x : i32
// CHECK: ^[[BB2]](%[[BB2_ARG:.*]]: i32):
^bb2(%y: i32):
  // CHECK: llvm.return %[[BB2_ARG]] : i32
  llvm.return %y : i32
}
