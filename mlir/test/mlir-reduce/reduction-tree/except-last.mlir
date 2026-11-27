// UNSUPPORTED: system-windows
// RUN: mlir-reduce %s -reduction-tree='traversal-mode=0 test=%S/../script/except-last.sh' | FileCheck %s
// We are testing the ability of keeping trivially-dead yet interesting code

func.func @except_last() {
  %0 = arith.constant 1 : i32
  %1 = arith.constant 2 : i32
  %2 = arith.constant 3 : i32
  %3 = arith.constant 4 : i32
  return
}

// CHECK-LABEL: func @except_last
//  CHECK-NEXT:   arith.constant 1 : i32
//  CHECK-NEXT:   arith.constant 2 : i32
//  CHECK-NEXT:   arith.constant 3 : i32
//  CHECK-NEXT: return
