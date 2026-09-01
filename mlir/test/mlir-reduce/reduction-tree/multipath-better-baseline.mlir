// UNSUPPORTED: system-windows
// RUN: mlir-reduce %s -reduction-tree='traversal-mode=0 test=%S/../script/multipath-better.sh' | FileCheck %s
// We are testing the ability of keeping trivially-dead yet interesting code

func.func @multipath_better() {
  %0 = arith.constant 1 : i32
  %1 = arith.constant 2 : i32
  %2 = arith.constant 3 : i32
  %3 = arith.constant 4 : i32
  %4 = arith.constant 5 : i32
  %5 = arith.constant 6 : i32
  return
}

// CHECK-LABEL: func @multipath_better
//  CHECK-NEXT:   arith.constant 1 : i32
//  CHECK-NEXT:   arith.constant 2 : i32
//  CHECK-NEXT:   arith.constant 3 : i32
//  CHECK-NEXT: return
