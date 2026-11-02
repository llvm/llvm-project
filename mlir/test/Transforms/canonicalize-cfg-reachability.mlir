// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(canonicalize{max-iterations=1}))' | FileCheck %s

// A single iteration shows which blocks the driver still processes after a
// rewrite disconnects a block. These tests check that blocks which remain
// reachable are not skipped; they pass without reachability tracking. The
// regression tests for rewriting disconnected blocks are in
// canonicalize-dce.mlir. The constant branches sit behind a loop header, so
// that the reachability cache is populated before they fold.

// Folding the constant branch drops the only edge into ^dead. The remaining
// blocks must stay eligible: the addition in ^next folds in the same iteration.
// CHECK-LABEL: func @lose_only_predecessor
// CHECK-NOT: arith.addi
// CHECK: cf.cond_br %arg1, ^{{.*}}(%[[A:.*]] : i32), ^[[EXIT:.*]]
// CHECK: ^[[EXIT]]:
// CHECK-NEXT: return %[[A]]
func.func @lose_only_predecessor(%x: i32, %c: i1) -> i32 {
  %true = arith.constant true
  %zero = arith.constant 0 : i32
  cf.br ^header(%x : i32)
^header(%h: i32):
  cf.cond_br %true, ^next(%h : i32), ^dead
^dead:
  cf.br ^next(%zero : i32)
^next(%a: i32):
  %r = arith.addi %a, %zero : i32
  cf.cond_br %c, ^header(%r : i32), ^exit
^exit:
  return %r : i32
}

// -----

// The dropped edge leads into a loop, so ^loop keeps a predecessor. The loop
// is unreachable and must be erased.
// CHECK-LABEL: func @lose_edge_into_loop
// CHECK-NOT: arith.addi
// CHECK: ^[[HEADER:[^:]*]]:
// CHECK-NEXT: cf.cond_br %arg1, ^[[BODY:[^,]*]], ^[[EXIT:.*]]
// CHECK: ^[[BODY]]:
// CHECK-NEXT: cf.br ^[[HEADER]]
// CHECK: ^[[EXIT]]:
// CHECK-NEXT: return
func.func @lose_edge_into_loop(%init: i32, %c: i1) {
  %true = arith.constant true
  %c1 = arith.constant 1 : i32
  cf.br ^header
^header:
  cf.cond_br %c, ^body, ^exit
^body:
  cf.cond_br %true, ^header, ^loop(%init : i32)
^loop(%iv: i32):
  %next = arith.addi %iv, %c1 : i32
  cf.br ^loop(%next : i32)
^exit:
  return
}
