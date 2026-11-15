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

// -----

// A block that starts with a label-like op is kept although it has no
// predecessors, but canonicalization does not rewrite its operations.
// CHECK-LABEL: func @label_block
// CHECK:       ^bb1:
// CHECK-NEXT:    test.label "L"
// CHECK-NEXT:    %[[SUM:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    return %[[SUM]] : i32
func.func @label_block(%x: i32) -> i32 {
  %zero = arith.constant 0 : i32
  return %x : i32
^bb1:
  test.label "L"
  %sum = arith.addi %x, %zero : i32
  return %sum : i32
}

// -----

// Region DCE keeps the values and block arguments used in blocks that only a
// label reaches.
// CHECK-LABEL: func @label_block_arguments
// CHECK:       test.label "L"
// CHECK-NEXT:    cf.br ^[[NEXT:bb[0-9]+]](%{{.*}} : i32)
// CHECK-NEXT:  ^[[NEXT]](%[[A:.*]]: i32):
// CHECK-NEXT:    return %[[A]] : i32
func.func @label_block_arguments(%x: i32) -> i32 {
  return %x : i32
^label:
  test.label "L"
  cf.br ^next(%x : i32)
^next(%a: i32):
  return %a : i32
}

// -----

// The verifier does not check dominance in a block that only a label reaches,
// so it may use a value of a block that nothing reaches. That block is kept
// too, so that erasing it leaves no use behind.
// CHECK-LABEL: func @label_block_uses_dead_value
// CHECK:       ^bb1:
// CHECK-NEXT:    %[[V:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    return %[[V]] : i32
// CHECK-NEXT:  ^bb2:
// CHECK-NEXT:    test.label "L"
// CHECK-NEXT:    return %[[V]] : i32
func.func @label_block_uses_dead_value(%x: i32) -> i32 {
  return %x : i32
^dead:
  %v = arith.addi %x, %x : i32
  return %v : i32
^label:
  test.label "L"
  return %v : i32
}
