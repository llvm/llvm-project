// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// Tests verification of successors with a non-trivial predicate.

// CHECK-LABEL: func @fallthrough_ok
func.func @fallthrough_ok() {
  // CHECK: test.fallthrough_br ^bb1 forward[]
  test.fallthrough_br ^bb1 forward []
^bb1:
  return
}

// -----

// CHECK-LABEL: func @forward_successors_ok
func.func @forward_successors_ok() {
  // CHECK: test.fallthrough_br ^bb1 forward[^bb2, ^bb3]
  test.fallthrough_br ^bb1 forward [^bb2, ^bb3]
^bb1:
  return
^bb2:
  return
^bb3:
  return
}

// -----

func.func @fallthrough_not_next_block() {
  // expected-error @+1 {{successor #0 ('target') failed to verify constraint: the fallthrough block (the block immediately following the op's block)}}
  test.fallthrough_br ^bb2 forward []
^bb1:
  return
^bb2:
  return
}

// -----

func.func @forward_successor_is_backward() {
  cf.br ^bb1
^bb1:
  cf.br ^bb2
^bb2:
  // expected-error @+1 {{successor #1 ('forwardTargets') failed to verify constraint: a forward block (a block listed after the op's block)}}
  test.fallthrough_br ^bb3 forward [^bb1]
^bb3:
  return
}
