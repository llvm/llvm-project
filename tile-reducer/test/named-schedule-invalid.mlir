// RUN: not tr-opt %s --transform-interpreter=entry-point=missing_schedule 2>&1 | FileCheck %s

// Symbol lookup failure: entry point is not in the symbol table.

func.func @row() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @row_sum_schedule(
      %payload: !transform.any_op {transform.readonly}) {
    transform.yield
  }
}

// CHECK: could not find a nested named sequence with name: missing_schedule
