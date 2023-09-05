// RUN: mlir-opt %s --test-transform-dialect-interpreter --split-input-file | FileCheck %s

func.func @select_single_i1_vector(%cond : i1) -> vector<1xi1> {
  %true = arith.constant dense<true> : vector<1xi1>
  %false = arith.constant dense<false> : vector<1xi1>
  %select = arith.select %cond, %true, %false : i1, vector<1xi1>
  return %select : vector<1xi1>
}

transform.sequence failures(propagate) {
^bb1(%func_op: !transform.op<"func.func">):
  transform.apply_patterns to %func_op {
    transform.apply_patterns.vector.materialize_masks
  } : !transform.op<"func.func">
}

// CHECK-LABEL: func @select_single_i1_vector
// CHECK-SAME: %[[COND:.*]]: i1
// CHECK:      %[[BCAST:.*]] = vector.broadcast %[[COND]] : i1 to vector<1xi1>
// CHECK:      return %[[BCAST]] : vector<1xi1>
