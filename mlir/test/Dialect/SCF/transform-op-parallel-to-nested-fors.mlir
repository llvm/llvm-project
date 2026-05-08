// RUN: mlir-opt %s --transform-interpreter --split-input-file --verify-diagnostics | FileCheck %s

func.func private @callee(%i: index, %j: index)

func.func @two_iters(%lb1: index, %lb2: index, %ub1: index, %ub2: index, %step1: index, %step2: index) {
  scf.parallel (%i, %j) = (%lb1, %lb2) to (%ub1, %ub2) step (%step1, %step2) {
    func.call @callee(%i, %j) : (index, index) -> ()
  }
  // CHECK:           scf.for %[[VAL_0:.*]] = %[[ARG0:.*]] to %[[ARG2:.*]] step %[[ARG4:.*]] {
  // CHECK:             scf.for %[[VAL_1:.*]] = %[[ARG1:.*]] to %[[ARG3:.*]] step %[[ARG5:.*]] {
  // CHECK:               func.call @callee(%[[VAL_0]], %[[VAL_1]]) : (index, index) -> ()
  // CHECK:             }
  // CHECK:           }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.parallel"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.loop.parallel_for_to_nested_fors %0 : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// -----

func.func private @callee(%i: index, %j: index)

func.func @repeated(%lb1: index, %lb2: index, %ub1: index, %ub2: index, %step1: index, %step2: index) {
  scf.parallel (%i, %j) = (%lb1, %lb2) to (%ub1, %ub2) step (%step1, %step2) {
    func.call @callee(%i, %j) : (index, index) -> ()
  }

  scf.parallel (%i, %j) = (%lb1, %lb2) to (%ub1, %ub2) step (%step1, %step2) {
    func.call @callee(%i, %j) : (index, index) -> ()
  }

  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.parallel"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected a single payload op}}
    transform.loop.parallel_for_to_nested_fors %0 : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// -----

// expected-note @below {{payload op}}
func.func private @callee(%i: index, %j: index)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected the payload to be scf.parallel}}
    transform.loop.parallel_for_to_nested_fors %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
