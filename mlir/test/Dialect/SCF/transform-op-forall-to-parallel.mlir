// RUN: mlir-opt %s --transform-interpreter --split-input-file --verify-diagnostics | FileCheck %s

func.func private @callee(%i: index, %j: index)

// CHECK-LABEL: @two_iters
// CHECK-SAME: %[[UB1:.+]]: index, %[[UB2:.+]]: index
func.func @two_iters(%ub1: index, %ub2: index) {
  scf.forall (%i, %j) in (%ub1, %ub2) {
    func.call @callee(%i, %j) : (index, index) -> ()
  }
  // CHECK: scf.parallel (%[[IV1:.+]], %[[IV2:.+]]) = (%{{.*}}, %{{.*}}) to (%[[UB1]], %[[UB2]])
  // CHECK:   func.call @callee(%[[IV1]], %[[IV2]]) : (index, index) -> ()
  // CHECK:   scf.reduce
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.loop.forall_to_parallel %0 : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// -----

func.func private @callee(%i: index, %j: index)

func.func @repeated(%ub1: index, %ub2: index) {
  scf.forall (%i, %j) in (%ub1, %ub2) {
    func.call @callee(%i, %j) : (index, index) -> ()
  }
  scf.forall (%i, %j) in (%ub1, %ub2) {
    func.call @callee(%i, %j) : (index, index) -> ()
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected a single payload op}}
    transform.loop.forall_to_parallel %0 : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// -----

// expected-note @below {{payload op}}
func.func private @callee(%i: index, %j: index)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected the payload to be scf.forall}}
    transform.loop.forall_to_for %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
