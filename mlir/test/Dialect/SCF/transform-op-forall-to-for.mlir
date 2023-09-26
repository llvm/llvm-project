// RUN: mlir-opt %s --test-transform-dialect-interpreter --split-input-file --verify-diagnostics | FileCheck %s

func.func private @callee(%i: index, %j: index)

// CHECK-LABEL: @two_iters
// CHECK-SAME: %[[UB1:.+]]: index, %[[UB2:.+]]: index
func.func @two_iters(%ub1: index, %ub2: index) {
  scf.forall (%i, %j) in (%ub1, %ub2) {
    func.call @callee(%i, %j) : (index, index) -> ()
  }
  // CHECK: scf.for %[[IV1:.+]] = %{{.*}} to %[[UB1]]
  // CHECK:   scf.for %[[IV2:.+]] = %{{.*}} to %[[UB2]]
  // CHECK:     func.call @callee(%[[IV1]], %[[IV2]])
  return
}

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %0 = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  transform.loop.forall_to_for %0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
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

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %0 = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{expected a single payload op}}
  transform.loop.forall_to_for %0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
}

// -----

func.func private @callee(%i: index, %j: index)

func.func @repeated(%ub1: index, %ub2: index) {
  // expected-note @below {{payload op}}
  scf.forall (%i, %j) in (%ub1, %ub2) {
    func.call @callee(%i, %j) : (index, index) -> ()
  }
  return
}

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %0 = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{op expects as many results (1) as payload has induction variables (2)}}
  transform.loop.forall_to_for %0 : (!transform.any_op) -> !transform.any_op
}

// -----

// expected-note @below {{payload op}}
func.func private @callee(%i: index, %j: index)

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{expected the payload to be scf.forall}}
  transform.loop.forall_to_for %0 : (!transform.any_op) -> !transform.any_op
}
