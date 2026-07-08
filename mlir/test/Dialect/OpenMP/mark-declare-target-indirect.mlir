// RUN: mlir-opt -omp-mark-declare-target -split-input-file %s | FileCheck %s

// The `omp-mark-declare-target` pass marks functions that are reachable from
// explicit target code as implicitly declare target. The `indirect` modifier
// however is a property of the specific declare target declaration and must NOT
// be propagated to functions that are only reached through (direct) calls.

// A function explicitly declared `indirect` that directly calls another
// function: the callee is implicitly captured and must be marked declare target
// with `indirect = false`, not inherit the parent's `indirect = true`.
module {
  // CHECK: func.func @indirect_parent() attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter), automap = false, indirect = true>}
  func.func @indirect_parent() attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter), automap = false, indirect = true>} {
    func.call @direct_callee() : () -> ()
    return
  }

  // CHECK: func.func @direct_callee() attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter), automap = false, indirect = false>}
  func.func @direct_callee() {
    return
  }
}

// -----

// A callee that is itself explicitly declared `indirect` keeps its own value
// (the pass must not clobber it).
module {
  // CHECK: func.func @indirect_parent2() attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter), automap = false, indirect = true>}
  func.func @indirect_parent2() attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter), automap = false, indirect = true>} {
    func.call @explicitly_indirect_callee() : () -> ()
    return
  }

  // CHECK: func.func @explicitly_indirect_callee() attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter), automap = false, indirect = true>}
  func.func @explicitly_indirect_callee() attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter), automap = false, indirect = true>} {
    return
  }
}
