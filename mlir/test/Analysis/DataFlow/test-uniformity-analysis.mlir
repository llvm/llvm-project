// RUN: mlir-opt -split-input-file -test-uniformity-analysis -verify-diagnostics %s

// A memory-effect-free operation of a transparent dialect is the join of its
// operands, which is the narrower scope; a constant is uniform.
func.func @join() {
  // expected-remark @below {{uniformity of "div": results = [divergent], execution = uniform}}
  %div = test.with_uniformity {scope = "divergent", tag = "div"} : index
  %sg = test.with_uniformity {scope = "subgroup"} : index
  %wg = test.with_uniformity {scope = "workgroup"} : index
  %cl = test.with_uniformity {scope = "cluster"} : index
  // expected-remark @below {{uniformity of "uni": results = [uniform], execution = uniform}}
  %uni = test.with_uniformity {scope = "uniform", tag = "uni"} : index
  // expected-remark @below {{uniformity of "c": results = [uniform], execution = uniform}}
  %c = arith.constant {tag = "c"} 4 : index
  // expected-remark @below {{uniformity of "uni_c": results = [uniform], execution = uniform}}
  %uni_c = arith.addi %uni, %c {tag = "uni_c"} : index
  // expected-remark @below {{uniformity of "wg_c": results = [workgroup], execution = uniform}}
  %wg_c = arith.addi %wg, %c {tag = "wg_c"} : index
  // expected-remark @below {{uniformity of "wg_sg": results = [subgroup], execution = uniform}}
  %wg_sg = arith.addi %wg, %sg {tag = "wg_sg"} : index
  // expected-remark @below {{uniformity of "cl_wg": results = [workgroup], execution = uniform}}
  %cl_wg = arith.addi %cl, %wg {tag = "cl_wg"} : index
  // expected-remark @below {{uniformity of "div_uni": results = [divergent], execution = uniform}}
  %div_uni = arith.addi %div, %uni {tag = "div_uni"} : index
  // expected-remark @below {{uniformity of "cmp": results = [subgroup], execution = uniform}}
  %cmp = arith.cmpi ult, %sg, %cl {tag = "cmp"} : index
  // A select of two uniform values picks the same value within the group
  // that agrees on the condition, and no wider.
  // expected-remark @below {{uniformity of "select": results = [subgroup], execution = uniform}}
  %sel = arith.select %cmp, %uni, %c {tag = "select"} : index
  return
}

// -----

// What is not a function of its operands: an operation that reads or allocates
// memory, an operation whose region captures values from above without region
// control flow, and an operation of a dialect the analysis was not told is
// transparent, even a pure one.
func.func @opaque() {
  %c0 = arith.constant 0 : index
  %uni = test.with_uniformity {scope = "uniform"} : index
  // expected-remark @below {{uniformity of "alloca": results = [divergent], execution = uniform}}
  %m = memref.alloca() {tag = "alloca"} : memref<4xindex>
  // expected-remark @below {{uniformity of "load": results = [divergent], execution = uniform}}
  %l = memref.load %m[%c0] {tag = "load"} : memref<4xindex>
  // expected-remark @below {{uniformity of "generate": results = [divergent], execution = uniform}}
  %t = tensor.generate {
  ^bb0(%i: index):
    tensor.yield %uni : index
  } {tag = "generate"} : tensor<4xindex>
  // expected-remark @below {{uniformity of "not_transparent": results = [divergent], execution = uniform}}
  %r = test.increment {tag = "not_transparent"} %uni : index
  return
}

// -----

// Structured control flow. The results of a region branch are the join of the
// values forwarded to them, tainted by the operands steering the branch; the
// entry block arguments of a region are only the join of the forwarded values.
// An operation executes as widely as the narrowest control operand around it.
func.func @structured() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %uni_flag = test.with_uniformity {scope = "uniform"} : i1
  %wg_flag = test.with_uniformity {scope = "workgroup"} : i1
  %div_flag = test.with_uniformity {scope = "divergent"} : i1
  %n = test.with_uniformity {scope = "uniform"} : index
  %tid = test.with_uniformity {scope = "divergent"} : index
  // expected-remark @below {{uniformity of "if_div": results = [divergent], execution = uniform}}
  %r0 = scf.if %div_flag -> index {
    // expected-remark @below {{uniformity of "in_then": results = [uniform], execution = divergent}}
    %a = arith.addi %c1, %c1 {tag = "in_then"} : index
    scf.yield %a : index
  } else {
    scf.yield %c0 : index
  } {tag = "if_div"}
  // expected-remark @below {{uniformity of "if_uni": results = [uniform], execution = uniform}}
  %r1 = scf.if %uni_flag -> index {
    scf.yield %c1 : index
  } else {
    scf.yield %c0 : index
  } {tag = "if_uni"}
  // expected-remark @below {{uniformity of "if_wg": results = [workgroup], execution = uniform}}
  %r2 = scf.if %wg_flag -> index {
    // expected-remark @below {{uniformity of "in_wg_then": results = [uniform], execution = workgroup}}
    %a = arith.addi %c1, %c1 {tag = "in_wg_then"} : index
    scf.yield %a : index
  } else {
    scf.yield %c0 : index
  } {tag = "if_wg"}
  // A loop with uniform bounds has uniform results, and its iteration argument
  // is the join of the initial value and the yielded value.
  // expected-remark @below {{uniformity of "for_uni": results = [uniform], execution = uniform}}
  %r3 = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c0) -> index {
    // expected-remark @below {{uniformity of "acc_uni": results = [uniform], execution = uniform}}
    %next = arith.addi %acc, %c1 {tag = "acc_uni"} : index
    scf.yield %next : index
  } {tag = "for_uni"}
  // A loop whose trip count differs between threads exits with different
  // values and its body executes divergently, but the threads still iterating
  // observe the same iteration argument.
  // expected-remark @below {{uniformity of "for_div_ub": results = [divergent], execution = uniform}}
  %r4 = scf.for %j = %c0 to %tid step %c1 iter_args(%acc = %c0) -> index {
    // expected-remark @below {{uniformity of "acc_div_ub": results = [uniform], execution = divergent}}
    %next = arith.addi %acc, %c1 {tag = "acc_div_ub"} : index
    scf.yield %next : index
  } {tag = "for_div_ub"}
  // Nested branches: the narrowest control operand wins.
  scf.if %wg_flag {
    // expected-remark @below {{uniformity of "in_wg_if": results = [uniform], execution = workgroup}}
    %a = arith.constant {tag = "in_wg_if"} 1 : index
    scf.if %div_flag {
      // expected-remark @below {{uniformity of "nested": results = [uniform], execution = divergent}}
      %b = arith.constant {tag = "nested"} 1 : index
    }
    scf.if %uni_flag {
      // expected-remark @below {{uniformity of "nested_uni": results = [uniform], execution = workgroup}}
      %d = arith.constant {tag = "nested_uni"} 1 : index
    }
  }
  return
}

// -----

// scf.while: the condition steers the results and the execution of the body.
func.func @while() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = test.with_uniformity {scope = "uniform"} : index
  %tid = test.with_uniformity {scope = "divergent"} : index
  // expected-remark @below {{uniformity of "while_div": results = [divergent], execution = uniform}}
  %r = scf.while (%i = %c0) : (index) -> index {
    %cond = arith.cmpi ult, %i, %tid : index
    scf.condition(%cond) %i : index
  } do {
  ^bb0(%i: index):
    // expected-remark @below {{uniformity of "in_while": results = [uniform], execution = divergent}}
    %next = arith.addi %i, %c1 {tag = "in_while"} : index
    scf.yield %next : index
  } attributes {tag = "while_div"}
  // expected-remark @below {{uniformity of "while_uni": results = [uniform], execution = uniform}}
  %s = scf.while (%i = %c0) : (index) -> index {
    %cond = arith.cmpi ult, %i, %n : index
    scf.condition(%cond) %i : index
  } do {
  ^bb0(%i: index):
    // expected-remark @below {{uniformity of "in_while_uni": results = [uniform], execution = uniform}}
    %next = arith.addi %i, %c1 {tag = "in_while_uni"} : index
    scf.yield %next : index
  } attributes {tag = "while_uni"}
  return
}

// -----

// Unstructured control flow: the arguments of a non-entry block are tainted by
// every branch of the region, and so is their execution.
func.func @cfg() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cond = test.with_uniformity {scope = "divergent"} : i1
  cf.cond_br %cond, ^bb1(%c0 : index), ^bb1(%c1 : index)
^bb1(%x: index):
  // expected-remark @below {{uniformity of "phi": results = [divergent], execution = divergent}}
  %y = arith.addi %x, %c1 {tag = "phi"} : index
  return
}

func.func @cfg_uniform() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cond = test.with_uniformity {scope = "uniform"} : i1
  cf.cond_br %cond, ^bb1(%c0 : index), ^bb1(%c1 : index)
^bb1(%x: index):
  // expected-remark @below {{uniformity of "phi_uni": results = [uniform], execution = uniform}}
  %y = arith.addi %x, %c1 {tag = "phi_uni"} : index
  return
}

// -----

// A launch boundary: its body is executed by the threads of the launch,
// whatever control flow the host wrapped the launch in, and the operation
// describes the arguments of its body.
func.func @launch() {
  %c1 = arith.constant 1 : index
  %go = test.with_uniformity {scope = "divergent"} : i1
  scf.if %go {
    test.uniformity_launch ["workgroup", "divergent"] {
    ^bb0(%block: index, %thread: index):
      // expected-remark @below {{uniformity of "launch_block": results = [workgroup], execution = uniform}}
      %a = arith.addi %block, %c1 {tag = "launch_block"} : index
      // expected-remark @below {{uniformity of "launch_thread": results = [divergent], execution = uniform}}
      %b = arith.addi %thread, %c1 {tag = "launch_thread"} : index
    }
  }
  return
}

// -----

// A value an operation could not describe yet, because an operand had not
// been visited, must not be pinned to divergent: the loop result below is only
// known once the regions of the while have settled, and the operation using
// it is visited first.
func.func @late_operand() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = test.with_uniformity {scope = "uniform"} : index
  %r = scf.while (%i = %c0) : (index) -> index {
    %c = arith.cmpi ult, %i, %n : index
    scf.condition(%c) %i : index
  } do {
  ^bb0(%i: index):
    %next = arith.addi %i, %c1 : index
    scf.yield %next : index
  }
  // expected-remark @below {{uniformity of "late": results = [uniform, divergent], execution = uniform}}
  %same, %other = test.uniformity_of %r {tag = "late"} : index
  return
}

// -----

// A callable the analysis is not told about receives divergent arguments: it
// joins every call site, and nothing says a function is only called from
// uniform control flow. Execution uniformity stops at the callable, so the
// body of a callee is reported as executed by every thread whatever the call
// site.
func.func @callee(%x: index) -> index {
  // expected-remark @below {{uniformity of "callee_arg": results = [divergent], execution = uniform}}
  %a = arith.addi %x, %x {tag = "callee_arg"} : index
  return %a : index
}

func.func @caller() {
  %n = test.with_uniformity {scope = "uniform"} : index
  %go = test.with_uniformity {scope = "divergent"} : i1
  // expected-remark @below {{uniformity of "call": results = [divergent], execution = uniform}}
  %r = func.call @callee(%n) {tag = "call"} : (index) -> index
  scf.if %go {
    %s = func.call @callee(%n) : (index) -> index
  }
  return
}

// -----

// A region branch that describes its successor inputs itself, like
// gpu.warp_execute_on_lane_0. Setting one input of a successor makes the
// operation authoritative for that successor: the forwarded values are not
// joined into its inputs, and the inputs it leaves undescribed are put in the
// entry state. A successor the operation says nothing about is the usual join,
// tainted by the control operand for the results only.
func.func @authoritative() {
  %div = test.with_uniformity {scope = "divergent"} : index
  %wg = test.with_uniformity {scope = "workgroup"} : index
  %uni = test.with_uniformity {scope = "uniform"} : index
  %uni_flag = test.with_uniformity {scope = "uniform"} : i1
  %wg_flag = test.with_uniformity {scope = "workgroup"} : i1
  // The result is declared subgroup although the region yields a divergent
  // value; the body argument, which the operation does not describe, is the
  // join of the divergent value forwarded to it.
  // expected-remark @below {{uniformity of "result_set": results = [subgroup], execution = uniform}}
  %r0 = test.uniformity_region %uni_flag(%div) results ["subgroup"] : (index) -> index {
  ^bb0(%x: index):
    // expected-remark @below {{uniformity of "result_set_arg": results = [divergent], execution = uniform}}
    %y = arith.addi %x, %x {tag = "result_set_arg"} : index
    test.region_if_yield %y : index
  } {tag = "result_set"}
  // The body argument is declared uniform although a divergent value is
  // forwarded to it; the result, which the operation does not describe, is
  // the join of the yielded value.
  // expected-remark @below {{uniformity of "arg_set": results = [workgroup], execution = uniform}}
  %r1 = test.uniformity_region %uni_flag(%div) args ["uniform"] : (index) -> index {
  ^bb0(%x: index):
    // expected-remark @below {{uniformity of "arg_set_arg": results = [uniform], execution = uniform}}
    %y = arith.addi %x, %x {tag = "arg_set_arg"} : index
    test.region_if_yield %wg : index
  } {tag = "arg_set"}
  // Nothing is described: the result is the join of the uniform yielded value
  // and of the workgroup control operand, the body argument is only the join
  // of the forwarded value, and the body executes within the workgroup.
  // expected-remark @below {{uniformity of "unset": results = [workgroup], execution = uniform}}
  %r2 = test.uniformity_region %wg_flag(%uni) : (index) -> index {
  ^bb0(%x: index):
    // expected-remark @below {{uniformity of "unset_arg": results = [uniform], execution = workgroup}}
    %y = arith.addi %x, %x {tag = "unset_arg"} : index
    test.region_if_yield %y : index
  } {tag = "unset"}
  // Setting one result makes the operation authoritative for both: the one it
  // leaves undescribed is put in the entry state, not joined with the uniform
  // value yielded to it.
  // expected-remark @below {{uniformity of "partial": results = [subgroup, divergent], execution = uniform}}
  %r3:2 = test.uniformity_region %uni_flag(%uni) results ["subgroup", ""] : (index) -> (index, index) {
  ^bb0(%x: index):
    test.region_if_yield %x, %x : index, index
  } {tag = "partial"}
  return
}

// -----

// The results of a call are the join of what every return site yields, and
// threads that leave the callee through different return sites observe
// different results: the control flow that selects the return site taints
// them, even when every return site yields a uniform value. The callees are
// private so that every call site is known and their arguments are the join
// of what the callers pass.
func.func private @two_returns(%c: i1) -> index {
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  cf.cond_br %c, ^bb1, ^bb2
^bb1:
  return %one : index
^bb2:
  return %two : index
}

func.func private @two_returns_uniform(%c: i1) -> index {
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  cf.cond_br %c, ^bb1, ^bb2
^bb1:
  return %one : index
^bb2:
  return %two : index
}

func.func @call_return_sites() {
  %div = test.with_uniformity {scope = "divergent"} : i1
  %uni = test.with_uniformity {scope = "uniform"} : i1
  // expected-remark @below {{uniformity of "call_div_sites": results = [divergent], execution = uniform}}
  %r0 = func.call @two_returns(%div) {tag = "call_div_sites"} : (i1) -> index
  // expected-remark @below {{uniformity of "call_uni_sites": results = [uniform], execution = uniform}}
  %r1 = func.call @two_returns_uniform(%uni) {tag = "call_uni_sites"} : (i1) -> index
  return
}
