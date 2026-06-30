// RUN: mlir-opt %s --split-input-file | FileCheck %s --check-prefix=PRINT
// RUN: mlir-opt --print-region-branch-op-interface %s --split-input-file | FileCheck %s --check-prefix=BRANCH

// PRINT-LABEL: func.func @immediate_continue(
func.func @immediate_continue(%depth: index) {
  // PRINT: test.breakable_loop
  test.breakable_loop {
    // PRINT: test.dynamic_continue %arg0
    test.dynamic_continue %depth
  }
  return
}

// -----

// PRINT-LABEL: func.func @break_with_result(
func.func @break_with_result(%depth: index, %value: i32) -> i32 {
  // PRINT: test.breakable_loop -> i32
  %result = test.breakable_loop -> i32 {
    // PRINT: test.dynamic_break %arg0 %arg1 : i32
    test.dynamic_break %depth %value : i32
  }
  return %result : i32
}

// -----

// PRINT-LABEL: func.func @continue_iter_args(
func.func @continue_iter_args(%depth: index, %init: i32) {
  // PRINT: test.breakable_loop iter_args(%{{.*}} = %arg1) : i32
  test.breakable_loop iter_args(%iter = %init) : i32 {
    // PRINT: test.dynamic_continue %arg0 %{{.*}} : i32
    test.dynamic_continue %depth %iter : i32
  }
  return
}

// -----

// PRINT-LABEL: func.func @constant_depth_selects_outer(
func.func @constant_depth_selects_outer(%outer_init: i32, %inner_init: f32) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  test.breakable_loop iter_args(%outer = %outer_init) : i32 {
    test.breakable_loop iter_args(%inner = %inner_init) : f32 {
      // PRINT: test.dynamic_continue %c2 %{{.*}} : i32
      test.dynamic_continue %c2 %outer : i32
    }
    test.dynamic_continue %c1 %outer : i32
  }
  return
}

// -----

// PRINT-LABEL: func.func @through_scf_if(
func.func @through_scf_if(%cond: i1, %depth: index) {
  %c1 = arith.constant 1 : index
  // BRANCH-LABEL: Found RegionBranchOpInterface operation: test.breakable_loop {{.*}} loc("loop")
  // BRANCH:  - Successor is region #0
  // BRANCH:  - Found 2 predecessor(s)
  // BRANCH:  - Predecessor is test.dynamic_break {{.*}} loc("if_break")
  // BRANCH:  - Predecessor is test.dynamic_continue {{.*}} loc("after_if")
  test.breakable_loop {
    // PRINT: scf.if
    scf.if %cond {
      // PRINT: test.dynamic_break %arg1
      test.dynamic_break %depth loc("if_break")
    } loc("if")
    test.dynamic_continue %c1 loc("after_if")
  } loc("loop")
  return
}

// -----

// A dynamic depth can target either the immediately enclosing loop or an outer
// compatible loop. The inner loop sees the dynamic break as a direct
// predecessor, and the outer loop sees both the propagated dynamic break and
// its own explicit continue.
func.func @nested_dynamic(%depth: index) {
  %c1 = arith.constant 1 : index
  // BRANCH-LABEL: Found RegionBranchOpInterface operation: test.breakable_loop {{.*}} loc("outer")
  // BRANCH:  - Successor is region #0
  // BRANCH:  - Found 2 predecessor(s)
  // BRANCH:  - Predecessor is test.dynamic_break {{.*}} loc("dyn_break")
  // BRANCH:  - Predecessor is test.dynamic_continue {{.*}} loc("outer_continue")
  test.breakable_loop {
    // BRANCH-LABEL: Found RegionBranchOpInterface operation: test.breakable_loop {{.*}} loc("inner")
    // BRANCH:  - Successor is region #0
    // BRANCH:  - Found 1 predecessor(s)
    // BRANCH:  - Predecessor is test.dynamic_break {{.*}} loc("dyn_break")
    test.breakable_loop {
      test.dynamic_break %depth loc("dyn_break")
    } loc("inner")
    test.dynamic_continue %c1 loc("outer_continue")
  } loc("outer")
  return
}

// -----

// The dynamic continue payload is f32. That makes the inner f32 loop a
// potential target, but filters out the enclosing i32 loop. The outer loop
// should only see its explicit continue predecessor.
func.func @dynamic_continue_filters_incompatible_outer(%depth: index, %i: i32,
                                                       %f: f32) {
  %c1 = arith.constant 1 : index
  // BRANCH-LABEL: Found RegionBranchOpInterface operation: test.breakable_loop {{.*}} loc("filtered_outer")
  // BRANCH:  - Successor is region #0
  // BRANCH:  - Found 1 predecessor(s)
  // BRANCH:  - Predecessor is test.dynamic_continue {{.*}} loc("filtered_outer_continue")
  test.breakable_loop iter_args(%outer = %i) : i32 {
    // BRANCH-LABEL: Found RegionBranchOpInterface operation: test.breakable_loop {{.*}} loc("filtered_inner")
    // BRANCH:  - Successor is region #0
    // BRANCH:  - Found 1 predecessor(s)
    // BRANCH:  - Predecessor is test.dynamic_continue {{.*}} loc("filtered_inner_continue")
    test.breakable_loop iter_args(%inner = %f) : f32 {
      test.dynamic_continue %depth %inner : f32 loc("filtered_inner_continue")
    } loc("filtered_inner")
    test.dynamic_continue %c1 %outer : i32 loc("filtered_outer_continue")
  } loc("filtered_outer")
  return
}
