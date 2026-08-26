// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(test-convert-breakable-loop-to-scf,convert-scf-to-cf,canonicalize,convert-arith-to-llvm),convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)" -o %t.mlir
// RUN: mlir-runner %t.mlir -e immediate_continue -entry-point-result=i32 | FileCheck %s --check-prefix=IMMEDIATE
// RUN: mlir-runner %t.mlir -e break_with_result -entry-point-result=i32 | FileCheck %s --check-prefix=BREAK
// RUN: mlir-runner %t.mlir -e constant_outer_break -entry-point-result=i32 | FileCheck %s --check-prefix=OUTER-BREAK
// RUN: mlir-runner %t.mlir -e constant_outer_continue -entry-point-result=i32 | FileCheck %s --check-prefix=OUTER-CONTINUE
// RUN: mlir-runner %t.mlir -e dynamic_break_depth_1 -entry-point-result=i32 | FileCheck %s --check-prefix=DYNAMIC-ONE
// RUN: mlir-runner %t.mlir -e dynamic_break_depth_2 -entry-point-result=i32 | FileCheck %s --check-prefix=DYNAMIC-TWO
// RUN: mlir-runner %t.mlir -e dynamic_filtered_depth_2 -entry-point-result=i32 | FileCheck %s --check-prefix=FILTERED-TWO
// RUN: mlir-runner %t.mlir -e dynamic_filtered_depth_3 -entry-point-result=i32 | FileCheck %s --check-prefix=FILTERED-THREE
// XFAIL: system-aix

func.func @immediate_continue() -> i32 {
  %depth = arith.constant 1 : index
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %result = test.breakable_loop iter_args(%i = %c0) : i32 -> i32 {
    %done = arith.cmpi eq, %i, %c4 : i32
    scf.if %done {
      test.dynamic_break %depth %i : i32
    }
    %next = arith.addi %i, %c1 : i32
    test.dynamic_continue %depth %next : i32
  }
  return %result : i32
}
// IMMEDIATE: 4

func.func @break_with_result() -> i32 {
  %depth = arith.constant 1 : index
  %value = arith.constant 17 : i32
  %result = test.breakable_loop -> i32 {
    test.dynamic_break %depth %value : i32
  }
  return %result : i32
}
// BREAK: 17

func.func @constant_outer_break() -> i32 {
  %inner_depth = arith.constant 1 : index
  %outer_depth = arith.constant 2 : index
  %outer_value = arith.constant 42 : i32
  %fallback = arith.constant 13 : i32
  %result = test.breakable_loop -> i32 {
    test.breakable_loop {
      test.dynamic_break %outer_depth %outer_value : i32
    }
    test.dynamic_break %inner_depth %fallback : i32
  }
  return %result : i32
}
// OUTER-BREAK: 42

func.func @constant_outer_continue() -> i32 {
  %inner_depth = arith.constant 1 : index
  %outer_depth = arith.constant 2 : index
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c3 = arith.constant 3 : i32
  %result = test.breakable_loop iter_args(%i = %c0) : i32 -> i32 {
    %done = arith.cmpi eq, %i, %c3 : i32
    scf.if %done {
      test.dynamic_break %inner_depth %i : i32
    }
    test.breakable_loop {
      %next = arith.addi %i, %c1 : i32
      test.dynamic_continue %outer_depth %next : i32
    }
    test.dynamic_break %inner_depth %i : i32
  }
  return %result : i32
}
// OUTER-CONTINUE: 3

// Distinguish the two compatible dynamic break targets at runtime. Depth 1
// breaks the inner loop, then the outer loop adds one; depth 2 skips that outer
// post-inner add by breaking directly to the outer loop result.
func.func @dynamic_break(%choose_outer: i1) -> i32 {
  %inner_depth = arith.constant 1 : index
  %outer_depth = arith.constant 2 : index
  %depth = arith.select %choose_outer, %outer_depth, %inner_depth : index
  %value = arith.constant 7 : i32
  %one = arith.constant 1 : i32
  %result = test.breakable_loop -> i32 {
    %inner_result = test.breakable_loop -> i32 {
      test.dynamic_break %depth %value : i32
    }
    %after_inner = arith.addi %inner_result, %one : i32
    test.dynamic_break %inner_depth %after_inner : i32
  }
  return %result : i32
}

func.func @dynamic_break_depth_1() -> i32 {
  %choose_outer = arith.constant false
  %result = func.call @dynamic_break(%choose_outer) : (i1) -> i32
  return %result : i32
}
// DYNAMIC-ONE: 8

func.func @dynamic_break_depth_2() -> i32 {
  %choose_outer = arith.constant true
  %result = func.call @dynamic_break(%choose_outer) : (i1) -> i32
  return %result : i32
}
// DYNAMIC-TWO: 7

// The innermost f32 loop is incompatible with the i32 break payload and is
// filtered out. Depth 2 therefore targets the middle loop and observes the
// outer post-middle add, while depth 3 targets the outer loop directly.
func.func @dynamic_break_filtered(%choose_outer: i1) -> i32 {
  %inner_depth = arith.constant 1 : index
  %middle_depth = arith.constant 2 : index
  %outer_depth = arith.constant 3 : index
  %depth = arith.select %choose_outer, %outer_depth, %middle_depth : index
  %value = arith.constant 5 : i32
  %one = arith.constant 1 : i32
  %f0 = arith.constant 0.000000e+00 : f32
  %result = test.breakable_loop -> i32 {
    %middle_result = test.breakable_loop -> i32 {
      test.breakable_loop iter_args(%inner = %f0) : f32 {
        test.dynamic_break %depth %value : i32
      }
      %after_inner = arith.addi %value, %one : i32
      test.dynamic_break %inner_depth %after_inner : i32
    }
    %after_middle = arith.addi %middle_result, %one : i32
    test.dynamic_break %inner_depth %after_middle : i32
  }
  return %result : i32
}

func.func @dynamic_filtered_depth_2() -> i32 {
  %choose_outer = arith.constant false
  %result = func.call @dynamic_break_filtered(%choose_outer) : (i1) -> i32
  return %result : i32
}
// FILTERED-TWO: 6

func.func @dynamic_filtered_depth_3() -> i32 {
  %choose_outer = arith.constant true
  %result = func.call @dynamic_break_filtered(%choose_outer) : (i1) -> i32
  return %result : i32
}
// FILTERED-THREE: 5
