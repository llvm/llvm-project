// RUN: mlir-opt %s --test-convert-breakable-loop-to-scf --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @immediate_continue(
// CHECK-SAME: %[[INIT:.*]]: i32
func.func @immediate_continue(%init: i32) -> i32 {
  %c1 = arith.constant 1 : index
  // CHECK: %[[RESULT:.*]] = scf.loop token(%[[TOKEN:.*]]) iter_args(%[[ITER:.*]] = %[[INIT]]) : i32 -> i32
  %result = test.breakable_loop iter_args(%iter = %init) : i32 -> i32 {
    // CHECK: scf.continue [%[[TOKEN]]] %[[ITER]] : i32
    test.dynamic_continue %c1 %iter : i32
  }
  // CHECK: return %[[RESULT]] : i32
  return %result : i32
}

// -----

// CHECK-LABEL: func.func @constant_outer_break(
func.func @constant_outer_break(%init: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // CHECK: %[[RESULT:.*]] = scf.loop token(%[[OUTER_TOKEN:.*]]) iter_args(%[[OUTER_ARG:.*]] =
  %result = test.breakable_loop iter_args(%outer = %init) : i32 -> i32 {
    // CHECK: scf.loop token(%[[INNER_TOKEN:.*]]) {
    test.breakable_loop {
      // CHECK: scf.break [%[[OUTER_TOKEN]]] %[[OUTER_ARG]] : i32
      test.dynamic_break %c2 %outer : i32
    }
    test.dynamic_break %c1 %outer : i32
  }
  // CHECK: return %[[RESULT]] : i32
  return %result : i32
}

// -----

// Dynamic break over two compatible loops lowers to a dispatch ladder with one
// arm for depth 1 (inner) and one for depth 2 (outer), plus a deterministic
// fallback for UB depth values.
// CHECK-LABEL: func.func @dynamic_break_dispatch(
// CHECK-SAME: %[[CHOOSE:.*]]: i1, %[[INIT:.*]]: i32
func.func @dynamic_break_dispatch(%choose_outer: i1, %init: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // CHECK: %[[DEPTH:.*]] = arith.select %[[CHOOSE]],
  %depth = arith.select %choose_outer, %c2, %c1 : index
  // CHECK: %[[RESULT:.*]] = scf.loop token(%[[OUTER_TOKEN:.*]]) iter_args(%[[OUTER_ARG:.*]] =
  %result = test.breakable_loop iter_args(%outer = %init) : i32 -> i32 {
    // CHECK: %[[INNER_RESULT:.*]] = scf.loop token(%[[INNER_TOKEN:.*]]) iter_args(%[[INNER_ARG:.*]] =
    %inner_result = test.breakable_loop iter_args(%inner = %outer) : i32 -> i32 {
      // CHECK: %[[DEPTH_ONE:.*]] = arith.constant 1 : index
      // CHECK: %[[IS_ONE:.*]] = arith.cmpi eq, %[[DEPTH]], %[[DEPTH_ONE]] : index
      // CHECK: scf.if %[[IS_ONE]] {
      // CHECK:   scf.break [%[[INNER_TOKEN]]] %[[INNER_ARG]] : i32
      // CHECK: %[[DEPTH_TWO:.*]] = arith.constant 2 : index
      // CHECK: %[[IS_TWO:.*]] = arith.cmpi eq, %[[DEPTH]], %[[DEPTH_TWO]] : index
      // CHECK: scf.if %[[IS_TWO]] {
      // CHECK:   scf.break [%[[OUTER_TOKEN]]] %[[INNER_ARG]] : i32
      // CHECK: scf.break [%[[INNER_TOKEN]]] %[[INNER_ARG]] : i32
      test.dynamic_break %depth %inner : i32
    }
    test.dynamic_break %c1 %inner_result : i32
  }
  // CHECK: return %[[RESULT]] : i32
  return %result : i32
}

// -----

// The same dispatch shape is required for dynamic continue, except each arm
// targets the next iteration of the selected loop.
// CHECK-LABEL: func.func @dynamic_continue_dispatch(
// CHECK-SAME: %[[CHOOSE:.*]]: i1, %[[INIT:.*]]: i32
func.func @dynamic_continue_dispatch(%choose_outer: i1, %init: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // CHECK: %[[DEPTH:.*]] = arith.select %[[CHOOSE]],
  %depth = arith.select %choose_outer, %c2, %c1 : index
  // CHECK: %[[RESULT:.*]] = scf.loop token(%[[OUTER_TOKEN:.*]]) iter_args(%[[OUTER_ARG:.*]] =
  %result = test.breakable_loop iter_args(%outer = %init) : i32 -> i32 {
    // CHECK: scf.loop token(%[[INNER_TOKEN:.*]]) iter_args(%[[INNER_ARG:.*]] =
    test.breakable_loop iter_args(%inner = %outer) : i32 {
      // CHECK: %[[DEPTH_ONE:.*]] = arith.constant 1 : index
      // CHECK: %[[IS_ONE:.*]] = arith.cmpi eq, %[[DEPTH]], %[[DEPTH_ONE]] : index
      // CHECK: scf.if %[[IS_ONE]] {
      // CHECK:   scf.continue [%[[INNER_TOKEN]]] %[[INNER_ARG]] : i32
      // CHECK: %[[DEPTH_TWO:.*]] = arith.constant 2 : index
      // CHECK: %[[IS_TWO:.*]] = arith.cmpi eq, %[[DEPTH]], %[[DEPTH_TWO]] : index
      // CHECK: scf.if %[[IS_TWO]] {
      // CHECK:   scf.continue [%[[OUTER_TOKEN]]] %[[INNER_ARG]] : i32
      // CHECK: scf.continue [%[[INNER_TOKEN]]] %[[INNER_ARG]] : i32
      test.dynamic_continue %depth %inner : i32
    }
    test.dynamic_break %c1 %outer : i32
  }
  // CHECK: return %[[RESULT]] : i32
  return %result : i32
}

// -----

// The innermost loop carries f32, so it is not compatible with the i32 break
// payload and must be filtered out. The remaining compatible targets are still
// addressed by their original depths: 2 for the middle loop and 3 for the outer
// loop.
// CHECK-LABEL: func.func @dynamic_break_dispatch_filtered_depths(
// CHECK-SAME: %[[CHOOSE:.*]]: i1, %[[INIT:.*]]: i32, %[[F_INIT:.*]]: f32
func.func @dynamic_break_dispatch_filtered_depths(%choose_outer: i1,
                                                  %init: i32, %inner_init: f32)
    -> i32 {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  // CHECK: %[[DEPTH:.*]] = arith.select %[[CHOOSE]],
  %depth = arith.select %choose_outer, %c3, %c2 : index
  // CHECK: %[[RESULT:.*]] = scf.loop token(%[[OUTER_TOKEN:.*]]) iter_args(%[[OUTER_ARG:.*]] =
  %result = test.breakable_loop iter_args(%outer = %init) : i32 -> i32 {
    // CHECK: %[[MIDDLE_RESULT:.*]] = scf.loop token(%[[MIDDLE_TOKEN:.*]]) iter_args(%[[MIDDLE_ARG:.*]] =
    %middle_result = test.breakable_loop iter_args(%middle = %outer) : i32 -> i32 {
      // CHECK: scf.loop token(%[[INNER_TOKEN:.*]]) iter_args(%[[INNER_ARG:.*]] =
      test.breakable_loop iter_args(%inner = %inner_init) : f32 {
        // CHECK: %[[DEPTH_TWO:.*]] = arith.constant 2 : index
        // CHECK: %[[IS_TWO:.*]] = arith.cmpi eq, %[[DEPTH]], %[[DEPTH_TWO]] : index
        // CHECK: scf.if %[[IS_TWO]] {
        // CHECK:   scf.break [%[[MIDDLE_TOKEN]]] %[[MIDDLE_ARG]] : i32
        // CHECK: %[[DEPTH_THREE:.*]] = arith.constant 3 : index
        // CHECK: %[[IS_THREE:.*]] = arith.cmpi eq, %[[DEPTH]], %[[DEPTH_THREE]] : index
        // CHECK: scf.if %[[IS_THREE]] {
        // CHECK:   scf.break [%[[OUTER_TOKEN]]] %[[MIDDLE_ARG]] : i32
        // CHECK: scf.break [%[[MIDDLE_TOKEN]]] %[[MIDDLE_ARG]] : i32
        test.dynamic_break %depth %middle : i32
      }
      test.dynamic_break %c1 %middle : i32
    }
    test.dynamic_break %c1 %middle_result : i32
  }
  // CHECK: return %[[RESULT]] : i32
  return %result : i32
}
