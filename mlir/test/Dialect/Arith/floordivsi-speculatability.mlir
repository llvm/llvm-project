// RUN: mlir-opt %s -loop-invariant-code-motion -split-input-file | FileCheck %s

// ===========================================================================
// Unproven divisor => must NOT hoist.
// ===========================================================================
// CHECK-LABEL: func.func @no_hoist_unproven
// CHECK: scf.for
// CHECK:   arith.floordivsi
// CHECK:   scf.yield
func.func @no_hoist_unproven(%a: i32, %b: i32, %n: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (i32) {
    %q = arith.floordivsi %a, %b : i32
    scf.yield %q : i32
  }
  return %r : i32
}

// -----

// ===========================================================================
// Constant 7 (non-zero, not -1) => hoist is legal.
// Do not use CHECK-NEXT here: LICM also hoists the init constant, so the
// floordivsi is not guaranteed to be immediately followed by scf.for.
// ===========================================================================
// CHECK-LABEL: func.func @hoist_constant_safe
// CHECK: %[[Q:.*]] = arith.floordivsi
// CHECK: scf.for
// CHECK: scf.yield %[[Q]]
func.func @hoist_constant_safe(%a: i32, %n: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d = arith.constant 7 : i32
  %init = arith.constant 0 : i32
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (i32) {
    %q = arith.floordivsi %a, %d : i32
    scf.yield %q : i32
  }
  return %r : i32
}

// -----

// ===========================================================================
// Constant 0 => must NOT hoist.
// ===========================================================================
// CHECK-LABEL: func.func @no_hoist_zero
// CHECK: scf.for
// CHECK:   arith.floordivsi
// CHECK:   scf.yield
func.func @no_hoist_zero(%a: i32, %n: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d = arith.constant 0 : i32
  %init = arith.constant 0 : i32
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (i32) {
    %q = arith.floordivsi %a, %d : i32
    scf.yield %q : i32
  }
  return %r : i32
}

// -----

// ===========================================================================
// Constant -1 => must NOT hoist (INT_MIN / -1 overflow).
// ===========================================================================
// CHECK-LABEL: func.func @no_hoist_neg_one
// CHECK: scf.for
// CHECK:   arith.floordivsi
// CHECK:   scf.yield
func.func @no_hoist_neg_one(%a: i32, %n: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d = arith.constant -1 : i32
  %init = arith.constant 0 : i32
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (i32) {
    %q = arith.floordivsi %a, %d : i32
    scf.yield %q : i32
  }
  return %r : i32
}

// -----

// ===========================================================================
// Divisor = addi 5, 3. The m_IntRangeWithoutZeroS / WithoutNegOneS matchers
// only inspect the divisor's defining op, not a dataflow analysis. Without
// a running DataFlowSolver, addi's range is unknown, so LICM must not hoist.
// ===========================================================================
// CHECK-LABEL: func.func @no_hoist_addi_divisor
// CHECK: scf.for
// CHECK:   arith.floordivsi
// CHECK:   scf.yield
func.func @no_hoist_addi_divisor(%a: i32, %n: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : i32
  %c3 = arith.constant 3 : i32
  %d  = arith.addi %c5, %c3 : i32
  %init = arith.constant 0 : i32
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (i32) {
    %q = arith.floordivsi %a, %d : i32
    scf.yield %q : i32
  }
  return %r : i32
}
