// RUN: mlir-opt %s -loop-invariant-code-motion -split-input-file \
// RUN:   | FileCheck %s

// Unproven divisor => must NOT hoist.
// CHECK-LABEL: func.func @urem_no_hoist
// CHECK: scf.for
// CHECK:   llvm.urem
// CHECK:   scf.yield
func.func @urem_no_hoist(%a: i32, %b: i32, %n: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (i32) {
    %q = llvm.urem %a, %b : i32
    scf.yield %q : i32
  }
  return %r : i32
}

// -----

// Constant 7 => hoist is legal.
// CHECK-LABEL: func.func @urem_hoist_constant
// CHECK: %[[Q:.*]] = llvm.urem
// CHECK: scf.for
// CHECK: scf.yield %[[Q]]
func.func @urem_hoist_constant(%a: i32, %n: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d = arith.constant 7 : i32
  %init = arith.constant 0 : i32
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (i32) {
    %q = llvm.urem %a, %d : i32
    scf.yield %q : i32
  }
  return %r : i32
}

// -----

// Constant 0 => must NOT hoist.
// CHECK-LABEL: func.func @urem_no_hoist_zero
// CHECK: scf.for
// CHECK:   llvm.urem
// CHECK:   scf.yield
func.func @urem_no_hoist_zero(%a: i32, %n: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d = arith.constant 0 : i32
  %init = arith.constant 0 : i32
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (i32) {
    %q = llvm.urem %a, %d : i32
    scf.yield %q : i32
  }
  return %r : i32
}

// -----

// Unproven divisor => must NOT hoist.
// CHECK-LABEL: func.func @srem_no_hoist
// CHECK: scf.for
// CHECK:   llvm.srem
// CHECK:   scf.yield
func.func @srem_no_hoist(%a: i32, %b: i32, %n: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (i32) {
    %q = llvm.srem %a, %b : i32
    scf.yield %q : i32
  }
  return %r : i32
}

// -----

// Constant 7 => hoist is legal.
// CHECK-LABEL: func.func @srem_hoist_constant
// CHECK: %[[Q:.*]] = llvm.srem
// CHECK: scf.for
// CHECK: scf.yield %[[Q]]
func.func @srem_hoist_constant(%a: i32, %n: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d = arith.constant 7 : i32
  %init = arith.constant 0 : i32
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (i32) {
    %q = llvm.srem %a, %d : i32
    scf.yield %q : i32
  }
  return %r : i32
}

// -----

// Constant 0 => must NOT hoist.
// CHECK-LABEL: func.func @srem_no_hoist_zero
// CHECK: scf.for
// CHECK:   llvm.srem
// CHECK:   scf.yield
func.func @srem_no_hoist_zero(%a: i32, %n: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d = arith.constant 0 : i32
  %init = arith.constant 0 : i32
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (i32) {
    %q = llvm.srem %a, %d : i32
    scf.yield %q : i32
  }
  return %r : i32
}

// -----

// Constant -1 => must NOT hoist (INT_MIN / -1 overflow).
// CHECK-LABEL: func.func @srem_no_hoist_neg_one
// CHECK: scf.for
// CHECK:   llvm.srem
// CHECK:   scf.yield
func.func @srem_no_hoist_neg_one(%a: i32, %n: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d = arith.constant -1 : i32
  %init = arith.constant 0 : i32
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (i32) {
    %q = llvm.srem %a, %d : i32
    scf.yield %q : i32
  }
  return %r : i32
}
