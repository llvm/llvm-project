// RUN: mlir-opt -allow-unregistered-dialect %s -sccp -split-input-file | FileCheck %s
// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline="builtin.module(builtin.module(sccp))" -split-input-file | FileCheck %s --check-prefix=NESTED
// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline="builtin.module(func.func(sccp))" -split-input-file | FileCheck %s --check-prefix=FUNC

/// Check that a constant is properly propagated through the arguments and
/// results of a private function.

// CHECK-LABEL: func private @private(
func.func private @private(%arg0 : i32) -> i32 {
  // CHECK: %[[CST:.*]] = arith.constant 1 : i32
  // CHECK: return %[[CST]] : i32

  return %arg0 : i32
}

// CHECK-LABEL: func @simple_private(
func.func @simple_private() -> i32 {
  // CHECK: %[[CST:.*]] = arith.constant 1 : i32
  // CHECK: return %[[CST]] : i32

  %1 = arith.constant 1 : i32
  %result = call @private(%1) : (i32) -> i32
  return %result : i32
}

// -----

/// Check that a constant is properly propagated through the arguments and
/// results of a visible nested function.

// CHECK: func nested @nested(
func.func nested @nested(%arg0 : i32) -> i32 {
  // CHECK: %[[CST:.*]] = arith.constant 1 : i32
  // CHECK: return %[[CST]] : i32

  return %arg0 : i32
}

// CHECK-LABEL: func @simple_nested(
func.func @simple_nested() -> i32 {
  // CHECK: %[[CST:.*]] = arith.constant 1 : i32
  // CHECK: return %[[CST]] : i32

  %1 = arith.constant 1 : i32
  %result = call @nested(%1) : (i32) -> i32
  return %result : i32
}

// -----

/// Check that non-visible nested functions do not track arguments.
module {
  // NESTED-LABEL: module @nested_module
  module @nested_module attributes { sym_visibility = "public" } {

    // NESTED: func nested @nested(
    func.func nested @nested(%arg0 : i32) -> (i32, i32) {
      // NESTED: %[[CST:.*]] = arith.constant 1 : i32
      // NESTED: return %[[CST]], %arg0 : i32, i32

      %1 = arith.constant 1 : i32
      return %1, %arg0 : i32, i32
    }

    // NESTED: func @nested_not_all_uses_visible(
    func.func @nested_not_all_uses_visible() -> (i32, i32) {
      // NESTED: %[[CST:.*]] = arith.constant 1 : i32
      // NESTED: %[[CALL:.*]]:2 = call @nested
      // NESTED: return %[[CST]], %[[CALL]]#1 : i32, i32

      %1 = arith.constant 1 : i32
      %result:2 = call @nested(%1) : (i32) -> (i32, i32)
      return %result#0, %result#1 : i32, i32
    }
  }
}

// -----

/// Check that public functions do not track arguments.

// CHECK-LABEL: func @public(
func.func @public(%arg0 : i32) -> (i32, i32) {
  %1 = arith.constant 1 : i32
  return %1, %arg0 : i32, i32
}

// CHECK-LABEL: func @simple_public(
func.func @simple_public() -> (i32, i32) {
  // CHECK: %[[CST:.*]] = arith.constant 1 : i32
  // CHECK: %[[CALL:.*]]:2 = call @public
  // CHECK: return %[[CST]], %[[CALL]]#1 : i32, i32

  %1 = arith.constant 1 : i32
  %result:2 = call @public(%1) : (i32) -> (i32, i32)
  return %result#0, %result#1 : i32, i32
}

// -----

/// Check that functions with non-call users don't have arguments tracked.

func.func private @callable(%arg0 : i32) -> (i32, i32) {
  %1 = arith.constant 1 : i32
  return %1, %arg0 : i32, i32
}

// CHECK-LABEL: func @non_call_users(
func.func @non_call_users() -> (i32, i32) {
  // CHECK: %[[CST:.*]] = arith.constant 1 : i32
  // CHECK: %[[CALL:.*]]:2 = call @callable
  // CHECK: return %[[CST]], %[[CALL]]#1 : i32, i32

  %1 = arith.constant 1 : i32
  %result:2 = call @callable(%1) : (i32) -> (i32, i32)
  return %result#0, %result#1 : i32, i32
}

"live.user"() {uses = [@callable]} : () -> ()

// -----

/// Check that return values are overdefined in the presence of an unknown terminator.

func.func private @callable(%arg0 : i32) -> i32 {
  "unknown.return"(%arg0) : (i32) -> ()
}

// CHECK-LABEL: func @unknown_terminator(
func.func @unknown_terminator() -> i32 {
  // CHECK: %[[CALL:.*]] = call @callable
  // CHECK: return %[[CALL]] : i32

  %1 = arith.constant 1 : i32
  %result = call @callable(%1) : (i32) -> i32
  return %result : i32
}

// -----

/// Check that return values are overdefined when the constant conflicts.

func.func private @callable(%arg0 : i32) -> i32 {
  return %arg0 : i32
}

// CHECK-LABEL: func @conflicting_constant(
func.func @conflicting_constant() -> (i32, i32) {
  // CHECK: %[[CALL1:.*]] = call @callable
  // CHECK: %[[CALL2:.*]] = call @callable
  // CHECK: return %[[CALL1]], %[[CALL2]] : i32, i32

  %1 = arith.constant 1 : i32
  %2 = arith.constant 2 : i32
  %result = call @callable(%1) : (i32) -> i32
  %result2 = call @callable(%2) : (i32) -> i32
  return %result, %result2 : i32, i32
}

// -----

/// Check that return values are overdefined when the constant conflicts with a
/// non-constant.

func.func private @callable(%arg0 : i32) -> i32 {
  "unknown.return"(%arg0) : (i32) -> ()
}

// CHECK-LABEL: func @conflicting_constant(
func.func @conflicting_constant(%arg0 : i32) -> (i32, i32) {
  // CHECK: %[[CALL1:.*]] = call @callable
  // CHECK: %[[CALL2:.*]] = call @callable
  // CHECK: return %[[CALL1]], %[[CALL2]] : i32, i32

  %1 = arith.constant 1 : i32
  %result = call @callable(%1) : (i32) -> i32
  %result2 = call @callable(%arg0) : (i32) -> i32
  return %result, %result2 : i32, i32
}

// -----

/// Check a more complex interaction with calls and control flow.

// CHECK-LABEL: func private @complex_inner_if(
func.func private @complex_inner_if(%arg0 : i32) -> i32 {
  // CHECK-DAG: %[[TRUE:.*]] = arith.constant true
  // CHECK-DAG: %[[CST:.*]] = arith.constant 1 : i32
  // CHECK: cf.cond_br %[[TRUE]], ^bb1

  %cst_20 = arith.constant 20 : i32
  %cond = arith.cmpi ult, %arg0, %cst_20 : i32
  cf.cond_br %cond, ^bb1, ^bb2

^bb1:
  // CHECK: ^bb1:
  // CHECK: return %[[CST]] : i32

  %cst_1 = arith.constant 1 : i32
  return %cst_1 : i32

^bb2:
  %cst_1_2 = arith.constant 1 : i32
  %arg_inc = arith.addi %arg0, %cst_1_2 : i32
  return %arg_inc : i32
}

func.func private @complex_cond() -> i1

// CHECK-LABEL: func private @complex_callee(
func.func private @complex_callee(%arg0 : i32) -> i32 {
  // CHECK: %[[CST:.*]] = arith.constant 1 : i32

  %loop_cond = call @complex_cond() : () -> i1
  cf.cond_br %loop_cond, ^bb1, ^bb2

^bb1:
  // CHECK: ^bb1:
  // CHECK-NEXT: return %[[CST]] : i32
  return %arg0 : i32

^bb2:
  // CHECK: ^bb2:
  // CHECK: call @complex_inner_if(%[[CST]]) : (i32) -> i32
  // CHECK: call @complex_callee(%[[CST]]) : (i32) -> i32
  // CHECK: return %[[CST]] : i32

  %updated_arg = call @complex_inner_if(%arg0) : (i32) -> i32
  %res = call @complex_callee(%updated_arg) : (i32) -> i32
  return %res : i32
}

// CHECK-LABEL: func @complex_caller(
func.func @complex_caller(%arg0 : i32) -> i32 {
  // CHECK: %[[CST:.*]] = arith.constant 1 : i32
  // CHECK: return %[[CST]] : i32

  %1 = arith.constant 1 : i32
  %result = call @complex_callee(%1) : (i32) -> i32
  return %result : i32
}

// -----

/// Check that non-symbol defining callables currently go to overdefined.

// CHECK-LABEL: func @non_symbol_defining_callable
func.func @non_symbol_defining_callable() -> i32 {
  // CHECK: %[[RES:.*]] = call_indirect
  // CHECK: return %[[RES]] : i32

  %fn = "test.functional_region_op"() ({
    %1 = arith.constant 1 : i32
    "test.return"(%1) : (i32) -> ()
  }) : () -> (() -> i32)
  %res = call_indirect %fn() : () -> (i32)
  return %res : i32
}

// -----

/// Check that private callables don't get processed if they have no uses.

// CHECK-LABEL: func private @unreferenced_private_function
func.func private @unreferenced_private_function() -> i32 {
  // CHECK: %[[RES:.*]] = arith.select
  // CHECK: return %[[RES]] : i32
  %true = arith.constant true
  %cst0 = arith.constant 0 : i32
  %cst1 = arith.constant 1 : i32
  %result = arith.select %true, %cst0, %cst1 : i32
  return %result : i32
}

// -----

/// Check that callables outside the analysis scope are marked as external.

func.func private @foo() -> index {
  %0 = arith.constant 10 : index
  return %0 : index
}

// CHECK-LABEL: func @bar
// FUNC-LABEL: func @bar
func.func @bar(%arg0: index) -> index {
  // CHECK: %[[C10:.*]] = arith.constant 10
  %c0 = arith.constant 0 : index
  %1 = arith.constant 420 : index
  %7 = arith.cmpi eq, %arg0, %c0 : index
  cf.cond_br %7, ^bb1(%1 : index), ^bb2

// CHECK: ^bb1(%[[ARG:.*]]: index):
// FUNC: ^bb1(%[[ARG:.*]]: index):
^bb1(%8: index):  // 2 preds: ^bb0, ^bb4
  // CHECK-NEXT: return %[[ARG]]
  // FUNC-NEXT: return %[[ARG]]
  return %8 : index

// CHECK: ^bb2
// FUNC: ^bb2
^bb2:
  // FUNC-NEXT: %[[FOO:.*]] = call @foo
  %13 = call @foo() : () -> index
  // CHECK: cf.br ^bb1(%[[C10]]
  // FUNC: cf.br ^bb1(%[[FOO]]
  cf.br ^bb1(%13 : index)
}
