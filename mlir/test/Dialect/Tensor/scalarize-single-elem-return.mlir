// RUN: mlir-opt -scalarize-single-element-tensor-return -split-input-file %s | FileCheck %s

/// Positive tests: functions with no users updated.

func.func private @rank0() -> tensor<i64> {
  %0 = arith.constant dense<-1> : tensor<i64>
  return %0 : tensor<i64>
}
/// Inserted ExtractOp gets folded for rank-0 tensors.
// CHECK-LABEL: func.func private @rank0
//  CHECK-SAME:     -> i64
//       CHECK:   %[[CST:.*]] = arith.constant -1 : i64
//       CHECK:   return %[[CST]] : i64

func.func private @rank1(%arg0: tensor<1xi64>) -> tensor<1xi64> {
  return %arg0 : tensor<1xi64>
}
// CHECK-LABEL: func.func private @rank1
// CHECK-SAME:      %[[SRC:.*]]: tensor<1xi64>) -> i64 {
//  CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//      CHECK:   %[[EXT:.*]] = tensor.extract %[[SRC]][%[[C0]]] : tensor<1xi64>
//      CHECK:   return %[[EXT]] : i64

func.func private @rank2_single_element(%arg0: tensor<1x1xi64>)
    -> tensor<1x1xi64> {
  return %arg0 : tensor<1x1xi64>
}
// CHECK-LABEL: func.func private @rank2_single_element
//  CHECK-SAME:     %[[SRC:.*]]: tensor<1x1xi64>) -> i64
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[EXT:.*]] = tensor.extract %[[SRC]][%[[C0]], %[[C0]]]
//  CHECK-SAME:     : tensor<1x1xi64>
//       CHECK:   return %[[EXT]] : i64

func.func private @non_return_terminator(%arg0: tensor<1xi64>, %cond: i1)
    -> tensor<1xi64> {
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  cf.br ^bb3(%arg0 : tensor<1xi64>)
^bb2:
  %0 = tensor.empty() : tensor<1xi64>
  cf.br ^bb3(%0 : tensor<1xi64>)
^bb3(%result: tensor<1xi64>):
  return %result : tensor<1xi64>
}
// CHECK-LABEL:   func.func private @non_return_terminator(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<1xi64>,
// CHECK-SAME:      %[[ARG1:.*]]: i1) -> i64 {
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           cf.cond_br %[[ARG1]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           cf.br ^bb3(%[[ARG0]] : tensor<1xi64>)
// CHECK:         ^bb2:
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<1xi64>
// CHECK:           cf.br ^bb3(%[[EMPTY_0]] : tensor<1xi64>)
// CHECK:         ^bb3(%[[VAL_0:.*]]: tensor<1xi64>):
// CHECK:           %[[EXT:.*]] = tensor.extract %[[VAL_0]][%[[C0]]]
// CHECK-SAME:        : tensor<1xi64>
// CHECK:           return %[[EXT]] : i64

/// Positive tests: both caller and callee updated.

func.func private @callee(%arg0: tensor<1xi64>) -> tensor<1xi64> {
  return %arg0 : tensor<1xi64>
}
// CHECK-LABEL: func.func private @callee
//  CHECK-SAME:     %[[SRC:.*]]: tensor<1xi64>) -> i64
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[EXT:.*]] = tensor.extract %[[SRC]][%[[C0]]] : tensor<1xi64>
//       CHECK:   return %[[EXT]] : i64

func.func private @caller(%arg0: tensor<1xi64>) -> tensor<1xi64> {
  %0 = call @callee(%arg0) : (tensor<1xi64>) -> tensor<1xi64>
  return %0 : tensor<1xi64>
}
// CHECK-LABEL: func.func private @caller
//  CHECK-SAME:     -> i64
//       CHECK:   %[[CALL:.*]] = call @callee(%arg0) : (tensor<1xi64>) -> i64
//       CHECK:   return %[[CALL]] : i64

// -----

/// Positive tests: only callee updated.

/// Private non-scalarizeable callers still allow rewritten callees.
/// The caller's signature remains unchanged.
func.func private @callee(%arg0: tensor<1xi64>) -> tensor<1xi64> {
  return %arg0 : tensor<1xi64>
}
// CHECK-LABEL: func.func private @callee
//  CHECK-SAME:     %[[SRC:.*]]: tensor<1xi64>) -> i64
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[EXT:.*]] = tensor.extract %[[SRC]][%[[C0]]] : tensor<1xi64>
//       CHECK:   return %[[EXT]] : i64

func.func private @non_scalarizable_caller_multiple_dimensions_return(
    %arg0: tensor<1xi64>) -> tensor<2xi64> {
  %0 = call @callee(%arg0) : (tensor<1xi64>) -> tensor<1xi64>
  %1 = tensor.empty() : tensor<2xi64>
  return %1 : tensor<2xi64>
}
// CHECK-LABEL: func.func private @non_scalarizable_caller_multiple_dimensions_return
//  CHECK-SAME:     -> tensor<2xi64>
//       CHECK:   %[[CALL:.*]] = call @callee(%arg0) : (tensor<1xi64>) -> i64
//       CHECK:   %[[EMPTY:.*]] = tensor.empty() : tensor<2xi64>
//       CHECK:   return %[[EMPTY]] : tensor<2xi64>

// -----

/// Positive tests: only callee updated, with callsite reboxing.
///
/// The private caller is not scalarizable because it has multiple results,
/// so the rewritten callee result is reboxed at the callsite.
func.func private @callee(%arg0: tensor<1xi64>) -> tensor<1xi64> {
  return %arg0 : tensor<1xi64>
}
// CHECK-LABEL: func.func private @callee
//  CHECK-SAME:     %[[SRC:.*]]: tensor<1xi64>) -> i64
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[EXT:.*]] = tensor.extract %[[SRC]][%[[C0]]] : tensor<1xi64>
//       CHECK:   return %[[EXT]] : i64

func.func private @non_scalarizeable_caller_with_multiple_returns(
    %arg0: tensor<1xi64>) -> (tensor<1xi64>, i64) {
  %0 = call @callee(%arg0) : (tensor<1xi64>) -> tensor<1xi64>
  %1 = arith.constant 7 : i64
  return %0, %1 : tensor<1xi64>, i64
}

// CHECK-LABEL: func.func private @non_scalarizeable_caller_with_multiple_returns
//  CHECK-SAME:     -> (tensor<1xi64>, i64)
//   CHECK-DAG:   %[[CST:.*]] = arith.constant 7 : i64
//   CHECK-DAG:   %[[CALL:.*]] = call @callee(%arg0) : (tensor<1xi64>) -> i64
//   CHECK-DAG:   %[[BOX:.*]] = tensor.from_elements %[[CALL]] : tensor<1xi64>
//       CHECK:   return %[[BOX]], %[[CST]] : tensor<1xi64>, i64

// -----

/// Positive tests: only callee updated, with callsite reboxing.
///
/// Some existing uses in the caller still expect the old tensor type.
func.func private @callee(%arg0: tensor<1xi64>) -> tensor<1xi64> {
  return %arg0 : tensor<1xi64>
}
// CHECK-LABEL: func.func private @callee
//  CHECK-SAME:     %[[SRC:.*]]: tensor<1xi64>) -> i64
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[EXT:.*]] = tensor.extract %[[SRC]][%[[C0]]] : tensor<1xi64>
//       CHECK:   return %[[EXT]] : i64

func.func private @non_scalarizeable_caller_expecting_tensor_type(
    %arg0: tensor<1xi64>, %arg1: tensor<1xi64>) -> (tensor<1xi64>) {
  %0 = call @callee(%arg0) : (tensor<1xi64>) -> tensor<1xi64>
  %init = tensor.empty() : tensor<1xi64>
  %1 = linalg.map { arith.addi } ins(%0, %arg1 : tensor<1xi64>, tensor<1xi64>)
    outs(%init : tensor<1xi64>)
  return %1 : tensor<1xi64>
}
// CHECK-LABEL: func.func private @non_scalarizeable_caller_expecting_tensor_type
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<1xi64>,
//  CHECK-SAME:     %[[ARG1:.*]]: tensor<1xi64>) -> i64
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[CALL:.*]] = call @callee(%[[ARG0]]) : (tensor<1xi64>) -> i64
//       CHECK:   %[[BOX:.*]] = tensor.from_elements %[[CALL]] : tensor<1xi64>
//       CHECK:   %[[EMPTY:.*]] = tensor.empty() : tensor<1xi64>
//       CHECK:   %[[MAP:.*]] = linalg.map { arith.addi
//       CHECK:     ins(%[[BOX]], %[[ARG1]] : tensor<1xi64>, tensor<1xi64>)
//       CHECK:     outs(%[[EMPTY]] : tensor<1xi64>)
//       CHECK:   %[[EXT:.*]] = tensor.extract %[[MAP]][%[[C0]]] : tensor<1xi64>
//       CHECK:   return %[[EXT]] : i64

// -----

/// Positive tests: callee updated for a direct call user that is not enclosed
/// by a func.func.

func.func private @callee(%arg0: tensor<1xi64>) -> tensor<1xi64> {
  return %arg0 : tensor<1xi64>
}
// CHECK-LABEL: func.func private @callee
// CHECK-SAME:      %[[SRC:.*]]: tensor<1xi64>) -> i64
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[EXT:.*]] = tensor.extract %[[SRC]][%[[C0]]] : tensor<1xi64>
//       CHECK:   return %[[EXT]] : i64

%0 = tensor.empty() : tensor<1xi64>
%1 = func.call @callee(%0) : (tensor<1xi64>) -> tensor<1xi64>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1xi64>
// CHECK: %[[CALL:.*]] = func.call @callee(%[[EMPTY]]) : (tensor<1xi64>) -> i64

// -----

/// Positive tests: only caller updated.

/// `func.constant` is a non-call symbol user of the target, so it blocks
/// scalarization.
func.func private @callee(%arg0: tensor<1xi64>) -> tensor<1xi64> {
  return %arg0 : tensor<1xi64>
}
// CHECK-LABEL: func.func private @callee
//  CHECK-SAME:     -> tensor<1xi64>
//   CHECK-NOT:   tensor.extract
//       CHECK:   return %arg0 : tensor<1xi64>

func.func private @caller_using_constant(%arg0: tensor<1xi64>) -> tensor<1xi64> {
  %fn = func.constant @callee : (tensor<1xi64>) -> tensor<1xi64>
  %result = func.call_indirect %fn(%arg0) : (tensor<1xi64>) -> tensor<1xi64>
  return %result : tensor<1xi64>
}
// CHECK-LABEL: func.func private @caller_using_constant
// CHECK-SAME:      %[[SRC:.*]]: tensor<1xi64>) -> i64
//  CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//      CHECK:   %[[CONST:.*]] = constant @callee : (tensor<1xi64>) -> tensor<1xi64>
//      CHECK:   %[[CALL:.*]] = call_indirect %[[CONST]](%[[SRC]])
// CHECK-SAME:     : (tensor<1xi64>) -> tensor<1xi64>
//      CHECK:   %[[EXT:.*]] = tensor.extract %[[CALL]][%[[C0]]] : tensor<1xi64>
//      CHECK:   return %[[EXT]] : i64

// -----

/// Negative tests: neither callee nor caller updated.

func.func private @multiple_elements(%arg0: tensor<2xi64>) -> tensor<2xi64> {
  return %arg0 : tensor<2xi64>
}
// CHECK-LABEL: func.func private @multiple_elements
//  CHECK-SAME:     -> tensor<2xi64>
//   CHECK-NOT:   tensor.extract
//       CHECK:   return %arg0 : tensor<2xi64>

func.func private @dynamic_shape(%arg0: tensor<?xi64>) -> tensor<?xi64> {
  return %arg0 : tensor<?xi64>
}
// CHECK-LABEL: func.func private @dynamic_shape
//  CHECK-SAME:     -> tensor<?xi64>
//   CHECK-NOT:   tensor.extract
//       CHECK:   return %arg0 : tensor<?xi64>

func.func private @unranked(%arg0: tensor<*xi64>) -> tensor<*xi64> {
  return %arg0 : tensor<*xi64>
}
// CHECK-LABEL: func.func private @unranked
//  CHECK-SAME:     -> tensor<*xi64>
//   CHECK-NOT:   tensor.extract
//       CHECK:   return %arg0 : tensor<*xi64>

func.func @public_function(%arg0: tensor<1xi64>) -> tensor<1xi64> {
  return %arg0 : tensor<1xi64>
}
// CHECK-LABEL: func.func @public_function
//  CHECK-SAME:     -> tensor<1xi64>
//   CHECK-NOT:   tensor.extract
//       CHECK:   return %arg0 : tensor<1xi64>

func.func private @callee(%arg0: tensor<1xi64>) -> tensor<1xi64> {
  return %arg0 : tensor<1xi64>
}
// CHECK-LABEL: func.func private @callee
//  CHECK-SAME:     -> tensor<1xi64>
//   CHECK-NOT:   tensor.extract
//       CHECK:   return %arg0 : tensor<1xi64>

func.func @public_caller(%arg0: tensor<1xi64>) -> tensor<1xi64> {
  %0 = call @callee(%arg0) : (tensor<1xi64>) -> tensor<1xi64>
  return %0 : tensor<1xi64>
}
// CHECK-LABEL: func.func @public_caller
//  CHECK-SAME:     -> tensor<1xi64>
//       CHECK:   %[[CALL:.*]] = call @callee(%arg0) : (tensor<1xi64>) -> tensor<1xi64>
//   CHECK-NOT:   tensor.extract
//       CHECK:   return %[[CALL]] : tensor<1xi64>

// -----

/// Recursive cycle - the DFS marks the cycle conservatively as blocked,
/// so neither function is scalarized.

func.func private @recursive_a(%arg0: tensor<1xi64>) -> tensor<1xi64> {
  %0 = call @recursive_b(%arg0) : (tensor<1xi64>) -> tensor<1xi64>
  return %0 : tensor<1xi64>
}
// CHECK-LABEL: func.func private @recursive_a
//  CHECK-SAME:     -> tensor<1xi64>
//       CHECK:   %[[CALL:.*]] = call @recursive_b(%arg0) : (tensor<1xi64>) -> tensor<1xi64>
//       CHECK:   return %[[CALL]] : tensor<1xi64>

func.func private @recursive_b(%arg0: tensor<1xi64>) -> tensor<1xi64> {
  %0 = call @recursive_a(%arg0) : (tensor<1xi64>) -> tensor<1xi64>
  return %0 : tensor<1xi64>
}
// CHECK-LABEL: func.func private @recursive_b
//  CHECK-SAME:     -> tensor<1xi64>
//       CHECK:   %[[CALL:.*]] = call @recursive_a(%arg0) : (tensor<1xi64>) -> tensor<1xi64>
//       CHECK:   return %[[CALL]] : tensor<1xi64>
