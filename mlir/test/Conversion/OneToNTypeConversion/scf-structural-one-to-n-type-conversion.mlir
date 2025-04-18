// RUN: mlir-opt %s -split-input-file \
// RUN:   -test-one-to-n-type-conversion="convert-func-ops convert-scf-ops" \
// RUN: | FileCheck %s

// Test case: Nested 1:N type conversion is carried through scf.if and
// scf.yield.

// CHECK-LABEL: func.func @if_result(
// CHECK-SAME:                       %[[ARG0:.*]]: i1,
// CHECK-SAME:                       %[[ARG1:.*]]: i2,
// CHECK-SAME:                       %[[ARG2:.*]]: i1) -> (i1, i2) {
// CHECK-NEXT:    %[[V0:.*]]:2 = scf.if %[[ARG2]] -> (i1, i2) {
// CHECK-NEXT:     scf.yield %[[ARG0]], %[[ARG1]] : i1, i2
// CHECK-NEXT:   } else {
// CHECK-NEXT:     scf.yield %[[ARG0]], %[[ARG1]] : i1, i2
// CHECK-NEXT:   }
// CHECK-NEXT:   return %[[V0]]#0, %[[V0]]#1 : i1, i2
func.func @if_result(%arg0: tuple<tuple<>, i1, tuple<i2>>, %arg1: i1) -> tuple<tuple<>, i1, tuple<i2>> {
  %0 = scf.if %arg1 -> (tuple<tuple<>, i1, tuple<i2>>) {
    scf.yield %arg0 : tuple<tuple<>, i1, tuple<i2>>
  } else {
    scf.yield %arg0 : tuple<tuple<>, i1, tuple<i2>>
  }
  return %0 : tuple<tuple<>, i1, tuple<i2>>
}

// -----

// Test case: Nested 1:N type conversion is carried through scf.if and
// scf.yield and unconverted ops inside have proper materializations.

// CHECK-LABEL: func.func @if_tuple_ops(
// CHECK-SAME:                          %[[ARG0:.*]]: i1,
// CHECK-SAME:                          %[[ARG1:.*]]: i1) -> i1 {
// CHECK-NEXT:    %[[V0:.*]] = "test.make_tuple"() : () -> tuple<>
// CHECK-NEXT:    %[[V1:.*]] = "test.make_tuple"(%[[V0]], %[[ARG0]]) : (tuple<>, i1) -> tuple<tuple<>, i1>
// CHECK-NEXT:    %[[V2:.*]] = scf.if %[[ARG1]] -> (i1) {
// CHECK-NEXT:      %[[V3:.*]] = "test.op"(%[[V1]]) : (tuple<tuple<>, i1>) -> tuple<tuple<>, i1>
// CHECK-NEXT:      %[[V4:.*]] = "test.get_tuple_element"(%[[V3]]) <{index = 0 : i32}> : (tuple<tuple<>, i1>) -> tuple<>
// CHECK-NEXT:      %[[V5:.*]] = "test.get_tuple_element"(%[[V3]]) <{index = 1 : i32}> : (tuple<tuple<>, i1>) -> i1
// CHECK-NEXT:      scf.yield %[[V5]] : i1
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %[[V6:.*]] = "test.source"() : () -> tuple<tuple<>, i1>
// CHECK-NEXT:      %[[V7:.*]] = "test.get_tuple_element"(%[[V6]]) <{index = 0 : i32}> : (tuple<tuple<>, i1>) -> tuple<>
// CHECK-NEXT:      %[[V8:.*]] = "test.get_tuple_element"(%[[V6]]) <{index = 1 : i32}> : (tuple<tuple<>, i1>) -> i1
// CHECK-NEXT:      scf.yield %[[V8]] : i1
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[V2]] : i1
func.func @if_tuple_ops(%arg0: tuple<tuple<>, i1>, %arg1: i1) -> tuple<tuple<>, i1> {
  %0 = scf.if %arg1 -> (tuple<tuple<>, i1>) {
    %1 = "test.op"(%arg0) : (tuple<tuple<>, i1>) -> tuple<tuple<>, i1>
    scf.yield %1 : tuple<tuple<>, i1>
  } else {
    %1 = "test.source"() : () -> tuple<tuple<>, i1>
    scf.yield %1 : tuple<tuple<>, i1>
  }
  return %0 : tuple<tuple<>, i1>
}
// -----

// Test case: Nested 1:N type conversion is carried through scf.while,
// scf.condition, and scf.yield.

// CHECK-LABEL: func.func @while_operands_results(
// CHECK-SAME:                                    %[[ARG0:.*]]: i1,
// CHECK-SAME:                                    %[[ARG1:.*]]: i2,
// CHECK-SAME:                                    %[[ARG2:.*]]: i1) -> (i1, i2) {
//   %[[V0:.*]]:2 = scf.while (%[[ARG3:.*]] = %[[ARG0]], %[[ARG4:.*]] = %[[ARG1]]) : (i1, i2) -> (i1, i2) {
//     scf.condition(%arg2) %[[ARG3]], %[[ARG4]] : i1, i2
//   } do {
//   ^bb0(%[[ARG5:.*]]: i1, %[[ARG6:.*]]: i2):
//     scf.yield %[[ARG5]], %[[ARG4]] : i1, i2
//   }
//   return %[[V0]]#0, %[[V0]]#1 : i1, i2
func.func @while_operands_results(%arg0: tuple<tuple<>, i1, tuple<i2>>, %arg1: i1) -> tuple<tuple<>, i1, tuple<i2>> {
  %0 = scf.while (%arg2 = %arg0) : (tuple<tuple<>, i1, tuple<i2>>) -> tuple<tuple<>, i1, tuple<i2>> {
    scf.condition(%arg1) %arg2 : tuple<tuple<>, i1, tuple<i2>>
  } do {
  ^bb0(%arg2: tuple<tuple<>, i1, tuple<i2>>):
    scf.yield %arg2 : tuple<tuple<>, i1, tuple<i2>>
  }
  return %0 : tuple<tuple<>, i1, tuple<i2>>
}

// -----

// Test case: Nested 1:N type conversion is carried through scf.while,
// scf.condition, and unconverted ops inside have proper materializations.

// CHECK-LABEL: func.func @while_tuple_ops(
// CHECK-SAME:                             %[[ARG0:.*]]: i1,
// CHECK-SAME:                             %[[ARG1:.*]]: i1) -> i1 {
// CHECK-NEXT:    %[[V0:.*]] = scf.while (%[[ARG2:.*]] = %[[ARG0]]) : (i1) -> i1 {
// CHECK-NEXT:      %[[V1:.*]] = "test.make_tuple"() : () -> tuple<>
// CHECK-NEXT:      %[[V2:.*]] = "test.make_tuple"(%[[V1]], %[[ARG2]]) : (tuple<>, i1) -> tuple<tuple<>, i1>
// CHECK-NEXT:      %[[V3:.*]] = "test.op"(%[[V2]]) : (tuple<tuple<>, i1>) -> tuple<tuple<>, i1>
// CHECK-NEXT:      %[[V4:.*]] = "test.get_tuple_element"(%[[V3]]) <{index = 0 : i32}> : (tuple<tuple<>, i1>) -> tuple<>
// CHECK-NEXT:      %[[V5:.*]] = "test.get_tuple_element"(%[[V3]]) <{index = 1 : i32}> : (tuple<tuple<>, i1>) -> i1
// CHECK-NEXT:      scf.condition(%[[ARG1]]) %[[V5]] : i1
// CHECK-NEXT:    } do {
// CHECK-NEXT:    ^bb0(%[[ARG3:.*]]: i1):
// CHECK-NEXT:      %[[V6:.*]] = "test.source"() : () -> tuple<tuple<>, i1>
// CHECK-NEXT:      %[[V7:.*]] = "test.get_tuple_element"(%[[V6]]) <{index = 0 : i32}> : (tuple<tuple<>, i1>) -> tuple<>
// CHECK-NEXT:      %[[V8:.*]] = "test.get_tuple_element"(%[[V6]]) <{index = 1 : i32}> : (tuple<tuple<>, i1>) -> i1
// CHECK-NEXT:      scf.yield %[[V8]] : i1
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[V0]] : i1
func.func @while_tuple_ops(%arg0: tuple<tuple<>, i1>, %arg1: i1) -> tuple<tuple<>, i1> {
  %0 = scf.while (%arg2 = %arg0) : (tuple<tuple<>, i1>) -> tuple<tuple<>, i1> {
    %1 = "test.op"(%arg2) : (tuple<tuple<>, i1>) -> tuple<tuple<>, i1>
    scf.condition(%arg1) %1 : tuple<tuple<>, i1>
  } do {
  ^bb0(%arg2: tuple<tuple<>, i1>):
    %1 = "test.source"() : () -> tuple<tuple<>, i1>
    scf.yield %1 : tuple<tuple<>, i1>
  }
  return %0 : tuple<tuple<>, i1>
}

// -----

// Test case: Nested 1:N type conversion is carried through scf.for and scf.yield.

// CHECK-LABEL: func.func @for_operands_results(
// CHECK-SAME:                                    %[[ARG0:.*]]: i1,
// CHECK-SAME:                                    %[[ARG1:.*]]: i2) -> (i1, i2) {
// CHECK-NEXT:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT:    %[[C10:.+]] = arith.constant 10 : index
// CHECK-NEXT:    %[[OUT:.+]]:2 = scf.for %arg2 = %[[C0]] to %[[C10]] step %[[C1]] iter_args(%[[ITER0:.+]] = %[[ARG0]], %[[ITER1:.+]] = %[[ARG1]]) -> (i1, i2) {
// CHECK-NEXT:     scf.yield %[[ITER0]], %[[ITER1]] : i1, i2
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[OUT]]#0, %[[OUT]]#1 : i1, i2

func.func @for_operands_results(%arg0: tuple<tuple<>, i1, tuple<i2>>) -> tuple<tuple<>, i1, tuple<i2>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  %0 = scf.for %i = %c0 to %c10 step %c1 iter_args(%acc = %arg0) -> tuple<tuple<>, i1, tuple<i2>> {
    scf.yield %acc : tuple<tuple<>, i1, tuple<i2>>
  }

  return %0 : tuple<tuple<>, i1, tuple<i2>>
}

// -----

// Test case: Nested 1:N type conversion is carried through scf.for and scf.yield

// CHECK-LABEL: func.func @for_tuple_ops(
// CHECK-SAME:                             %[[ARG0:.+]]: i1) -> i1 {
// CHECK-NEXT: %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT: %[[C10:.+]] = arith.constant 10 : index
// CHECK-NEXT: %[[FOR:.+]] = scf.for %arg1 = %[[C0]] to %[[C10]] step %[[C1]] iter_args(%[[ITER:.+]] = %[[ARG0]]) -> (i1) {
// CHECK-NEXT:   %[[V1:.+]] = "test.make_tuple"() : () -> tuple<>
// CHECK-NEXT:   %[[V2:.+]] = "test.make_tuple"(%[[V1]], %[[ITER]]) : (tuple<>, i1) -> tuple<tuple<>, i1>
// CHECK-NEXT:   %[[V3:.+]] = "test.op"(%[[V2]]) : (tuple<tuple<>, i1>) -> tuple<tuple<>, i1>
// CHECK-NEXT:   %[[V4:.+]] = "test.get_tuple_element"(%[[V3]]) <{index = 0 : i32}> : (tuple<tuple<>, i1>) -> tuple<>
// CHECK-NEXT:   %[[V5:.+]] = "test.get_tuple_element"(%[[V3]]) <{index = 1 : i32}> : (tuple<tuple<>, i1>) -> i1
// CHECK-NEXT:   scf.yield %[[V5]] : i1
// CHECK-NEXT: }
// CHECK-NEXT: %[[V6:.+]] = "test.make_tuple"() : () -> tuple<>
// CHECK-NEXT: %[[V7:.+]] = "test.make_tuple"(%[[V6]], %[[FOR]]) : (tuple<>, i1) -> tuple<tuple<>, i1>
// CHECK-NEXT: %[[V8:.+]] = "test.op"(%[[V7]]) : (tuple<tuple<>, i1>) -> tuple<tuple<>, i1>
// CHECK-NEXT: %[[V9:.+]] = "test.get_tuple_element"(%[[V8]]) <{index = 0 : i32}> : (tuple<tuple<>, i1>) -> tuple<>
// CHECK-NEXT: %[[V10:.+]] = "test.get_tuple_element"(%[[V8]]) <{index = 1 : i32}> : (tuple<tuple<>, i1>) -> i1
// CHECK-NEXT: return %[[V10]] : i1

func.func @for_tuple_ops(%arg0: tuple<tuple<>, i1>) -> tuple<tuple<>, i1> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  %0 = scf.for %i = %c0 to %c10 step %c1 iter_args(%acc = %arg0) -> tuple<tuple<>, i1> {
    %1 = "test.op"(%acc) : (tuple<tuple<>, i1>) -> tuple<tuple<>, i1>
    scf.yield %1 : tuple<tuple<>, i1>
  } 

  %1 = "test.op"(%0) : (tuple<tuple<>, i1>) -> tuple<tuple<>, i1>
  return %1 : tuple<tuple<>, i1>
}
