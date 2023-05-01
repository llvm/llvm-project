// RUN: mlir-opt %s -split-input-file \
// RUN:   -test-one-to-n-type-conversion="convert-tuple-ops" \
// RUN: | FileCheck --check-prefix=CHECK-TUP %s

// RUN: mlir-opt %s -split-input-file \
// RUN:   -test-one-to-n-type-conversion="convert-func-ops" \
// RUN: | FileCheck --check-prefix=CHECK-FUNC %s

// RUN: mlir-opt %s -split-input-file \
// RUN:   -test-one-to-n-type-conversion="convert-func-ops convert-tuple-ops" \
// RUN: | FileCheck --check-prefix=CHECK-BOTH %s

// Test case: Matching nested packs and unpacks just disappear.

// CHECK-TUP-LABEL: func.func @pack_unpack(
// CHECK-TUP-SAME:                          %[[ARG0:.*]]: i1,
// CHECK-TUP-SAME:                          %[[ARG1:.*]]: i2) -> (i1, i2) {
// CHECK-TUP-DAG:     return %[[ARG0]], %[[ARG1]] : i1, i2
func.func @pack_unpack(%arg0: i1, %arg1: i2) -> (i1, i2) {
  %0 = "test.make_tuple"() : () -> tuple<>
  %1 = "test.make_tuple"(%arg1) : (i2) -> tuple<i2>
  %2 = "test.make_tuple"(%1) : (tuple<i2>) -> tuple<tuple<i2>>
  %3 = "test.make_tuple"(%0, %arg0, %2) : (tuple<>, i1, tuple<tuple<i2>>) -> tuple<tuple<>, i1, tuple<tuple<i2>>>
  %4 = "test.get_tuple_element"(%3) {index = 0 : i32} : (tuple<tuple<>, i1, tuple<tuple<i2>>>) -> tuple<>
  %5 = "test.get_tuple_element"(%3) {index = 1 : i32} : (tuple<tuple<>, i1, tuple<tuple<i2>>>) -> i1
  %6 = "test.get_tuple_element"(%3) {index = 2 : i32} : (tuple<tuple<>, i1, tuple<tuple<i2>>>) -> tuple<tuple<i2>>
  %7 = "test.get_tuple_element"(%6) {index = 0 : i32} : (tuple<tuple<i2>>) -> tuple<i2>
  %8 = "test.get_tuple_element"(%7) {index = 0 : i32} : (tuple<i2>) -> i2
  return %5, %8 : i1, i2
}

// -----

// Test case: Appropriate materializations are created depending on which ops
// are converted.

// If we only convert the tuple ops, the original `get_tuple_element` ops will
// disappear but one target materialization will be inserted from the
// unconverted function arguments to each of the return values (which have
// redundancy among themselves).
//
// CHECK-TUP-LABEL: func.func @materializations_tuple_args(
// CHECK-TUP-SAME:                                         %[[ARG0:.*]]: tuple<tuple<>, i1, tuple<tuple<i2>>>) -> (i1, i2) {
// CHECK-TUP-DAG:     %[[V0:.*]] = "test.get_tuple_element"(%[[ARG0]]) {index = 0 : i32} : (tuple<tuple<>, i1, tuple<tuple<i2>>>) -> tuple<>
// CHECK-TUP-DAG:     %[[V1:.*]] = "test.get_tuple_element"(%[[ARG0]]) {index = 1 : i32} : (tuple<tuple<>, i1, tuple<tuple<i2>>>) -> i1
// CHECK-TUP-DAG:     %[[V2:.*]] = "test.get_tuple_element"(%[[ARG0]]) {index = 2 : i32} : (tuple<tuple<>, i1, tuple<tuple<i2>>>) -> tuple<tuple<i2>>
// CHECK-TUP-DAG:     %[[V3:.*]] = "test.get_tuple_element"(%[[V2]]) {index = 0 : i32} : (tuple<tuple<i2>>) -> tuple<i2>
// CHECK-TUP-DAG:     %[[V4:.*]] = "test.get_tuple_element"(%[[V3]]) {index = 0 : i32} : (tuple<i2>) -> i2
// CHECK-TUP-DAG:     %[[V5:.*]] = "test.get_tuple_element"(%[[ARG0]]) {index = 0 : i32} : (tuple<tuple<>, i1, tuple<tuple<i2>>>) -> tuple<>
// CHECK-TUP-DAG:     %[[V6:.*]] = "test.get_tuple_element"(%[[ARG0]]) {index = 1 : i32} : (tuple<tuple<>, i1, tuple<tuple<i2>>>) -> i1
// CHECK-TUP-DAG:     %[[V7:.*]] = "test.get_tuple_element"(%[[ARG0]]) {index = 2 : i32} : (tuple<tuple<>, i1, tuple<tuple<i2>>>) -> tuple<tuple<i2>>
// CHECK-TUP-DAG:     %[[V8:.*]] = "test.get_tuple_element"(%[[V7]]) {index = 0 : i32} : (tuple<tuple<i2>>) -> tuple<i2>
// CHECK-TUP-DAG:     %[[V9:.*]] = "test.get_tuple_element"(%[[V8]]) {index = 0 : i32} : (tuple<i2>) -> i2
// CHECK-TUP-DAG:     return %[[V1]], %[[V9]] : i1, i2

// If we only convert the func ops, argument materializations are created from
// the converted tuple elements back to the tuples that the `get_tuple_element`
// ops expect.
//
// CHECK-FUNC-LABEL: func.func @materializations_tuple_args(
// CHECK-FUNC-SAME:                                         %[[ARG0:.*]]: i1,
// CHECK-FUNC-SAME:                                         %[[ARG1:.*]]: i2) -> (i1, i2) {
// CHECK-FUNC-DAG:     %[[V0:.*]] = "test.make_tuple"() : () -> tuple<>
// CHECK-FUNC-DAG:     %[[V1:.*]] = "test.make_tuple"(%[[ARG1]]) : (i2) -> tuple<i2>
// CHECK-FUNC-DAG:     %[[V2:.*]] = "test.make_tuple"(%[[V1]]) : (tuple<i2>) -> tuple<tuple<i2>>
// CHECK-FUNC-DAG:     %[[V3:.*]] = "test.make_tuple"(%[[V0]], %[[ARG0]], %[[V2]]) : (tuple<>, i1, tuple<tuple<i2>>) -> tuple<tuple<>, i1, tuple<tuple<i2>>>
// CHECK-FUNC-DAG:     %[[V4:.*]] = "test.get_tuple_element"(%[[V3]]) {index = 0 : i32} : (tuple<tuple<>, i1, tuple<tuple<i2>>>) -> tuple<>
// CHECK-FUNC-DAG:     %[[V5:.*]] = "test.get_tuple_element"(%[[V3]]) {index = 1 : i32} : (tuple<tuple<>, i1, tuple<tuple<i2>>>) -> i1
// CHECK-FUNC-DAG:     %[[V6:.*]] = "test.get_tuple_element"(%[[V3]]) {index = 2 : i32} : (tuple<tuple<>, i1, tuple<tuple<i2>>>) -> tuple<tuple<i2>>
// CHECK-FUNC-DAG:     %[[V7:.*]] = "test.get_tuple_element"(%[[V6]]) {index = 0 : i32} : (tuple<tuple<i2>>) -> tuple<i2>
// CHECK-FUNC-DAG:     %[[V8:.*]] = "test.get_tuple_element"(%[[V7]]) {index = 0 : i32} : (tuple<i2>) -> i2
// CHECK-FUNC-DAG:     return %[[V5]], %[[V8]] : i1, i2

// If we convert both tuple and func ops, basically everything disappears.
//
// CHECK-BOTH-LABEL: func.func @materializations_tuple_args(
// CHECK-BOTH-SAME:                                         %[[ARG0:.*]]: i1,
// CHECK-BOTH-SAME:                                         %[[ARG1:.*]]: i2) -> (i1, i2) {
// CHECK-BOTH-DAG:     return %[[ARG0]], %[[ARG1]] : i1, i2

func.func @materializations_tuple_args(%arg0: tuple<tuple<>, i1, tuple<tuple<i2>>>) -> (i1, i2) {
  %0 = "test.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tuple<>, i1, tuple<tuple<i2>>>) -> tuple<>
  %1 = "test.get_tuple_element"(%arg0) {index = 1 : i32} : (tuple<tuple<>, i1, tuple<tuple<i2>>>) -> i1
  %2 = "test.get_tuple_element"(%arg0) {index = 2 : i32} : (tuple<tuple<>, i1, tuple<tuple<i2>>>) -> tuple<tuple<i2>>
  %3 = "test.get_tuple_element"(%2) {index = 0 : i32} : (tuple<tuple<i2>>) -> tuple<i2>
  %4 = "test.get_tuple_element"(%3) {index = 0 : i32} : (tuple<i2>) -> i2
  return %1, %4 : i1, i2
}
// -----

// Test case: Appropriate materializations are created depending on which ops
// are converted.

// If we only convert the tuple ops, the original `make_tuple` ops will
// disappear but a source materialization will be inserted from the result of
// conversion (which, for `make_tuple`, are the original ops that get forwarded)
// to the operands of the unconverted op with the original type (i.e.,
// `return`).

// CHECK-TUP-LABEL: func.func @materializations_tuple_return(
// CHECK-TUP-SAME:                                           %[[ARG0:.*]]: i1,
// CHECK-TUP-SAME:                                           %[[ARG1:.*]]: i2) -> tuple<tuple<>, i1, tuple<tuple<i2>>> {
// CHECK-TUP-DAG:     %[[V0:.*]] = "test.make_tuple"() : () -> tuple<>
// CHECK-TUP-DAG:     %[[V1:.*]] = "test.make_tuple"(%[[ARG1]]) : (i2) -> tuple<i2>
// CHECK-TUP-DAG:     %[[V2:.*]] = "test.make_tuple"(%[[V1]]) : (tuple<i2>) -> tuple<tuple<i2>>
// CHECK-TUP-DAG:     %[[V3:.*]] = "test.make_tuple"(%[[V0]], %[[ARG0]], %[[V2]]) : (tuple<>, i1, tuple<tuple<i2>>) -> tuple<tuple<>, i1, tuple<tuple<i2>>>
// CHECK-TUP-DAG:     return %[[V3]] : tuple<tuple<>, i1, tuple<tuple<i2>>>

// If we only convert the func ops, target materializations are created from
// original tuples produced by `make_tuple` to its constituent elements that the
// converted op (i.e., `return`) expect.
//
// CHECK-FUNC-LABEL: func.func @materializations_tuple_return(
// CHECK-FUNC-SAME:                                           %[[ARG0:.*]]: i1,
// CHECK-FUNC-SAME:                                           %[[ARG1:.*]]: i2) -> (i1, i2) {
// CHECK-FUNC-DAG:     %[[V0:.*]] = "test.make_tuple"() : () -> tuple<>
// CHECK-FUNC-DAG:     %[[V1:.*]] = "test.make_tuple"(%[[ARG1]]) : (i2) -> tuple<i2>
// CHECK-FUNC-DAG:     %[[V2:.*]] = "test.make_tuple"(%[[V1]]) : (tuple<i2>) -> tuple<tuple<i2>>
// CHECK-FUNC-DAG:     %[[V3:.*]] = "test.make_tuple"(%[[V0]], %[[ARG0]], %[[V2]]) : (tuple<>, i1, tuple<tuple<i2>>) -> tuple<tuple<>, i1, tuple<tuple<i2>>>
// CHECK-FUNC-DAG:     %[[V4:.*]] = "test.get_tuple_element"(%[[V3]]) {index = 0 : i32} : (tuple<tuple<>, i1, tuple<tuple<i2>>>) -> tuple<>
// CHECK-FUNC-DAG:     %[[V5:.*]] = "test.get_tuple_element"(%[[V3]]) {index = 1 : i32} : (tuple<tuple<>, i1, tuple<tuple<i2>>>) -> i1
// CHECK-FUNC-DAG:     %[[V6:.*]] = "test.get_tuple_element"(%[[V3]]) {index = 2 : i32} : (tuple<tuple<>, i1, tuple<tuple<i2>>>) -> tuple<tuple<i2>>
// CHECK-FUNC-DAG:     %[[V7:.*]] = "test.get_tuple_element"(%[[V6]]) {index = 0 : i32} : (tuple<tuple<i2>>) -> tuple<i2>
// CHECK-FUNC-DAG:     %[[V8:.*]] = "test.get_tuple_element"(%[[V7]]) {index = 0 : i32} : (tuple<i2>) -> i2
// CHECK-FUNC-DAG:     return %[[V5]], %[[V8]] : i1, i2

// If we convert both tuple and func ops, basically everything disappears.
//
// CHECK-BOTH-LABEL: func.func @materializations_tuple_return(
// CHECK-BOTH-SAME:                                           %[[ARG0:.*]]: i1,
// CHECK-BOTH-SAME:                                           %[[ARG1:.*]]: i2) -> (i1, i2) {
// CHECK-BOTH-DAG:     return %[[ARG0]], %[[ARG1]] : i1, i2

func.func @materializations_tuple_return(%arg0: i1, %arg1: i2) -> tuple<tuple<>, i1, tuple<tuple<i2>>> {
  %0 = "test.make_tuple"() : () -> tuple<>
  %1 = "test.make_tuple"(%arg1) : (i2) -> tuple<i2>
  %2 = "test.make_tuple"(%1) : (tuple<i2>) -> tuple<tuple<i2>>
  %3 = "test.make_tuple"(%0, %arg0, %2) : (tuple<>, i1, tuple<tuple<i2>>) -> tuple<tuple<>, i1, tuple<tuple<i2>>>
  return %3 : tuple<tuple<>, i1, tuple<tuple<i2>>>
}
