// RUN: mlir-opt %s --transform-interpreter -allow-unregistered-dialect --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @basic_cast_and_call
func.func @basic_cast_and_call() {
  // CHECK-NEXT: call @second()
  "test.foo"() : () -> ()
  // CHECK-NEXT: test.foo
  // CHECK-NEXT: call @third()
  func.return
}

func.func @second() {
  "test.bar"() : () -> ()
  func.return
}

func.func private @third()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %funcs = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %f:3 = transform.split_handle %funcs : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %foo = transform.structured.match ops{["test.foo"]} in %f#0 : (!transform.any_op) -> !transform.any_op
    transform.func.cast_and_call @second before %foo : (!transform.any_op) -> !transform.any_op
    transform.func.cast_and_call %f#2 after %foo : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @non_empty_arg_and_out
func.func @non_empty_arg_and_out(%arg0 : index) -> i32 {
  // CHECK-NEXT: %[[FOO:.+]] = "test.foo"
  %0 = "test.foo"(%arg0) : (index) -> (index)
  // CHECK-NEXT: %[[CALL:.+]] = call @second(%[[FOO]]) : (index) -> i32
  %1 = "test.bar"(%0) : (index) -> (i32)
  // CHECK: return %[[CALL]] : i32
  func.return %1 : i32
}

func.func private @second(%arg1 : index) -> i32

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %funcs = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %f:2 = transform.split_handle %funcs : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %foo = transform.structured.match ops{["test.foo"]} in %f#0 : (!transform.any_op) -> !transform.any_op
    %bar = transform.structured.match ops{["test.bar"]} in %f#0 : (!transform.any_op) -> !transform.any_op
    %in = transform.get_result %foo[0] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %bar[0] : (!transform.any_op) -> !transform.any_value
    transform.func.cast_and_call %f#1(%in) -> %out before %bar
        : (!transform.any_op, !transform.any_value,
           !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @multi_arg_and_result
func.func @multi_arg_and_result(%arg0 : index) -> (index, index) {
  // CHECK-NEXT: %[[FOO:.+]] = "test.foo"
  %0 = "test.foo"(%arg0) : (index) -> (index)
  %1 = "test.bar"(%0) : (index) -> (index)
  %2 = "test.bar"(%0) : (index) -> (index)
  // CHECK: %[[CALL:.+]]:2 = call @second(%[[FOO]], %[[FOO]]) : (index, index) -> (index, index)
  // CHECK: return %[[CALL]]#0, %[[CALL]]#1 : index, index
  func.return %1, %2 : index, index
}

func.func private @second(%arg1: index, %arg2: index) -> (index, index)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %funcs = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %f:2 = transform.split_handle %funcs : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %foo = transform.structured.match ops{["test.foo"]} in %f#0 : (!transform.any_op) -> !transform.any_op
    %bars = transform.structured.match ops{["test.bar"]} in %f#0 : (!transform.any_op) -> !transform.any_op
    %in0 = transform.get_result %foo[0] : (!transform.any_op) -> !transform.any_value
    %in1 = transform.get_result %foo[0] : (!transform.any_op) -> !transform.any_value
    %ins = transform.merge_handles %in0, %in1 : !transform.any_value

    %outs = transform.get_result %bars[0] : (!transform.any_op) -> !transform.any_value

    transform.func.cast_and_call %f#1(%ins) -> %outs after %foo
        : (!transform.any_op, !transform.any_value,
           !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @nested_call
func.func @nested_call() {
  // CHECK-NEXT: %[[ARG:.+]] = "test.arg"
  // CHECK-NEXT: test.foo
  %0 = "test.arg"() : () -> (index)
  "test.foo"() ({
    // CHECK-NEXT: call @second(%[[ARG]]) : (index) -> ()
    "test.bar"(%0) : (index) -> ()
  }) : () -> ()
}

func.func private @second(%arg1: index) -> ()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %funcs = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %f:2 = transform.split_handle %funcs : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %arg = transform.structured.match ops{["test.arg"]} in %f#0 : (!transform.any_op) -> !transform.any_op
    %bar = transform.structured.match ops{["test.bar"]} in %f#0 : (!transform.any_op) -> !transform.any_op
    %in = transform.get_result %arg[0] : (!transform.any_op) -> !transform.any_value

    transform.func.cast_and_call %f#1(%in) before %bar
        : (!transform.any_op, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}
