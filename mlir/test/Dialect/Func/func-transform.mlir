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

// -----

module {
  // CHECK:           func.func private @func_with_reverse_order_no_result_no_calls(%[[ARG0:.*]]: memref<1xi8, 1>, %[[ARG1:.*]]: memref<3xi8, 1>, %[[ARG2:.*]]: memref<2xi8, 1>) {
  func.func private @func_with_reverse_order_no_result_no_calls(%arg0: memref<1xi8, 1>, %arg1: memref<2xi8, 1>, %arg2: memref<3xi8, 1>) {
    // CHECK:             %[[C0:.*]] = arith.constant 0 : index
    %c0 = arith.constant 0 : index
    // CHECK:             %[[VAL_4:.*]] = memref.view %[[ARG0]]{{\[}}%[[C0]]][] : memref<1xi8, 1> to memref<1xi8, 1>
    %view = memref.view %arg0[%c0][] : memref<1xi8, 1> to memref<1xi8, 1>
    // CHECK:             %[[VAL_5:.*]] = memref.view %[[ARG2]]{{\[}}%[[C0]]][] : memref<2xi8, 1> to memref<2xi8, 1>
    %view0 = memref.view %arg1[%c0][] : memref<2xi8, 1> to memref<2xi8, 1>
    // CHECK:             %[[VAL_6:.*]] = memref.view %[[ARG1]]{{\[}}%[[C0]]][] : memref<3xi8, 1> to memref<3xi8, 1>
    %view1 = memref.view %arg2[%c0][] : memref<3xi8, 1> to memref<3xi8, 1>
    return
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %module = transform.get_parent_op %func : (!transform.any_op) -> !transform.any_op
    transform.func.replace_func_signature @func_with_reverse_order_no_result_no_calls args_interchange = [0, 2, 1] results_interchange = [] at %module : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

module {
  // CHECK:           func.func private @func_with_reverse_order_no_result(%[[ARG0:.*]]: memref<1xi8, 1>, %[[ARG1:.*]]: memref<3xi8, 1>, %[[ARG2:.*]]: memref<2xi8, 1>) {
  func.func private @func_with_reverse_order_no_result(%arg0: memref<1xi8, 1>, %arg1: memref<2xi8, 1>, %arg2: memref<3xi8, 1>) {
    // CHECK:             %[[C0:.*]] = arith.constant 0 : index
    %c0 = arith.constant 0 : index
    // CHECK:             %[[VAL_4:.*]] = memref.view %[[ARG0]]{{\[}}%[[C0]]][] : memref<1xi8, 1> to memref<1xi8, 1>
    %view = memref.view %arg0[%c0][] : memref<1xi8, 1> to memref<1xi8, 1>
    // CHECK:             %[[VAL_5:.*]] = memref.view %[[ARG2]]{{\[}}%[[C0]]][] : memref<2xi8, 1> to memref<2xi8, 1>
    %view0 = memref.view %arg1[%c0][] : memref<2xi8, 1> to memref<2xi8, 1>
    // CHECK:             %[[VAL_6:.*]] = memref.view %[[ARG1]]{{\[}}%[[C0]]][] : memref<3xi8, 1> to memref<3xi8, 1>
    %view1 = memref.view %arg2[%c0][] : memref<3xi8, 1> to memref<3xi8, 1>
    return
  }

  // CHECK:           func.func @func_with_reverse_order_no_result_caller(%[[ARG0:.*]]: memref<1xi8, 1>, %[[ARG1:.*]]: memref<2xi8, 1>, %[[ARG2:.*]]: memref<3xi8, 1>) {
  func.func @func_with_reverse_order_no_result_caller(%arg0: memref<1xi8, 1>, %arg1: memref<2xi8, 1>, %arg2: memref<3xi8, 1>) {
    // CHECK:             call @func_with_reverse_order_no_result(%[[ARG0]], %[[ARG2]], %[[ARG1]]) : (memref<1xi8, 1>, memref<3xi8, 1>, memref<2xi8, 1>) -> ()
    call @func_with_reverse_order_no_result(%arg0, %arg1, %arg2) : (memref<1xi8, 1>, memref<2xi8, 1>, memref<3xi8, 1>) -> ()
    return
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %f:2 = transform.split_handle %funcs : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %module = transform.get_parent_op %f#0 : (!transform.any_op) -> !transform.any_op
    transform.func.replace_func_signature @func_with_reverse_order_no_result args_interchange = [0, 2, 1] results_interchange = [] at %module {adjust_func_calls} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

module {
  // CHECK:           func.func private @func_with_reverse_order(%[[ARG0:.*]]: memref<1xi8, 1>, %[[ARG1:.*]]: memref<3xi8, 1>, %[[ARG2:.*]]: memref<2xi8, 1>) -> (memref<2xi8, 1>, memref<1xi8, 1>) {
  func.func private @func_with_reverse_order(%arg0: memref<1xi8, 1>, %arg1: memref<2xi8, 1>, %arg2: memref<3xi8, 1>) -> (memref<1xi8, 1>, memref<2xi8, 1>) {
    // CHECK:             %[[C0:.*]] = arith.constant 0 : index
    %c0 = arith.constant 0 : index
    // CHECK:             %[[RET_0:.*]] = memref.view %[[ARG0]]{{\[}}%[[C0]]][] : memref<1xi8, 1> to memref<1xi8, 1>
    %view = memref.view %arg0[%c0][] : memref<1xi8, 1> to memref<1xi8, 1>
    // CHECK:             %[[RET_1:.*]] = memref.view %[[ARG2]]{{\[}}%[[C0]]][] : memref<2xi8, 1> to memref<2xi8, 1>
    %view0 = memref.view %arg1[%c0][] : memref<2xi8, 1> to memref<2xi8, 1>
    // CHECK:             %[[VAL_6:.*]] = memref.view %[[ARG1]]{{\[}}%[[C0]]][] : memref<3xi8, 1> to memref<3xi8, 1>
    %view1 = memref.view %arg2[%c0][] : memref<3xi8, 1> to memref<3xi8, 1>
    // CHECK:             return %[[RET_1]], %[[RET_0]] : memref<2xi8, 1>, memref<1xi8, 1>
    return %view, %view0 : memref<1xi8, 1>, memref<2xi8, 1>
  }

  // CHECK:           func.func @func_with_reverse_order_caller(%[[ARG0:.*]]: memref<1xi8, 1>, %[[ARG1:.*]]: memref<2xi8, 1>, %[[ARG2:.*]]: memref<3xi8, 1>) -> (memref<1xi8, 1>, memref<2xi8, 1>) {
  func.func @func_with_reverse_order_caller(%arg0: memref<1xi8, 1>, %arg1: memref<2xi8, 1>, %arg2: memref<3xi8, 1>) -> (memref<1xi8, 1>, memref<2xi8, 1>) {
    // CHECK:             %[[RET:.*]]:2 = call @func_with_reverse_order(%[[ARG0]], %[[ARG2]], %[[ARG1]]) : (memref<1xi8, 1>, memref<3xi8, 1>, memref<2xi8, 1>) -> (memref<2xi8, 1>, memref<1xi8, 1>)
    %0, %1 = call @func_with_reverse_order(%arg0, %arg1, %arg2) : (memref<1xi8, 1>, memref<2xi8, 1>, memref<3xi8, 1>) -> (memref<1xi8, 1>, memref<2xi8, 1>)
    // CHECK:             return %[[RET]]#1, %[[RET]]#0 : memref<1xi8, 1>, memref<2xi8, 1>
    return %0, %1 : memref<1xi8, 1>, memref<2xi8, 1>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %f:2 = transform.split_handle %funcs : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %module = transform.get_parent_op %f#0 : (!transform.any_op) -> !transform.any_op
    transform.func.replace_func_signature @func_with_reverse_order args_interchange = [0, 2, 1] results_interchange = [1, 0] at %module {adjust_func_calls} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

module {
  // CHECK:           func.func private @func_with_reverse_order_with_attr(%[[ARG0:.*]]: memref<1xi8, 1>, %[[ARG1:.*]]: memref<3xi8, 1>, %[[ARG2:.*]]: memref<2xi8, 1> {transform.readonly}) -> (memref<2xi8, 1>, memref<1xi8, 1>) {
  func.func private @func_with_reverse_order_with_attr(%arg0: memref<1xi8, 1>, %arg1: memref<2xi8, 1>{transform.readonly}, %arg2: memref<3xi8, 1>) -> (memref<1xi8, 1>, memref<2xi8, 1>) {
    // CHECK:             %[[C0:.*]] = arith.constant 0 : index
    %c0 = arith.constant 0 : index
    // CHECK:             %[[RET_0:.*]] = memref.view %[[ARG0]]{{\[}}%[[C0]]][] : memref<1xi8, 1> to memref<1xi8, 1>
    %view = memref.view %arg0[%c0][] : memref<1xi8, 1> to memref<1xi8, 1>
    // CHECK:             %[[RET_1:.*]] = memref.view %[[ARG2]]{{\[}}%[[C0]]][] : memref<2xi8, 1> to memref<2xi8, 1>
    %view0 = memref.view %arg1[%c0][] : memref<2xi8, 1> to memref<2xi8, 1>
    // CHECK:             %[[VAL_6:.*]] = memref.view %[[ARG1]]{{\[}}%[[C0]]][] : memref<3xi8, 1> to memref<3xi8, 1>
    %view1 = memref.view %arg2[%c0][] : memref<3xi8, 1> to memref<3xi8, 1>
    // CHECK:             return %[[RET_1]], %[[RET_0]] : memref<2xi8, 1>, memref<1xi8, 1>
    return %view, %view0 : memref<1xi8, 1>, memref<2xi8, 1>
  }

  // CHECK:           func.func @func_with_reverse_order_with_attr_caller(%[[ARG0:.*]]: memref<1xi8, 1>, %[[ARG1:.*]]: memref<2xi8, 1>, %[[ARG2:.*]]: memref<3xi8, 1>) -> (memref<1xi8, 1>, memref<2xi8, 1>) {
  func.func @func_with_reverse_order_with_attr_caller(%arg0: memref<1xi8, 1>, %arg1: memref<2xi8, 1>, %arg2: memref<3xi8, 1>) -> (memref<1xi8, 1>, memref<2xi8, 1>) {
    // CHECK:             %[[RET:.*]]:2 = call @func_with_reverse_order_with_attr(%[[ARG0]], %[[ARG2]], %[[ARG1]]) : (memref<1xi8, 1>, memref<3xi8, 1>, memref<2xi8, 1>) -> (memref<2xi8, 1>, memref<1xi8, 1>)
    %0, %1 = call @func_with_reverse_order_with_attr(%arg0, %arg1, %arg2) : (memref<1xi8, 1>, memref<2xi8, 1>, memref<3xi8, 1>) -> (memref<1xi8, 1>, memref<2xi8, 1>)
    // CHECK:             return %[[RET]]#1, %[[RET]]#0 : memref<1xi8, 1>, memref<2xi8, 1>
    return %0, %1 : memref<1xi8, 1>, memref<2xi8, 1>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %f:2 = transform.split_handle %funcs : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %module = transform.get_parent_op %f#0 : (!transform.any_op) -> !transform.any_op
    transform.func.replace_func_signature @func_with_reverse_order_with_attr args_interchange = [0, 2, 1] results_interchange = [1, 0] at %module {adjust_func_calls} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK:           func.func private @func_with_duplicate_args(%[[ARG0:.*]]: memref<1xi8, 1>, %[[ARG1:.*]]: memref<2xi8, 1>) {
func.func private @func_with_duplicate_args(%arg0: memref<1xi8, 1>, %arg1: memref<2xi8, 1>, %arg2: memref<1xi8, 1>) {
  %c0 = arith.constant 0 : index
  // CHECK:             %[[VAL_3:.*]] = memref.view %[[ARG0]]{{\[}}%[[C0:.*]]][] : memref<1xi8, 1> to memref<1xi8, 1>
  %view = memref.view %arg0[%c0][] : memref<1xi8, 1> to memref<1xi8, 1>
  // CHECK:             %[[VAL_4:.*]] = memref.view %[[ARG1]]{{\[}}%[[C0]]][] : memref<2xi8, 1> to memref<2xi8, 1>
  %view0 = memref.view %arg1[%c0][] : memref<2xi8, 1> to memref<2xi8, 1>
  // CHECK:             %[[VAL_5:.*]] = memref.view %[[ARG0]]{{\[}}%[[C0]]][] : memref<1xi8, 1> to memref<1xi8, 1>
  %view1 = memref.view %arg2[%c0][] : memref<1xi8, 1> to memref<1xi8, 1>
  return
}

// CHECK:           func.func @func_with_duplicate_args_caller(%[[ARG0:.*]]: memref<1xi8, 1>, %[[ARG1:.*]]: memref<2xi8, 1>) {
func.func @func_with_duplicate_args_caller(%arg0: memref<1xi8, 1>, %arg1: memref<2xi8, 1>) {
  // CHECK:             call @func_with_duplicate_args(%[[ARG0]], %[[ARG1]]) : (memref<1xi8, 1>, memref<2xi8, 1>) -> ()
  call @func_with_duplicate_args(%arg0, %arg1, %arg0) : (memref<1xi8, 1>, memref<2xi8, 1>, memref<1xi8, 1>) -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    transform.func.deduplicate_func_args @func_with_duplicate_args at %module : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK:           func.func private @func_with_complex_duplicate_args(%[[ARG0:.*]]: memref<1xi8, 1>, %[[ARG1:.*]]: memref<2xi8, 1>, %[[ARG2:.*]]: memref<3xi8, 1>) -> (memref<1xi8, 1>, memref<2xi8, 1>, memref<1xi8, 1>, memref<3xi8, 1>, memref<2xi8, 1>) {
func.func private @func_with_complex_duplicate_args(%arg0: memref<1xi8, 1>, %arg1: memref<2xi8, 1>, %arg2: memref<1xi8, 1>, %arg3: memref<3xi8, 1>, %arg4: memref<2xi8, 1>) -> (memref<1xi8, 1>, memref<2xi8, 1>, memref<1xi8, 1>, memref<3xi8, 1>, memref<2xi8, 1>) {
  %c0 = arith.constant 0 : index
  // CHECK:             %[[RET_0:.*]] = memref.view %[[ARG0]]{{\[}}%[[C0:.*]]][] : memref<1xi8, 1> to memref<1xi8, 1>
  %view0 = memref.view %arg0[%c0][] : memref<1xi8, 1> to memref<1xi8, 1>
  // CHECK:             %[[RET_1:.*]] = memref.view %[[ARG1]]{{\[}}%[[C0]]][] : memref<2xi8, 1> to memref<2xi8, 1>
  %view1 = memref.view %arg1[%c0][] : memref<2xi8, 1> to memref<2xi8, 1>
  // CHECK:             %[[RET_2:.*]] = memref.view %[[ARG0]]{{\[}}%[[C0]]][] : memref<1xi8, 1> to memref<1xi8, 1>
  %view2 = memref.view %arg2[%c0][] : memref<1xi8, 1> to memref<1xi8, 1>
  // CHECK:             %[[RET_3:.*]] = memref.view %[[ARG2]]{{\[}}%[[C0]]][] : memref<3xi8, 1> to memref<3xi8, 1>
  %view3 = memref.view %arg3[%c0][] : memref<3xi8, 1> to memref<3xi8, 1>
  // CHECK:             %[[RET_4:.*]] = memref.view %[[ARG1]]{{\[}}%[[C0]]][] : memref<2xi8, 1> to memref<2xi8, 1>
  %view4 = memref.view %arg4[%c0][] : memref<2xi8, 1> to memref<2xi8, 1>
  // CHECK:             return %[[RET_0]], %[[RET_1]], %[[RET_2]], %[[RET_3]], %[[RET_4]] : memref<1xi8, 1>, memref<2xi8, 1>, memref<1xi8, 1>, memref<3xi8, 1>, memref<2xi8, 1>
  return %view0, %view1, %view2, %view3, %view4 : memref<1xi8, 1>, memref<2xi8, 1>, memref<1xi8, 1>, memref<3xi8, 1>, memref<2xi8, 1>
}

// CHECK:           func.func @func_with_complex_duplicate_args_caller(%[[ARG0:.*]]: memref<1xi8, 1>, %[[ARG1:.*]]: memref<2xi8, 1>, %[[ARG2:.*]]: memref<3xi8, 1>) -> (memref<1xi8, 1>, memref<2xi8, 1>, memref<1xi8, 1>, memref<3xi8, 1>, memref<2xi8, 1>) {
func.func @func_with_complex_duplicate_args_caller(%arg0: memref<1xi8, 1>, %arg1: memref<2xi8, 1>, %arg2: memref<3xi8, 1>) -> (memref<1xi8, 1>, memref<2xi8, 1>, memref<1xi8, 1>, memref<3xi8, 1>, memref<2xi8, 1>) {
  // CHECK:             %[[RET:.*]]:5 = call @func_with_complex_duplicate_args(%[[ARG0]], %[[ARG1]], %[[ARG2]]) : (memref<1xi8, 1>, memref<2xi8, 1>, memref<3xi8, 1>) -> (memref<1xi8, 1>, memref<2xi8, 1>, memref<1xi8, 1>, memref<3xi8, 1>, memref<2xi8, 1>)
  %0:5 = call @func_with_complex_duplicate_args(%arg0, %arg1, %arg0, %arg2, %arg1) : (memref<1xi8, 1>, memref<2xi8, 1>, memref<1xi8, 1>, memref<3xi8, 1>, memref<2xi8, 1>) -> (memref<1xi8, 1>, memref<2xi8, 1>, memref<1xi8, 1>, memref<3xi8, 1>, memref<2xi8, 1>)
  // CHECK:             return %[[RET]]#0, %[[RET]]#1, %[[RET]]#2, %[[RET]]#3, %[[RET]]#4 : memref<1xi8, 1>, memref<2xi8, 1>, memref<1xi8, 1>, memref<3xi8, 1>, memref<2xi8, 1>
  return %0#0, %0#1, %0#2, %0#3, %0#4 : memref<1xi8, 1>, memref<2xi8, 1>, memref<1xi8, 1>, memref<3xi8, 1>, memref<2xi8, 1>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    transform.func.deduplicate_func_args @func_with_complex_duplicate_args at %module : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
