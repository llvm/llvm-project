// RUN: mlir-opt %s --transform-interpreter -allow-unregistered-dialect --split-input-file -verify-diagnostics

module {
  func.func private @func_with_reverse_order_no_result_no_calls(%arg0: memref<1xi8, 1>, %arg1: memref<2xi8, 1>, %arg2: memref<3xi8, 1>) {
    %c0 = arith.constant 0 : index
    %view = memref.view %arg0[%c0][] : memref<1xi8, 1> to memref<1xi8, 1>
    %view0 = memref.view %arg1[%c0][] : memref<2xi8, 1> to memref<2xi8, 1>
    %view1 = memref.view %arg2[%c0][] : memref<3xi8, 1> to memref<3xi8, 1>
    return
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %module = transform.get_parent_op %func : (!transform.any_op) -> !transform.any_op
    // expected-error @+1 {{function with name '@func_not_in_module' not found}}
    transform.func.replace_func_signature @func_not_in_module args_interchange = [0, 2, 1] results_interchange = [] at %module : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

module {
  func.func private @func_with_reverse_order_no_result_no_calls(%arg0: memref<1xi8, 1>, %arg1: memref<2xi8, 1>, %arg2: memref<3xi8, 1>) {
    %c0 = arith.constant 0 : index
    %view = memref.view %arg0[%c0][] : memref<1xi8, 1> to memref<1xi8, 1>
    %view0 = memref.view %arg1[%c0][] : memref<2xi8, 1> to memref<2xi8, 1>
    %view1 = memref.view %arg2[%c0][] : memref<3xi8, 1> to memref<3xi8, 1>
    return
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %module = transform.get_parent_op %func : (!transform.any_op) -> !transform.any_op
    // expected-error @+1 {{function with name '@func_with_reverse_order_no_result_no_calls' has 3 arguments, but 2 args interchange were given}}
    transform.func.replace_func_signature @func_with_reverse_order_no_result_no_calls args_interchange = [0, 2] results_interchange = [] at %module : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

module {
  func.func private @func_with_reverse_order_no_result_no_calls(%arg0: memref<1xi8, 1>, %arg1: memref<2xi8, 1>, %arg2: memref<3xi8, 1>) {
    %c0 = arith.constant 0 : index
    %view = memref.view %arg0[%c0][] : memref<1xi8, 1> to memref<1xi8, 1>
    %view0 = memref.view %arg1[%c0][] : memref<2xi8, 1> to memref<2xi8, 1>
    %view1 = memref.view %arg2[%c0][] : memref<3xi8, 1> to memref<3xi8, 1>
    return
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %module = transform.get_parent_op %func : (!transform.any_op) -> !transform.any_op
    // expected-error @+1 {{function with name '@func_with_reverse_order_no_result_no_calls' has 0 results, but 1 results interchange were given}}
    transform.func.replace_func_signature @func_with_reverse_order_no_result_no_calls args_interchange = [0, 2, 1] results_interchange = [0] at %module : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

module {
  func.func private @func_with_reverse_order_no_result_no_calls(%arg0: memref<1xi8, 1>, %arg1: memref<2xi8, 1>, %arg2: memref<3xi8, 1>) {
    %c0 = arith.constant 0 : index
    %view = memref.view %arg0[%c0][] : memref<1xi8, 1> to memref<1xi8, 1>
    %view0 = memref.view %arg1[%c0][] : memref<2xi8, 1> to memref<2xi8, 1>
    %view1 = memref.view %arg2[%c0][] : memref<3xi8, 1> to memref<3xi8, 1>
    return
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %module = transform.get_parent_op %func : (!transform.any_op) -> !transform.any_op
    // expected-error @+1 {{args interchange must be unique}}
    transform.func.replace_func_signature @func_with_reverse_order_no_result_no_calls args_interchange = [0, 2, 2] results_interchange = [] at %module : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
