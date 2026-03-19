// RUN: mlir-opt %s --transform-interpreter | FileCheck %s

module @td_module_4 attributes {transform.with_named_sequence} {
  module @foo_module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) -> () {
      // CHECK: IR printer: foo_module top-level
      transform.print {name="foo_module"}
      transform.yield
    }
  }
  module @bar_module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) -> () {
      // CHECK: IR printer: bar_module top-level
      transform.print {name="bar_module"}
      transform.yield
    }
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) -> () {
    transform.include @foo_module::@__transform_main failures(suppress) (%arg0) : (!transform.any_op) -> ()
    transform.include @bar_module::@__transform_main failures(suppress) (%arg0) : (!transform.any_op) -> ()
    transform.yield
  }
}
