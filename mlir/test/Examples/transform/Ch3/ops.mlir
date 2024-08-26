// RUN: transform-opt-ch3 %s --transform-interpreter \
// RUN:   --allow-unregistered-dialect --split-input-file | FileCheck %s

// ****************************** IMPORTANT NOTE ******************************
//
// If you are changing this file, you may also need to change
// mlir/docs/Tutorials/Transform accordingly.
//
// ****************************************************************************

func.func private @orig()
func.func private @updated()

// CHECK-LABEL: func @test1
func.func @test1() {
  // CHECK: call @updated
  call @orig() : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %call = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.op<"func.call">
    // CHECK: transform.my.change_call_target %{{.*}}, "updated" : !transform.op<"func.call">
    transform.my.change_call_target %call, "updated" : !transform.op<"func.call">
    transform.yield
  }
}

// -----

func.func private @orig()

// CHECK-LABEL: func @test2
func.func @test2() {
  // CHECK: "my.mm4"
  call @orig() : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %call = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.my.call_op_interface
    // CHECK: transform.my.call_to_op %{{.*}} : (!transform.my.call_op_interface) -> !transform.any_op
    transform.my.call_to_op %call : (!transform.my.call_op_interface) -> !transform.any_op
    transform.yield
  }
}
