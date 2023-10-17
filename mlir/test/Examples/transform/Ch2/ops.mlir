// RUN: transform-opt-ch2 %s --test-transform-dialect-interpreter | FileCheck %s

// ****************************** IMPORTANT NOTE ******************************
//
// If you are changing this file, you may also need to change
// mlir/docs/Tutorials/Transform accordingly.
//
// ****************************************************************************

func.func private @orig()
func.func private @updated()

// CHECK-LABEL: func @test
func.func @test() {
  // CHECK: call @updated
  call @orig() : () -> ()
  return
}

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %call = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  // CHECK: transform.my.change_call_target %{{.*}}, "updated" : !transform.any_op
  transform.my.change_call_target %call, "updated" : !transform.any_op
  transform.yield
}
