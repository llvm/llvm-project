// RUN: mlir-opt %s --pass-pipeline='builtin.module(transform-interpreter)'

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.yield
  }
  
  func.func @f() {
    return
  }

  func.func @main() {
    // This call is marked noinline, so it is illegal to inline.
    // The fix ensures that this does not cause an error during symbol merging.
    "test.conversion_call_op"() {callee = @f, noinline} : () -> ()
    return
  }
}
