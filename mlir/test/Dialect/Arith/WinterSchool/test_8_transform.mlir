// RUN: mlir-opt %s -test-arith-reduce-float-bitwidth="patterns=arith.addf"

func.func @test_add(%arg0: f32, %arg1: f32) -> f32 {
  %0 = arith.addf %arg0, %arg1 : f32
  return %0 : f32
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.op<"func.func">
    %add_op = transform.structured.match ops{["arith.addf"]} in %func_op : (!transform.op<"func.func">) -> !transform.op<"arith.addf">
    transform.debug.emit_remark_at %add_op, "before pattern application" : !transform.op<"arith.addf">

    transform.apply_patterns to %func_op {
      transform.apply_patterns.arith.reduce_float_bitwidth ["arith.addf"] from f32 to f16
      // transform.apply_patterns.arith.reduce_float_bitwidth ["func.func", "func.return", "arith.addf"] from f32 to f16
      // transform.apply_patterns.arith.reduce_float_bitwidth ["arith.addf_v2"] from f32 to f16
    } : !transform.op<"func.func">

    transform.debug.emit_remark_at %add_op, "after pattern application" : !transform.op<"arith.addf">
    transform.yield
  }
}
