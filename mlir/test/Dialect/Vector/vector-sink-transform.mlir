// RUN: mlir-opt %s --transform-interpreter --split-input-file | FileCheck %s

// This is smoke test for `transform.apply_patterns.vector.sink_ops` the actual
// patterns are tested in `vector-sink.mlir`.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.vector.sink_ops
    } : !transform.any_op
    transform.yield
  }
}


// CHECK-LABEL: @extract_elementwise_scalar
//  CHECK-SAME:   (%[[ARG0:.*]]: vector<4xf32>, %[[ARG1:.*]]: vector<4xf32>)
func.func @extract_elementwise_scalar(%arg0: vector<4xf32>, %arg1: vector<4xf32>) -> f32 {
// CHECK:   %[[EXT0:.*]] = vector.extract %[[ARG0]][1] : f32 from vector<4xf32>
// CHECK:   %[[EXT1:.*]] = vector.extract %[[ARG1]][1] : f32 from vector<4xf32>
// CHECK:   %[[RES:.*]] = arith.addf %[[EXT0]], %[[EXT1]] : f32
// CHECK:   return %[[RES]] : f32
  %0 = arith.addf %arg0, %arg1 : vector<4xf32>
  %1 = vector.extract %0[1] : f32 from vector<4xf32>
  return %1 : f32
}
