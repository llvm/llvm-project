// RUN: mlir-opt %s --transform-interpreter --split-input-file | FileCheck %s

// CHECK-LABEL: @extract_elementwise
//  CHECK-SAME:   (%[[ARG0:.*]]: vector<4xf32>, %[[ARG1:.*]]: vector<4xf32>)
func.func @extract_elementwise(%arg0: vector<4xf32>, %arg1: vector<4xf32>) -> f32 {
// CHECK:   %[[EXT0:.*]] = vector.extract %[[ARG0]][1] : f32 from vector<4xf32>
// CHECK:   %[[EXT1:.*]] = vector.extract %[[ARG1]][1] : f32 from vector<4xf32>
// CHECK:   %[[RES:.*]] = arith.addf %[[EXT0]], %[[EXT1]] : f32
// CHECK:   return %[[RES]] : f32
  %0 = arith.addf %arg0, %arg1 : vector<4xf32>
  %1 = vector.extract %0[1] : f32 from vector<4xf32>
  return %1 : f32
}

// CHECK-LABEL: @extract_vec_elementwise
//  CHECK-SAME:   (%[[ARG0:.*]]: vector<2x4xf32>, %[[ARG1:.*]]: vector<2x4xf32>)
func.func @extract_vec_elementwise(%arg0: vector<2x4xf32>, %arg1: vector<2x4xf32>) -> vector<4xf32> {
// CHECK:   %[[EXT0:.*]] = vector.extract %[[ARG0]][1] : vector<4xf32> from vector<2x4xf32>
// CHECK:   %[[EXT1:.*]] = vector.extract %[[ARG1]][1] : vector<4xf32> from vector<2x4xf32>
// CHECK:   %[[RES:.*]] = arith.addf %[[EXT0]], %[[EXT1]] : vector<4xf32>
// CHECK:   return %[[RES]] : vector<4xf32>
  %0 = arith.addf %arg0, %arg1 : vector<2x4xf32>
  %1 = vector.extract %0[1] : vector<4xf32> from vector<2x4xf32>
  return %1 : vector<4xf32>
}

// CHECK-LABEL: @extract_elementwise_use
//  CHECK-SAME:   (%[[ARG0:.*]]: vector<4xf32>, %[[ARG1:.*]]: vector<4xf32>)
func.func @extract_elementwise_use(%arg0: vector<4xf32>, %arg1: vector<4xf32>) -> (f32, vector<4xf32>) {
// Do not propagate extract, as elementwise has other uses.
// CHECK:   %[[ELT:.*]] = arith.addf %[[ARG0]], %[[ARG1]] : vector<4xf32>
// CHECK:   %[[EXT:.*]] = vector.extract %[[ELT]][1] : f32 from vector<4xf32>
// CHECK:   return %[[EXT]], %[[ELT]] : f32, vector<4xf32>
  %0 = arith.addf %arg0, %arg1 : vector<4xf32>
  %1 = vector.extract %0[1] : f32 from vector<4xf32>
  return %1, %0 : f32, vector<4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.vector.propagate_extract
    } : !transform.any_op
    transform.yield
  }
}
