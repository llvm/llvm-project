// RUN: mlir-opt %s --transform-interpreter | FileCheck %s

// Patterns applied:
// * ReduceMultiDimReductionRank from populateVectorMultiReductionFlatteningPatterns
func.func @vector_multi_reduction_flattening(%arg0: vector<2x4xf32>, %acc: f32) -> f32 {
    %0 = vector.multi_reduction <mul>, %arg0, %acc [0, 1] : vector<2x4xf32> to f32
    return %0 : f32
}

// CHECK-LABEL: func @vector_multi_reduction_flattening
//  CHECK-SAME:   %[[INPUT:.+]]: vector<2x4xf32>, %[[ACC:.*]]: f32)
//       CHECK:   %[[CASTED:.*]] = vector.shape_cast %[[INPUT]] : vector<2x4xf32> to vector<8xf32>
//       CHECK:   %[[RESULT:.+]] = vector.multi_reduction <mul>, %[[CASTED]], %[[ACC]] [0]
//       CHECK:   return %[[RESULT]]

// CHECK-LABEL: func @vector_multi_reduction_parallel_dim
// CHECK-SAME:    %[[INPUT:.+]]: vector<2x3x4xi32>
// CHECK-SAME:    %[[ACC:.+]]: vector<2xi32>
func.func @vector_multi_reduction_parallel_dim(%arg0: vector<2x3x4xi32>, %acc: vector<2xi32>) -> vector<2xi32> {
    // CHECK: %[[CASTED:.+]] = vector.shape_cast %[[INPUT]] : vector<2x3x4xi32> to vector<2x12xi32>
    // CHECK: %[[RESULT:.+]] = vector.multi_reduction <add>, %[[CASTED]], %[[ACC]] [1]
    %0 = vector.multi_reduction <add>, %arg0, %acc [1, 2] : vector<2x3x4xi32> to vector<2xi32>
    // CHECK: return %[[RESULT]]
    return %0 : vector<2xi32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.lower_multi_reduction_flattening lowering_strategy = "innerreduction"
    } : !transform.op<"func.func">
    transform.yield
  }
}
