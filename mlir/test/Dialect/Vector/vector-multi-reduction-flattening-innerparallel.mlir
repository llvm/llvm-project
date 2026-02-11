// RUN: mlir-opt %s --transform-interpreter | FileCheck %s

// CHECK-LABEL: func @vector_multi_reduction_to_scalar
// CHECK-SAME:    %[[INPUT:.+]]: vector<2x3xf32>
// CHECK-SAME:    %[[ACC:.+]]: f32
func.func @vector_multi_reduction_to_scalar(%arg0: vector<2x3xf32>, %acc: f32) -> f32 {
    // CHECK: %[[CASTED:.+]] = vector.shape_cast %[[INPUT]] : vector<2x3xf32> to vector<6xf32>
    // CHECK: %[[RESULT:.+]] = vector.multi_reduction <add>, %[[CASTED]], %[[ACC]] [0]
    %0 = vector.multi_reduction <add>, %arg0, %acc [0, 1] : vector<2x3xf32> to f32
    // CHECK: return %[[RESULT]]
    return %0 : f32
}

// Test with parallel dimension: reduction dims [0,1] are already outermost,
// parallel dim [2] is innermost - this can be flattened.
// CHECK-LABEL: func @vector_multi_reduction_parallel_dim
// CHECK-SAME:    %[[INPUT:.+]]: vector<3x4x2xi32>
// CHECK-SAME:    %[[ACC:.+]]: vector<2xi32>
func.func @vector_multi_reduction_parallel_dim(%arg0: vector<3x4x2xi32>, %acc: vector<2xi32>) -> vector<2xi32> {
    // CHECK: %[[CASTED:.+]] = vector.shape_cast %[[INPUT]] : vector<3x4x2xi32> to vector<12x2xi32>
    // CHECK: %[[RESULT:.+]] = vector.multi_reduction <mul>, %[[CASTED]], %[[ACC]] [0]
    %0 = vector.multi_reduction <mul>, %arg0, %acc [0, 1] : vector<3x4x2xi32> to vector<2xi32>
    // CHECK: return %[[RESULT]]
    return %0 : vector<2xi32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.lower_multi_reduction_flattening lowering_strategy = "innerparallel"
    } : !transform.op<"func.func">
    transform.yield
  }
}
