// RUN: mlir-opt %s --transform-interpreter | FileCheck %s

// CHECK-LABEL: func @unroll_multi_reduction_inner_parallel
// CHECK-SAME:    %[[INPUT:.+]]: vector<4x2x3xf32>, %[[ACC:.+]]: vector<2xf32>
func.func @unroll_multi_reduction_inner_parallel(%arg0: vector<4x2x3xf32>, %acc: vector<2xf32>) -> vector<2xf32> {
    // CHECK: %[[V0:.+]] = vector.extract %[[INPUT]][0] : vector<2x3xf32> from vector<4x2x3xf32>
    // CHECK: %[[V1:.+]] = vector.extract %[[INPUT]][1] : vector<2x3xf32> from vector<4x2x3xf32>
    // CHECK: %[[V2:.+]] = vector.extract %[[INPUT]][2] : vector<2x3xf32> from vector<4x2x3xf32>
    // CHECK: %[[V3:.+]] = vector.extract %[[INPUT]][3] : vector<2x3xf32> from vector<4x2x3xf32>
    // CHECK: %[[RV0:.+]] = vector.multi_reduction <mul>, %[[V0]], %[[ACC]] [1] : vector<2x3xf32> to vector<2xf32>
    // CHECK: %[[RV1:.+]] = vector.multi_reduction <mul>, %[[V1]], %[[RV0]] [1] : vector<2x3xf32> to vector<2xf32>
    // CHECK: %[[RV2:.+]] = vector.multi_reduction <mul>, %[[V2]], %[[RV1]] [1] : vector<2x3xf32> to vector<2xf32>
    // CHECK: %[[RESULT:.+]] = vector.multi_reduction <mul>, %[[V3]], %[[RV2]] [1] : vector<2x3xf32> to vector<2xf32>
    %0 = vector.multi_reduction <mul>, %arg0, %acc [0, 2] : vector<4x2x3xf32> to vector<2xf32>
    // CHECK:             return %[[RESULT]]
    return %0 : vector<2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.multi_reduction_unrolling
    } : !transform.op<"func.func">
    transform.yield
  }
}
