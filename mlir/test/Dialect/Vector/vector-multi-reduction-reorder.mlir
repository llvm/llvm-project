// RUN: mlir-opt %s --transform-interpreter='entry-point=innerreduction' | FileCheck %s --check-prefix=INNER_REDUCTION,ALL
// RUN: mlir-opt %s --transform-interpreter='entry-point=innerparallel' | FileCheck %s --check-prefix=INNER_PARALLEL,ALL

// INNER_REDUCTION-LABEL: func @inner_reduction_to_inner_parallel
// INNER_REDUCTION-SAME:    %[[INPUT:.+]]: vector<3x2x4xf32>
// INNER_REDUCTION-SAME:    %[[ACC:.+]]: vector<2x4xf32>
func.func @inner_reduction_to_inner_parallel(%arg0: vector<3x2x4xf32>, %acc: vector<2x4xf32>) -> vector<2x4xf32> {
    // INNER_REDUCTION: %[[TRANSPOSED:.+]] = vector.transpose %[[INPUT]], [1, 2, 0]
    // INNER_REDUCTION: %[[RESULT:.+]] = vector.multi_reduction <mul>, %[[TRANSPOSED]], %[[ACC]] [2]
    %0 = vector.multi_reduction <mul>, %arg0, %acc [0] : vector<3x2x4xf32> to vector<2x4xf32>
    // INNER_REDUCTION: return %[[RESULT]]
    return %0 : vector<2x4xf32>
}

// INNER_PARALLEL-LABEL: func @inner_parallel_to_inner_reduction
// INNER_PARALLEL-SAME:    %[[INPUT:.+]]: vector<3x2x4xf32>
// INNER_PARALLEL-SAME:    %[[ACC:.+]]: vector<3x2xf32>
func.func @inner_parallel_to_inner_reduction(%arg0: vector<3x2x4xf32>, %acc: vector<3x2xf32>) -> vector<3x2xf32> {
    // INNER_PARALLEL: %[[TRANSPOSED:.+]] = vector.transpose %[[INPUT]], [2, 0, 1]
    // INNER_PARALLEL: %[[RESULT:.+]] = vector.multi_reduction <mul>, %[[TRANSPOSED]], %[[ACC]] [0]
    %0 = vector.multi_reduction <mul>, %arg0, %acc [2] : vector<3x2x4xf32> to vector<3x2xf32>
    // INNER_PARALLEL: return %[[RESULT]]
    return %0 : vector<3x2xf32>
}

// ALL-LABEL: func @transpose_parallel_middle
// ALL-SAME:    %[[INPUT:.+]]: vector<3x4x5xf32>
// ALL-SAME:    %[[ACC:.+]]: vector<4xf32>
func.func @transpose_parallel_middle(%arg0: vector<3x4x5xf32>, %acc: vector<4xf32>) -> vector<4xf32> {
    // INNER_REDUCTION: %[[TRANSPOSED:.+]] = vector.transpose %[[INPUT]], [1, 0, 2]
    // INNER_REDUCTION: %[[RESULT:.+]] = vector.multi_reduction <add>, %[[TRANSPOSED]], %[[ACC]] [1, 2]
    // INNER_PARALLEL: %[[TRANSPOSED:.+]] = vector.transpose %[[INPUT]], [0, 2, 1]
    // INNER_PARALLEL: %[[RESULT:.+]] = vector.multi_reduction <add>, %[[TRANSPOSED]], %[[ACC]] [0, 1]
    %0 = vector.multi_reduction <add>, %arg0, %acc [0, 2] : vector<3x4x5xf32> to vector<4xf32>
    // ALL: return %[[RESULT]]
    return %0 : vector<4xf32>
}

// ALL-LABEL: func @negative_one_dim
func.func @negative_one_dim(%arg0: vector<8xf32>, %acc: f32) -> f32 {
    // ALL: vector.multi_reduction <add>, {{.+}} [0] : vector<8xf32> to f32
    %0 = vector.multi_reduction <add>, %arg0, %acc [0] : vector<8xf32> to f32
    return %0 : f32
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @innerreduction(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.reorder_multi_reduction_dims lowering_strategy = "innerreduction"
    } : !transform.op<"func.func">
    transform.yield
  }

  transform.named_sequence @innerparallel(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.reorder_multi_reduction_dims lowering_strategy = "innerparallel"
    } : !transform.op<"func.func">
    transform.yield
  }
}
