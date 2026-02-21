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

// ALL-LABEL: func @one_dim_to_two_dim
// ALL-SAME:    %[[INPUT:.+]]: vector<8xf32>
// ALL-SAME:    %[[ACC:.+]]: f32
func.func @one_dim_to_two_dim(%arg0: vector<8xf32>, %acc: f32) -> f32 {
    // ALL: %[[CAST:.+]] = vector.shape_cast %[[INPUT]] : vector<8xf32> to vector<1x8xf32>
    // ALL: %[[BROADCAST:.+]] = vector.broadcast %[[ACC]] : f32 to vector<1xf32>
    // INNER_REDUCTION: %[[RESULT:.+]] = vector.multi_reduction <add>, %[[CAST]], %[[BROADCAST]] [1]
    // INNER_REDUCTION: %[[SCALAR:.+]] = vector.extract %[[RESULT]][0]
    // INNER_PARALLEL: %[[TRANSPOSED:.+]] = vector.transpose %[[CAST]], [1, 0]
    // INNER_PARALLEL: %[[RESULT:.+]] = vector.multi_reduction <add>, %[[TRANSPOSED]], %[[BROADCAST]] [0]
    // INNER_PARALLEL: %[[SCALAR:.+]] = vector.extract %[[RESULT]][0]
    %0 = vector.multi_reduction <add>, %arg0, %acc [0] : vector<8xf32> to f32
    // ALL: return %[[SCALAR]]
    return %0 : f32
}

// INNER_REDUCTION-LABEL: func @one_dim_to_two_dim_scalable
// INNER_REDUCTION-SAME:    %[[INPUT:.+]]: vector<[4]xf32>
// INNER_REDUCTION-SAME:    %[[ACC:.+]]: f32
func.func @one_dim_to_two_dim_scalable(%arg0: vector<[4]xf32>, %acc: f32) -> f32 {
    // INNER_REDUCTION: %[[CAST:.+]] = vector.shape_cast %[[INPUT]] : vector<[4]xf32> to vector<1x[4]xf32>
    // INNER_REDUCTION: %[[BROADCAST:.+]] = vector.broadcast %[[ACC]] : f32 to vector<1xf32>
    // INNER_REDUCTION: %[[RESULT:.+]] = vector.multi_reduction <add>, %[[CAST]], %[[BROADCAST]] [1]
    %0 = vector.multi_reduction <add>, %arg0, %acc [0] : vector<[4]xf32> to f32
    // INNER_REDUCTION: %[[EXTRACT:.+]] = vector.extract %[[RESULT]][0]
    // INNER_REDUCTION: return %[[EXTRACT]]
    return %0 : f32
}

// INNER_REDUCTION-LABEL: func @one_dim_to_two_dim_masked
// INNER_REDUCTION-SAME:    %[[INPUT:.+]]: vector<8xf32>
// INNER_REDUCTION-SAME:    %[[ACC:.+]]: f32
// INNER_REDUCTION-SAME:    %[[MASK:.+]]: vector<8xi1>
func.func @one_dim_to_two_dim_masked(%arg0: vector<8xf32>, %acc: f32, %mask: vector<8xi1>) -> f32 {
    // INNER_REDUCTION: %[[CAST:.+]] = vector.shape_cast %[[INPUT]] : vector<8xf32> to vector<1x8xf32>
    // INNER_REDUCTION: %[[BROADCAST_ACC:.+]] = vector.broadcast %[[ACC]] : f32 to vector<1xf32>
    // INNER_REDUCTION: %[[BROADCAST_MASK:.+]] = vector.broadcast %[[MASK]] : vector<8xi1> to vector<1x8xi1>
    // INNER_REDUCTION: %[[RESULT:.+]] = vector.mask %[[BROADCAST_MASK]] {
    // INNER_REDUCTION:   vector.multi_reduction <add>, %[[CAST]], %[[BROADCAST_ACC]] [1]
    %0 = vector.mask %mask {
      vector.multi_reduction <add>, %arg0, %acc [0] : vector<8xf32> to f32
    } : vector<8xi1> -> f32
    // INNER_REDUCTION: %[[EXTRACT:.+]] = vector.extract %[[RESULT]][0]
    // INNER_REDUCTION: return %[[EXTRACT]]
    return %0 : f32
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @innerreduction(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.reorder_and_expand_multi_reduction_dims lowering_strategy = "innerreduction"
    } : !transform.op<"func.func">
    transform.yield
  }

  transform.named_sequence @innerparallel(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.reorder_and_expand_multi_reduction_dims lowering_strategy = "innerparallel"
    } : !transform.op<"func.func">
    transform.yield
  }
}
