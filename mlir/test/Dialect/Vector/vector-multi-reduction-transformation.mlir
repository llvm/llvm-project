// RUN: mlir-opt %s --transform-interpreter='entry-point=innerreduction' | FileCheck %s --check-prefix=INNERREDUCTION
// RUN: mlir-opt %s --transform-interpreter='entry-point=innerparallel' | FileCheck %s --check-prefix=INNERPARALLEL

// INNERREDUCTION-LABEL: func @transpose_reduction_dims_innerreduction
// INNERREDUCTION-SAME:    %[[INPUT:.+]]: vector<3x2x4xf32>
// INNERREDUCTION-SAME:    %[[ACC:.+]]: vector<2x4xf32>
func.func @transpose_reduction_dims_innerreduction(%arg0: vector<3x2x4xf32>, %acc: vector<2x4xf32>) -> vector<2x4xf32> {
    // INNERREDUCTION: %[[TRANSPOSED:.+]] = vector.transpose %[[INPUT]], [1, 2, 0]
    // INNERREDUCTION: %[[RESULT:.+]] = vector.multi_reduction <mul>, %[[TRANSPOSED]], %[[ACC]] [2]
    %0 = vector.multi_reduction <mul>, %arg0, %acc [0] : vector<3x2x4xf32> to vector<2x4xf32>
    // INNERREDUCTION: return %[[RESULT]]
    return %0 : vector<2x4xf32>
}

// INNERPARALLEL-LABEL: func @transpose_reduction_dims_innerparallel
// INNERPARALLEL-SAME:    %[[INPUT:.+]]: vector<3x2x4xf32>
// INNERPARALLEL-SAME:    %[[ACC:.+]]: vector<3x2xf32>
func.func @transpose_reduction_dims_innerparallel(%arg0: vector<3x2x4xf32>, %acc: vector<3x2xf32>) -> vector<3x2xf32> {
    // INNERPARALLEL: %[[TRANSPOSED:.+]] = vector.transpose %[[INPUT]], [2, 0, 1]
    // INNERPARALLEL: vector.multi_reduction <mul>, %[[TRANSPOSED]], %[[ACC]] [0]
    %0 = vector.multi_reduction <mul>, %arg0, %acc [2] : vector<3x2x4xf32> to vector<3x2xf32>
    return %0 : vector<3x2xf32>
}

// INNERREDUCTION-LABEL: func @transpose_multi_reduction_dims
// INNERREDUCTION-SAME:    %[[INPUT:.+]]: vector<2x3x4x5xf32>
// INNERREDUCTION-SAME:    %[[ACC:.+]]: vector<2x5xf32>
func.func @transpose_multi_reduction_dims(%arg0: vector<2x3x4x5xf32>, %acc: vector<2x5xf32>) -> vector<2x5xf32> {
    // INNERREDUCTION: %[[TRANSPOSED:.+]] = vector.transpose %[[INPUT]], [0, 3, 1, 2]
    // INNERREDUCTION: %[[RESULT:.+]] = vector.multi_reduction <add>, %[[TRANSPOSED]], %[[ACC]] [2, 3]
    %0 = vector.multi_reduction <add>, %arg0, %acc [1, 2] : vector<2x3x4x5xf32> to vector<2x5xf32>
    // INNERREDUCTION: return %[[RESULT]]
    return %0 : vector<2x5xf32>
}

// INNERREDUCTION-LABEL: func @transpose_parallel_middle
// INNERREDUCTION-SAME:    %[[INPUT:.+]]: vector<3x4x5xf32>
// INNERREDUCTION-SAME:    %[[ACC:.+]]: vector<4xf32>
// INNERPARALLEL-LABEL: func @transpose_parallel_middle
// INNERPARALLEL-SAME:    %[[INPUT:.+]]: vector<3x4x5xf32>
// INNERPARALLEL-SAME:    %[[ACC:.+]]: vector<4xf32>
func.func @transpose_parallel_middle(%arg0: vector<3x4x5xf32>, %acc: vector<4xf32>) -> vector<4xf32> {
    // INNERREDUCTION: %[[TRANSPOSED:.+]] = vector.transpose %[[INPUT]], [1, 0, 2]
    // INNERREDUCTION: %[[RESULT:.+]] = vector.multi_reduction <add>, %[[TRANSPOSED]], %[[ACC]] [1, 2]
    // INNERPARALLEL: %[[TRANSPOSED:.+]] = vector.transpose %[[INPUT]], [0, 2, 1]
    // INNERPARALLEL: vector.multi_reduction <add>, %[[TRANSPOSED]], %[[ACC]] [0, 1]
    %0 = vector.multi_reduction <add>, %arg0, %acc [0, 2] : vector<3x4x5xf32> to vector<4xf32>
    // INNERREDUCTION: return %[[RESULT]]
    return %0 : vector<4xf32>
}

// INNERREDUCTION-LABEL: func @one_dim_to_two_dim_innerreduction
// INNERREDUCTION-SAME:    %[[INPUT:.+]]: vector<8xf32>
// INNERREDUCTION-SAME:    %[[ACC:.+]]: f32
func.func @one_dim_to_two_dim_innerreduction(%arg0: vector<8xf32>, %acc: f32) -> f32 {
    // INNERREDUCTION: %[[CAST:.+]] = vector.shape_cast %[[INPUT]] : vector<8xf32> to vector<1x8xf32>
    // INNERREDUCTION: %[[BROADCAST:.+]] = vector.broadcast %[[ACC]] : f32 to vector<1xf32>
    // INNERREDUCTION: %[[RESULT:.+]] = vector.multi_reduction <add>, %[[CAST]], %[[BROADCAST]] [1]
    %0 = vector.multi_reduction <add>, %arg0, %acc [0] : vector<8xf32> to f32
    // INNERREDUCTION: %[[EXTRACT:.+]] = vector.extract %[[RESULT]][0]
    // INNERREDUCTION: return %[[EXTRACT]]
    return %0 : f32
}

// INNERPARALLEL-LABEL: func @one_dim_to_two_dim_innerparallel
// INNERPARALLEL-SAME:    %[[INPUT:.+]]: vector<2xf32>
// INNERPARALLEL-SAME:    %[[ACC:.+]]: f32
func.func @one_dim_to_two_dim_innerparallel(%arg0: vector<2xf32>, %acc: f32) -> f32 {
    // INNERPARALLEL: %[[CAST:.+]] = vector.shape_cast %[[INPUT]] : vector<2xf32> to vector<1x2xf32>
    // INNERPARALLEL: %[[BROADCAST:.+]] = vector.broadcast %[[ACC]] : f32 to vector<1xf32>
    // INNERPARALLEL: %[[TRANSPOSED:.+]] = vector.transpose %[[CAST]], [1, 0]
    // INNERPARALLEL: vector.multi_reduction <maxnumf>, %[[TRANSPOSED]], %[[BROADCAST]] [0]
    %0 = vector.multi_reduction <maxnumf>, %arg0, %acc [0] : vector<2xf32> to f32
    return %0 : f32
}

// INNERREDUCTION-LABEL: func @one_dim_to_two_dim_scalable
// INNERREDUCTION-SAME:    %[[INPUT:.+]]: vector<[4]xf32>
// INNERREDUCTION-SAME:    %[[ACC:.+]]: f32
func.func @one_dim_to_two_dim_scalable(%arg0: vector<[4]xf32>, %acc: f32) -> f32 {
    // INNERREDUCTION: %[[CAST:.+]] = vector.shape_cast %[[INPUT]] : vector<[4]xf32> to vector<1x[4]xf32>
    // INNERREDUCTION: %[[BROADCAST:.+]] = vector.broadcast %[[ACC]] : f32 to vector<1xf32>
    // INNERREDUCTION: %[[RESULT:.+]] = vector.multi_reduction <add>, %[[CAST]], %[[BROADCAST]] [1]
    %0 = vector.multi_reduction <add>, %arg0, %acc [0] : vector<[4]xf32> to f32
    // INNERREDUCTION: %[[EXTRACT:.+]] = vector.extract %[[RESULT]][0]
    // INNERREDUCTION: return %[[EXTRACT]]
    return %0 : f32
}

// INNERREDUCTION-LABEL: func @one_dim_to_two_dim_masked
// INNERREDUCTION-SAME:    %[[INPUT:.+]]: vector<8xf32>
// INNERREDUCTION-SAME:    %[[ACC:.+]]: f32
// INNERREDUCTION-SAME:    %[[MASK:.+]]: vector<8xi1>
func.func @one_dim_to_two_dim_masked(%arg0: vector<8xf32>, %acc: f32, %mask: vector<8xi1>) -> f32 {
    // INNERREDUCTION: %[[CAST:.+]] = vector.shape_cast %[[INPUT]] : vector<8xf32> to vector<1x8xf32>
    // INNERREDUCTION: %[[BROADCAST_ACC:.+]] = vector.broadcast %[[ACC]] : f32 to vector<1xf32>
    // INNERREDUCTION: %[[BROADCAST_MASK:.+]] = vector.broadcast %[[MASK]] : vector<8xi1> to vector<1x8xi1>
    // INNERREDUCTION: %[[RESULT:.+]] = vector.mask %[[BROADCAST_MASK]] {
    // INNERREDUCTION:   vector.multi_reduction <add>, %[[CAST]], %[[BROADCAST_ACC]] [1]
    %0 = vector.mask %mask {
      vector.multi_reduction <add>, %arg0, %acc [0] : vector<8xf32> to f32
    } : vector<8xi1> -> f32
    // INNERREDUCTION: %[[EXTRACT:.+]] = vector.extract %[[RESULT]][0]
    // INNERREDUCTION: return %[[EXTRACT]]
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
