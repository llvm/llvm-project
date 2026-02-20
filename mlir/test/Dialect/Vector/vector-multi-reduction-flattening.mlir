// RUN: mlir-opt %s --transform-interpreter='entry-point=innerreduction' | FileCheck %s --check-prefix=INNER_REDUCTION,ALL
// RUN: mlir-opt %s --transform-interpreter='entry-point=innerparallel' | FileCheck %s --check-prefix=INNER_PARALLEL,ALL

// ALL-LABEL: func @negative_flattening_cases
func.func @negative_flattening_cases(
    %v1d: vector<8xf32>,
    %v2d: vector<4x8xf32>,
    %v_scalable: vector<[2]x[4]x8xf32>,
    %v_non_contig: vector<2x3x4x5xi32>,
    %acc_scalar: f32,
    %acc_1d: vector<8xf32>,
    %acc_2d: vector<2x4xi32>) -> (f32, vector<8xf32>, vector<8xf32>, vector<2x4xi32>) {

    // Test 1: Less than 2 dimensions
    // ALL: %[[R1:.+]] = vector.multi_reduction <add>, %{{.+}}, %{{.+}} [0] : vector<8xf32> to f32
    %r1 = vector.multi_reduction <add>, %v1d, %acc_scalar [0] : vector<8xf32> to f32

    // Test 2: More than one scalable dimensions
    // ALL: %[[R2:.+]] = vector.multi_reduction <mul>, %{{.+}}, %{{.+}} [0, 1] : vector<[2]x[4]x8xf32> to vector<8xf32>
    %r2 = vector.multi_reduction <mul>, %v_scalable, %acc_1d [0, 1] : vector<[2]x[4]x8xf32> to vector<8xf32>

    // Test 3: Already 2D with reduction on single dim
    // ALL: %[[R3:.+]] = vector.multi_reduction <add>, %{{.+}}, %{{.+}} [0] : vector<4x8xf32> to vector<8xf32>
    %r3 = vector.multi_reduction <add>, %v2d, %acc_1d [0] : vector<4x8xf32> to vector<8xf32>

    // Test 4: Non-contiguous parallel dimensions
    // ALL: %[[R4:.+]] = vector.multi_reduction <add>, %{{.+}}, %{{.+}} [1, 3] : vector<2x3x4x5xi32> to vector<2x4xi32>
    %r4 = vector.multi_reduction <add>, %v_non_contig, %acc_2d [1, 3] : vector<2x3x4x5xi32> to vector<2x4xi32>

    // ALL: return %[[R1]], %[[R2]], %[[R3]], %[[R4]]
    return %r1, %r2, %r3, %r4 : f32, vector<8xf32>, vector<8xf32>, vector<2x4xi32>
}

// ALL-LABEL: func @vector_multi_reduction_flattening
// ALL-SAME:   %[[INPUT:.+]]: vector<2x4xf32>, %[[ACC:.*]]: f32)
func.func @vector_multi_reduction_flattening(%arg0: vector<2x4xf32>, %acc: f32) -> f32 {
    // ALL: %[[CASTED:.*]] = vector.shape_cast %[[INPUT]] : vector<2x4xf32> to vector<8xf32>
    // ALL: %[[RESULT:.+]] = vector.multi_reduction <mul>, %[[CASTED]], %[[ACC]] [0]
    %0 = vector.multi_reduction <mul>, %arg0, %acc [0, 1] : vector<2x4xf32> to f32
    // ALL: return %[[RESULT]]
    return %0 : f32
}

// INNER_REDUCTION-LABEL: func @vector_multi_reduction_parallel_dim_innerreduction
// INNER_REDUCTION-SAME:    %[[INPUT:.+]]: vector<2x3x4xi32>
// INNER_REDUCTION-SAME:    %[[ACC:.+]]: vector<2xi32>
func.func @vector_multi_reduction_parallel_dim_innerreduction(%arg0: vector<2x3x4xi32>, %acc: vector<2xi32>) -> vector<2xi32> {
    // INNER_REDUCTION: %[[CASTED:.+]] = vector.shape_cast %[[INPUT]] : vector<2x3x4xi32> to vector<2x12xi32>
    // INNER_REDUCTION: %[[RESULT:.+]] = vector.multi_reduction <add>, %[[CASTED]], %[[ACC]] [1]
    %0 = vector.multi_reduction <add>, %arg0, %acc [1, 2] : vector<2x3x4xi32> to vector<2xi32>
    // INNER_REDUCTION: return %[[RESULT]]
    return %0 : vector<2xi32>
}

// INNER_REDUCTION-LABEL: func @output_shapecast_multiple_parallel
// INNER_REDUCTION-SAME:    %[[INPUT:.+]]: vector<2x3x4x5x6xi32>
// INNER_REDUCTION-SAME:    %[[ACC:.+]]: vector<2x3x4xi32>
func.func @output_shapecast_multiple_parallel(%arg0: vector<2x3x4x5x6xi32>, %acc: vector<2x3x4xi32>) -> vector<2x3x4xi32> {
    // INNER_REDUCTION: %[[INPUT_CAST:.+]] = vector.shape_cast %[[INPUT]] : vector<2x3x4x5x6xi32> to vector<24x30xi32>
    // INNER_REDUCTION: %[[ACC_CAST:.+]] = vector.shape_cast %[[ACC]] : vector<2x3x4xi32> to vector<24xi32>
    // INNER_REDUCTION: %[[RESULT_FLAT:.+]] = vector.multi_reduction <mul>, %[[INPUT_CAST]], %[[ACC_CAST]] [1]
    // INNER_REDUCTION: %[[RESULT:.+]] = vector.shape_cast %[[RESULT_FLAT]] : vector<24xi32> to vector<2x3x4xi32>
    %0 = vector.multi_reduction <mul>, %arg0, %acc [3, 4] : vector<2x3x4x5x6xi32> to vector<2x3x4xi32>
    // INNER_REDUCTION: return %[[RESULT]]
    return %0 : vector<2x3x4xi32>
}

// INNER_PARALLEL-LABEL: func @vector_multi_reduction_parallel_dim_innerparallel
// INNER_PARALLEL-SAME:    %[[INPUT:.+]]: vector<3x4x2xi32>
// INNER_PARALLEL-SAME:    %[[ACC:.+]]: vector<2xi32>
func.func @vector_multi_reduction_parallel_dim_innerparallel(%arg0: vector<3x4x2xi32>, %acc: vector<2xi32>) -> vector<2xi32> {
    // INNER_PARALLEL: %[[CASTED:.+]] = vector.shape_cast %[[INPUT]] : vector<3x4x2xi32> to vector<12x2xi32>
    // INNER_PARALLEL: %[[RESULT:.+]] = vector.multi_reduction <mul>, %[[CASTED]], %[[ACC]] [0]
    %0 = vector.multi_reduction <mul>, %arg0, %acc [0, 1] : vector<3x4x2xi32> to vector<2xi32>
    // INNER_PARALLEL: return %[[RESULT]]
    return %0 : vector<2xi32>
}

// ALL-LABEL: func @single_scalable_dim
// ALL-SAME:    %[[INPUT:.+]]: vector<4x[8]xf32>
// ALL-SAME:    %[[ACC:.+]]: f32
func.func @single_scalable_dim(%arg0: vector<4x[8]xf32>, %acc: f32) -> f32 {
    // ALL: %[[CASTED:.+]] = vector.shape_cast %[[INPUT]] : vector<4x[8]xf32> to vector<[32]xf32>
    // ALL: %[[RESULT:.+]] = vector.multi_reduction <add>, %[[CASTED]], %[[ACC]] [0]
    %0 = vector.multi_reduction <add>, %arg0, %acc [0, 1] : vector<4x[8]xf32> to f32
    // ALL: return %[[RESULT]]
    return %0 : f32
}

// ALL-LABEL: func @masked_multi_reduction
// ALL-SAME:    %[[INPUT:.+]]: vector<2x4xf32>
// ALL-SAME:    %[[ACC:.+]]: f32
// ALL-SAME:    %[[MASK:.+]]: vector<2x4xi1>
func.func @masked_multi_reduction(%arg0: vector<2x4xf32>, %acc: f32, %mask: vector<2x4xi1>) -> f32 {
    // ALL: %[[CASTED_MASK:.+]] = vector.shape_cast %[[MASK]] : vector<2x4xi1> to vector<8xi1>
    // ALL: %[[CASTED_INPUT:.+]] = vector.shape_cast %[[INPUT]] : vector<2x4xf32> to vector<8xf32>
    // ALL: %[[RESULT:.+]] = vector.mask %[[CASTED_MASK]]
    // ALL:   vector.multi_reduction <mul>, %[[CASTED_INPUT]], %[[ACC]] [0]
    %0 = vector.mask %mask {
      vector.multi_reduction <mul>, %arg0, %acc [0, 1] : vector<2x4xf32> to f32
    } : vector<2x4xi1> -> f32
    // ALL: return %[[RESULT]]
    return %0 : f32
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @innerreduction(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.multi_reduction_flattening lowering_strategy = "innerreduction"
    } : !transform.op<"func.func">
    transform.yield
  }

  transform.named_sequence @innerparallel(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.multi_reduction_flattening lowering_strategy = "innerparallel"
    } : !transform.op<"func.func">
    transform.yield
  }
}
