// RUN: mlir-opt %s --transform-interpreter='entry-point=innerreduction' | FileCheck %s --check-prefixes=INNER_REDUCTION,ALL
// RUN: mlir-opt %s --transform-interpreter='entry-point=innerparallel' | FileCheck %s --check-prefixes=INNER_PARALLEL,ALL

// ALL-LABEL: func @negative_rank1_and_rank3
func.func @negative_rank1_and_rank3(
    %rank1: vector<8xf32>, %rank1_acc: f32,
    %rank3: vector<2x3x4xf32>, %rank3_acc: vector<2x3xf32>) -> (f32, vector<2x3xf32>) {
  // ALL: vector.multi_reduction <add>, {{.+}} [0] : vector<8xf32> to f32
  %0 = vector.multi_reduction <add>, %rank1, %rank1_acc [0] : vector<8xf32> to f32
  // ALL: vector.multi_reduction <add>, {{.+}} [2] : vector<2x3x4xf32> to vector<2x3xf32>
  %1 = vector.multi_reduction <add>, %rank3, %rank3_acc [2] : vector<2x3x4xf32> to vector<2x3xf32>
  return %0, %1 : f32, vector<2x3xf32>
}

// ALL-LABEL: func @inner_reduction_2d
// ALL-SAME:    %[[INPUT:.+]]: vector<2x4xf32>, %[[ACC:.+]]: vector<2xf32>
func.func @inner_reduction_2d(%arg0: vector<2x4xf32>, %acc: vector<2xf32>) -> vector<2xf32> {
    // INNER_REDUCTION:     %[[RESULT_VEC_0:.+]] = arith.constant dense<{{.+}}> : vector<2xf32>
    // INNER_REDUCTION:     %[[V0:.+]] = vector.extract %[[INPUT]][0]
    // INNER_REDUCTION:     %[[ACC0:.+]] = vector.extract %[[ACC]][0]
    // INNER_REDUCTION:     %[[RV0:.+]] = vector.reduction <mul>, %[[V0]], %[[ACC0]] : vector<4xf32> into f32
    // INNER_REDUCTION:     %[[RESULT_VEC_1:.+]] = vector.insert %[[RV0]], %[[RESULT_VEC_0]] [0] : f32 into vector<2xf32>
    // INNER_REDUCTION:     %[[V1:.+]] = vector.extract %[[INPUT]][1]
    // INNER_REDUCTION:     %[[ACC1:.+]] = vector.extract %[[ACC]][1]
    // INNER_REDUCTION:     %[[RV1:.+]] = vector.reduction <mul>, %[[V1]], %[[ACC1]] : vector<4xf32> into f32
    // INNER_REDUCTION:     %[[RESULT:.+]] = vector.insert %[[RV1]], %[[RESULT_VEC_1]] [1] : f32 into vector<2xf32>

    // INNER_PARALLEL:      %[[RESULT:.+]] = vector.multi_reduction <mul>, %[[INPUT]], %[[ACC]] [1]
    %0 = vector.multi_reduction <mul>, %arg0, %acc [1] : vector<2x4xf32> to vector<2xf32>
    // ALL:                 return %[[RESULT]]
    return %0 : vector<2xf32>
}

// ALL-LABEL: func @inner_reduction_2d_scalable
// ALL-SAME:    %[[INPUT:.+]]: vector<2x[4]xf32>
// ALL-SAME:    %[[ACC:.+]]: vector<2xf32>
// ALL-SAME:    %[[MASK:.+]]: vector<2x[4]xi1>
func.func @inner_reduction_2d_scalable(%input: vector<2x[4]xf32>, %acc: vector<2xf32>, %mask: vector<2x[4]xi1>) -> vector<2xf32> {
    // INNER_REDUCTION: %[[INIT:.+]] = arith.constant dense<0.000000e+00> : vector<2xf32>
    // INNER_REDUCTION: %[[INPUT_0:.+]] = vector.extract %[[INPUT]][0] : vector<[4]xf32> from vector<2x[4]xf32>
    // INNER_REDUCTION: %[[ACC_0:.+]] = vector.extract %[[ACC]][0] : f32 from vector<2xf32>
    // INNER_REDUCTION: %[[MASK_0:.+]] = vector.extract %[[MASK]][0] : vector<[4]xi1> from vector<2x[4]xi1>
    // INNER_REDUCTION: %[[REDUCE_0:.+]] = vector.mask %[[MASK_0]] { vector.reduction <add>, %[[INPUT_0]], %[[ACC_0]] : vector<[4]xf32> into f32 } : vector<[4]xi1> -> f32
    // INNER_REDUCTION: %[[INSERT_0:.+]] = vector.insert %[[REDUCE_0]], %[[INIT]] [0] : f32 into vector<2xf32>
    // INNER_REDUCTION: %[[INPUT_1:.+]] = vector.extract %[[INPUT]][1] : vector<[4]xf32> from vector<2x[4]xf32>
    // INNER_REDUCTION: %[[ACC_1:.+]] = vector.extract %[[ACC]][1] : f32 from vector<2xf32>
    // INNER_REDUCTION: %[[MASK_1:.+]] = vector.extract %[[MASK]][1] : vector<[4]xi1> from vector<2x[4]xi1>
    // INNER_REDUCTION: %[[REDUCE_1:.+]] = vector.mask %[[MASK_1]] { vector.reduction <add>, %[[INPUT_1]], %[[ACC_1]] : vector<[4]xf32> into f32 } : vector<[4]xi1> -> f32
    // INNER_REDUCTION: %[[RESULT:.+]] = vector.insert %[[REDUCE_1]], %[[INSERT_0]] [1] : f32 into vector<2xf32>

    // INNER_PARALLEL:  %[[RESULT:.+]] = vector.mask %[[MASK]] { vector.multi_reduction <add>, %[[INPUT]], %[[ACC]] [1] {{.+}} } : vector<2x[4]xi1> -> vector<2xf32>
    // ALL:             return %[[RESULT]] : vector<2xf32>
    %0 = vector.mask %mask { vector.multi_reduction <add>, %input, %acc [1] : vector<2x[4]xf32> to vector<2xf32> } : vector<2x[4]xi1> -> vector<2xf32>
    return %0 : vector<2xf32>
}

// ALL-LABEL: func @inner_parallel_base
// ALL-SAME:    %[[INPUT:.+]]: vector<4x2xf32>, %[[ACC:.+]]: vector<2xf32>
func.func @inner_parallel_base(%arg0: vector<4x2xf32>, %acc: vector<2xf32>) -> vector<2xf32> {
    // INNER_PARALLEL: %[[V0:.+]] = vector.extract %[[INPUT]][0] : vector<2xf32> from vector<4x2xf32>
    // INNER_PARALLEL: %[[V1:.+]] = vector.extract %[[INPUT]][1] : vector<2xf32> from vector<4x2xf32>
    // INNER_PARALLEL: %[[V2:.+]] = vector.extract %[[INPUT]][2] : vector<2xf32> from vector<4x2xf32>
    // INNER_PARALLEL: %[[V3:.+]] = vector.extract %[[INPUT]][3] : vector<2xf32> from vector<4x2xf32>
    // INNER_PARALLEL: %[[RV0:.+]] = arith.mulf %[[V0]], %[[ACC]] : vector<2xf32>
    // INNER_PARALLEL: %[[RV1:.+]] = arith.mulf %[[V1]], %[[RV0]] : vector<2xf32>
    // INNER_PARALLEL: %[[RV2:.+]] = arith.mulf %[[V2]], %[[RV1]] : vector<2xf32>
    // INNER_PARALLEL: %[[RESULT:.+]] = arith.mulf %[[V3]], %[[RV2]] : vector<2xf32>
    // INNER_REDUCTION: %[[RESULT:.+]] = vector.multi_reduction <mul>, %[[INPUT]], %[[ACC]] [0]
    // ALL:             return %[[RESULT]] : vector<2xf32>
    %0 = vector.multi_reduction <mul>, %arg0, %acc [0] : vector<4x2xf32> to vector<2xf32>
    return %0 : vector<2xf32>
}

// ALL-LABEL: func @inner_parallel_general
// ALL-SAME:    %[[INPUT:.+]]: vector<4x2x3xf32>, %[[ACC:.+]]: vector<2xf32>
func.func @inner_parallel_general(%arg0: vector<4x2x3xf32>, %acc: vector<2xf32>) -> vector<2xf32> {
    // INNER_PARALLEL: %[[V0:.+]] = vector.extract %[[INPUT]][0] : vector<2x3xf32> from vector<4x2x3xf32>
    // INNER_PARALLEL: %[[V1:.+]] = vector.extract %[[INPUT]][1] : vector<2x3xf32> from vector<4x2x3xf32>
    // INNER_PARALLEL: %[[V2:.+]] = vector.extract %[[INPUT]][2] : vector<2x3xf32> from vector<4x2x3xf32>
    // INNER_PARALLEL: %[[V3:.+]] = vector.extract %[[INPUT]][3] : vector<2x3xf32> from vector<4x2x3xf32>
    // INNER_PARALLEL: %[[RV0:.+]] = vector.multi_reduction <mul>, %[[V0]], %[[ACC]] [1] : vector<2x3xf32>
    // INNER_PARALLEL: %[[RV1:.+]] = vector.multi_reduction <mul>, %[[V1]], %[[RV0]] [1] : vector<2x3xf32>
    // INNER_PARALLEL: %[[RV2:.+]] = vector.multi_reduction <mul>, %[[V2]], %[[RV1]] [1] : vector<2x3xf32>
    // INNER_PARALLEL: %[[RESULT:.+]] = vector.multi_reduction <mul>, %[[V3]], %[[RV2]] [1] : vector<2x3xf32>
    // INNER_REDUCTION: %[[RESULT:.+]] = vector.multi_reduction <mul>, %[[INPUT]], %[[ACC]] [0, 2]
    %0 = vector.multi_reduction <mul>, %arg0, %acc [0, 2] : vector<4x2x3xf32> to vector<2xf32>
    // ALL:             return %[[RESULT]]
    return %0 : vector<2xf32>
}

// ALL-LABEL: func @inner_parallel_base_masked
// ALL-SAME:    %[[INPUT:.+]]: vector<4x2xf32>, %[[ACC:.+]]: vector<2xf32>, %[[MASK:.+]]: vector<4x2xi1>
func.func @inner_parallel_base_masked(%arg0: vector<4x2xf32>, %acc: vector<2xf32>, %mask: vector<4x2xi1>) -> vector<2xf32> {
    // INNER_PARALLEL: %[[V0:.+]] = vector.extract %[[INPUT]][0] : vector<2xf32> from vector<4x2xf32>
    // INNER_PARALLEL: %[[V1:.+]] = vector.extract %[[INPUT]][1] : vector<2xf32> from vector<4x2xf32>
    // INNER_PARALLEL: %[[V2:.+]] = vector.extract %[[INPUT]][2] : vector<2xf32> from vector<4x2xf32>
    // INNER_PARALLEL: %[[V3:.+]] = vector.extract %[[INPUT]][3] : vector<2xf32> from vector<4x2xf32>

    // INNER_PARALLEL: %[[M0:.+]] = vector.extract %[[MASK]][0] : vector<2xi1> from vector<4x2xi1>
    // INNER_PARALLEL: %[[M1:.+]] = vector.extract %[[MASK]][1] : vector<2xi1> from vector<4x2xi1>
    // INNER_PARALLEL: %[[M2:.+]] = vector.extract %[[MASK]][2] : vector<2xi1> from vector<4x2xi1>
    // INNER_PARALLEL: %[[M3:.+]] = vector.extract %[[MASK]][3] : vector<2xi1> from vector<4x2xi1>

    // INNER_PARALLEL: %[[RED0:.+]] = arith.mulf %[[V0]], %[[ACC]] : vector<2xf32>
    // INNER_PARALLEL: %[[RV0:.+]] = arith.select %[[M0]], %[[RED0]], %[[ACC]] : vector<2xi1>, vector<2xf32>
    // INNER_PARALLEL: %[[RED1:.+]] = arith.mulf %[[V1]], %[[RV0]] : vector<2xf32>
    // INNER_PARALLEL: %[[RV1:.+]] = arith.select %[[M1]], %[[RED1]], %[[RV0]] : vector<2xi1>, vector<2xf32>
    // INNER_PARALLEL: %[[RED2:.+]] = arith.mulf %[[V2]], %[[RV1]] : vector<2xf32>
    // INNER_PARALLEL: %[[RV2:.+]] = arith.select %[[M2]], %[[RED2]], %[[RV1]] : vector<2xi1>, vector<2xf32>
    // INNER_PARALLEL: %[[RED3:.+]] = arith.mulf %[[V3]], %[[RV2]] : vector<2xf32>
    // INNER_PARALLEL: %[[RESULT:.+]] = arith.select %[[M3]], %[[RED3]], %[[RV2]] : vector<2xi1>, vector<2xf32>
    // INNER_REDUCTION: %[[RESULT:.+]] = vector.mask %[[MASK]] { vector.multi_reduction <mul>, %[[INPUT]], %[[ACC]] [0] {{.+}} } : vector<4x2xi1> -> vector<2xf32>
    // ALL:             return %[[RESULT]] : vector<2xf32>
    %0 = vector.mask %mask { vector.multi_reduction <mul>, %arg0, %acc [0] : vector<4x2xf32> to vector<2xf32> } : vector<4x2xi1> -> vector<2xf32>
    return %0 : vector<2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @innerreduction(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.multi_reduction_unrolling lowering_strategy = "innerreduction"
    } : !transform.op<"func.func">
    transform.yield
  }

  transform.named_sequence @innerparallel(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.multi_reduction_unrolling lowering_strategy = "innerparallel"
    } : !transform.op<"func.func">
    transform.yield
  }
}
