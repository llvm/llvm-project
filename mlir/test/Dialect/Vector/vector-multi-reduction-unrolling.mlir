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

func.func @inner_reduction_2d_masked_dynamic(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %c1 = arith.constant 1 : index
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %c0_1 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = vector.create_mask %dim, %dim_0 : vector<4x8xi1>
  %1 = vector.mask %0 { vector.transfer_read %arg0[%c0_1, %c0_1], %cst {in_bounds = [true, true]} : tensor<?x?xf32>, vector<4x8xf32> } : vector<4x8xi1> -> vector<4x8xf32>
  %cst_2 = arith.constant 0.000000e+00 : f32
  %2 = vector.create_mask %dim : vector<4xi1>
  %3 = vector.mask %2 { vector.transfer_read %arg1[%c0_1], %cst_2 {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
  %4 = vector.mask %0 { vector.multi_reduction <add>, %1, %3 [1] : vector<4x8xf32> to vector<4xf32> } : vector<4x8xi1> -> vector<4xf32>
  %c0_3 = arith.constant 0 : index
  %5 = vector.mask %2 { vector.transfer_write %4, %arg1[%c0_3] {in_bounds = [true]} : vector<4xf32>, tensor<?xf32> } : vector<4xi1> -> tensor<?xf32>
  return %5 : tensor<?xf32>
}

// ALL-LABEL: func @inner_reduction_2d_masked_dynamic
// INNER_REDUCTION:           %[[DIM_0:.+]] = tensor.dim
// INNER_REDUCTION:           %[[DIM_1:.+]] = tensor.dim
// INNER_REDUCTION:           %[[MASK_2D:.+]] = vector.create_mask %[[DIM_0]], %[[DIM_1]] : vector<4x8xi1>
//
// INNER_REDUCTION:           %[[MASK_SLICE_0:.+]] = vector.extract %[[MASK_2D]][0] : vector<8xi1> from vector<4x8xi1>
// INNER_REDUCTION:           %[[REDUCE_0:.+]] = vector.mask %[[MASK_SLICE_0]] { vector.reduction <add>, %{{.+}} : vector<8xf32> into f32 } : vector<8xi1> -> f32
// INNER_REDUCTION:           %[[INSERT_0:.+]] = vector.insert
//
// INNER_REDUCTION:           %[[MASK_SLICE_1:.+]] = vector.extract %[[MASK_2D]][1] : vector<8xi1> from vector<4x8xi1>
// INNER_REDUCTION:           %[[REDUCE_1:.+]] = vector.mask %[[MASK_SLICE_1]] { vector.reduction <add>, %{{.+}} : vector<8xf32> into f32 } : vector<8xi1> -> f32
// INNER_REDUCTION:           %[[INSERT_1:.+]] = vector.insert
//
// INNER_REDUCTION:           %[[MASK_SLICE_2:.+]] = vector.extract %[[MASK_2D]][2] : vector<8xi1> from vector<4x8xi1>
// INNER_REDUCTION:           %[[REDUCE_2:.+]] = vector.mask %[[MASK_SLICE_2]] { vector.reduction <add>, %{{.+}} : vector<8xf32> into f32 } : vector<8xi1> -> f32
// INNER_REDUCTION:           %[[INSERT_2:.+]] = vector.insert
//
// INNER_REDUCTION:           %[[MASK_SLICE_3:.+]] = vector.extract %[[MASK_2D]][3] : vector<8xi1> from vector<4x8xi1>
// INNER_REDUCTION:           %[[REDUCE_3:.+]] = vector.mask %[[MASK_SLICE_3]] { vector.reduction <add>, %{{.+}} : vector<8xf32> into f32 } : vector<8xi1> -> f32
// INNER_REDUCTION:           %[[INSERT_3:.+]] = vector.insert
//
// INNER_PARALLEL:            vector.multi_reduction <add>

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

// ALL-LABEL: func @inner_parallel_2d
// ALL-SAME:    %[[INPUT:.+]]: vector<4x2xf32>, %[[ACC:.+]]: vector<2xf32>
func.func @inner_parallel_2d(%arg0: vector<4x2xf32>, %acc: vector<2xf32>) -> vector<2xf32> {
    // INNER_PARALLEL: %[[V0:.+]] = vector.extract %[[INPUT]][0] : vector<2xf32> from vector<4x2xf32>
    // INNER_PARALLEL: %[[RV0:.+]] = arith.mulf %[[V0]], %[[ACC]] : vector<2xf32>
    // INNER_PARALLEL: %[[V1:.+]] = vector.extract %[[INPUT]][1] : vector<2xf32> from vector<4x2xf32>
    // INNER_PARALLEL: %[[RV1:.+]] = arith.mulf %[[V1]], %[[RV0]] : vector<2xf32>
    // INNER_PARALLEL: %[[V2:.+]] = vector.extract %[[INPUT]][2] : vector<2xf32> from vector<4x2xf32>
    // INNER_PARALLEL: %[[RV2:.+]] = arith.mulf %[[V2]], %[[RV1]] : vector<2xf32>
    // INNER_PARALLEL: %[[V3:.+]] = vector.extract %[[INPUT]][3] : vector<2xf32> from vector<4x2xf32>
    // INNER_PARALLEL: %[[RESULT:.+]] = arith.mulf %[[V3]], %[[RV2]] : vector<2xf32>
    // INNER_REDUCTION: %[[RESULT:.+]] = vector.multi_reduction <mul>, %[[INPUT]], %[[ACC]] [0]
    // ALL:             return %[[RESULT]] : vector<2xf32>
    %0 = vector.multi_reduction <mul>, %arg0, %acc [0] : vector<4x2xf32> to vector<2xf32>
    return %0 : vector<2xf32>
}

// ALL-LABEL: func @inner_parallel_2d_masked
// ALL-SAME:    %[[INPUT:.+]]: vector<4x2xf32>, %[[ACC:.+]]: vector<2xf32>, %[[MASK:.+]]: vector<4x2xi1>
func.func @inner_parallel_2d_masked(%arg0: vector<4x2xf32>, %acc: vector<2xf32>, %mask: vector<4x2xi1>) -> vector<2xf32> {
    // INNER_PARALLEL: %[[V0:.+]] = vector.extract %[[INPUT]][0] : vector<2xf32> from vector<4x2xf32>
    // INNER_PARALLEL: %[[M0:.+]] = vector.extract %[[MASK]][0] : vector<2xi1> from vector<4x2xi1>
    // INNER_PARALLEL: %[[RED0:.+]] = arith.mulf %[[V0]], %[[ACC]] : vector<2xf32>
    // INNER_PARALLEL: %[[RV0:.+]] = arith.select %[[M0]], %[[RED0]], %[[ACC]] : vector<2xi1>, vector<2xf32>
    // INNER_PARALLEL: %[[V1:.+]] = vector.extract %[[INPUT]][1] : vector<2xf32> from vector<4x2xf32>
    // INNER_PARALLEL: %[[M1:.+]] = vector.extract %[[MASK]][1] : vector<2xi1> from vector<4x2xi1>
    // INNER_PARALLEL: %[[RED1:.+]] = arith.mulf %[[V1]], %[[RV0]] : vector<2xf32>
    // INNER_PARALLEL: %[[RV1:.+]] = arith.select %[[M1]], %[[RED1]], %[[RV0]] : vector<2xi1>, vector<2xf32>
    // INNER_PARALLEL: %[[V2:.+]] = vector.extract %[[INPUT]][2] : vector<2xf32> from vector<4x2xf32>
    // INNER_PARALLEL: %[[M2:.+]] = vector.extract %[[MASK]][2] : vector<2xi1> from vector<4x2xi1>
    // INNER_PARALLEL: %[[RED2:.+]] = arith.mulf %[[V2]], %[[RV1]] : vector<2xf32>
    // INNER_PARALLEL: %[[RV2:.+]] = arith.select %[[M2]], %[[RED2]], %[[RV1]] : vector<2xi1>, vector<2xf32>
    // INNER_PARALLEL: %[[V3:.+]] = vector.extract %[[INPUT]][3] : vector<2xf32> from vector<4x2xf32>
    // INNER_PARALLEL: %[[M3:.+]] = vector.extract %[[MASK]][3] : vector<2xi1> from vector<4x2xi1>
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
