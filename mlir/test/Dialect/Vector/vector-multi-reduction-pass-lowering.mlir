// RUN: mlir-opt -lower-vector-multi-reduction="lowering-strategy=inner-reduction" -split-input-file %s | FileCheck %s --check-prefixes=ALL,INNER-REDUCTION
// RUN: mlir-opt -lower-vector-multi-reduction="lowering-strategy=inner-parallel" -split-input-file %s | FileCheck %s --check-prefixes=ALL,INNER-PARALLEL
// RUN: mlir-opt -lower-vector-multi-reduction -split-input-file %s | FileCheck %s --check-prefixes=ALL,INNER-PARALLEL

func.func @vector_multi_reduction(%arg0: vector<2x4xf32>, %acc: vector<2xf32>) -> vector<2xf32> {
    %0 = vector.multi_reduction <mul>, %arg0, %acc [1] : vector<2x4xf32> to vector<2xf32>
    return %0 : vector<2xf32>
}
//           ALL-LABEL: func @vector_multi_reduction
//            ALL-SAME: %[[INPUT:.+]]: vector<2x4xf32>, %[[ACC:.*]]: vector<2xf32>)
// INNER-REDUCTION-DAG: %[[RESULT_VEC_0:.+]] = arith.constant dense<{{.*}}> : vector<2xf32>
//     INNER-REDUCTION: %[[V0:.+]] = vector.extract %[[INPUT]][0]
//     INNER-REDUCTION: %[[ACC0:.+]] = vector.extract %[[ACC]][0]
//     INNER-REDUCTION: %[[RV0:.+]] = vector.reduction <mul>, %[[V0]], %[[ACC0]] : vector<4xf32> into f32
//     INNER-REDUCTION: %[[RESULT_VEC_1:.+]] = vector.insert %[[RV0:.+]], %[[RESULT_VEC_0]] [0] : f32 into vector<2xf32>
//     INNER-REDUCTION: %[[V1:.+]] = vector.extract %[[INPUT]][1]
//     INNER-REDUCTION: %[[ACC1:.+]] = vector.extract %[[ACC]][1]
//     INNER-REDUCTION: %[[RV1:.+]] = vector.reduction <mul>, %[[V1]], %[[ACC1]] : vector<4xf32> into f32
//     INNER-REDUCTION: %[[RESULT_VEC:.+]] = vector.insert %[[RV1:.+]], %[[RESULT_VEC_1]] [1] : f32 into vector<2xf32>
//     INNER-REDUCTION: return %[[RESULT_VEC]]

//      INNER-PARALLEL: %[[TRANSPOSED:.+]] = vector.transpose %[[INPUT]], [1, 0] : vector<2x4xf32> to vector<4x2xf32>
//      INNER-PARALLEL: %[[V0:.+]] = vector.extract %[[TRANSPOSED]][0] : vector<2xf32> from vector<4x2xf32>
//      INNER-PARALLEL: %[[RV0:.+]] = arith.mulf %[[V0]], %[[ACC]] : vector<2xf32>
//      INNER-PARALLEL: %[[V1:.+]] = vector.extract %[[TRANSPOSED]][1] : vector<2xf32> from vector<4x2xf32>
//      INNER-PARALLEL: %[[RV01:.+]] = arith.mulf %[[V1]], %[[RV0]] : vector<2xf32>
//      INNER-PARALLEL: %[[V2:.+]] = vector.extract %[[TRANSPOSED]][2] : vector<2xf32> from vector<4x2xf32>
//      INNER-PARALLEL: %[[RV012:.+]] = arith.mulf %[[V2]], %[[RV01]] : vector<2xf32>
//      INNER-PARALLEL: %[[V3:.+]] = vector.extract %[[TRANSPOSED]][3] : vector<2xf32> from vector<4x2xf32>
//      INNER-PARALLEL: %[[RESULT_VEC:.+]] = arith.mulf %[[V3]], %[[RV012]] : vector<2xf32>
//      INNER-PARALLEL: return %[[RESULT_VEC]] : vector<2xf32>

// -----

func.func @vector_multi_reduction_parallel_middle(%arg0: vector<3x4x5xf32>, %acc: vector<4xf32>) -> vector<4xf32> {
    %0 = vector.multi_reduction <add>, %arg0, %acc [0, 2] : vector<3x4x5xf32> to vector<4xf32>
    return %0 : vector<4xf32>
}

//       ALL-LABEL: func @vector_multi_reduction_parallel_middle
//        ALL-SAME: %[[INPUT:.+]]: vector<3x4x5xf32>, %[[ACC:.+]]: vector<4xf32>
// INNER-REDUCTION: vector.transpose %[[INPUT]], [1, 0, 2] : vector<3x4x5xf32> to vector<4x3x5xf32>
//  INNER-PARALLEL: vector.transpose %[[INPUT]], [0, 2, 1] : vector<3x4x5xf32> to vector<3x5x4xf32>
