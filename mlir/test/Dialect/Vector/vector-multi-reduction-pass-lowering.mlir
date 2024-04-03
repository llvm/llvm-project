// RUN: mlir-opt -lower-vector-multi-reduction="lowering-strategy=inner-reduction" -split-input-file %s | FileCheck %s --check-prefix=CHECK-RED
// RUN: mlir-opt -lower-vector-multi-reduction="lowering-strategy=inner-parallel" -split-input-file %s | FileCheck %s --check-prefix=CHECK-PAR
// RUN: mlir-opt -lower-vector-multi-reduction -split-input-file %s | FileCheck %s --check-prefix=CHECK-PAR

func.func @vector_multi_reduction(%arg0: vector<2x4xf32>, %acc: vector<2xf32>) -> vector<2xf32> {
    %0 = vector.multi_reduction <mul>, %arg0, %acc [1] : vector<2x4xf32> to vector<2xf32>
    return %0 : vector<2xf32>
}
// CHECK-RED-LABEL: func @vector_multi_reduction
// CHECK-RED-SAME:   %[[INPUT:.+]]: vector<2x4xf32>, %[[ACC:.*]]: vector<2xf32>)
//  CHECK-RED-DAG:       %[[RESULT_VEC_0:.+]] = arith.constant dense<{{.*}}> : vector<2xf32>
//  CHECK-RED-DAG:       %[[C0:.+]] = arith.constant 0 : index
//  CHECK-RED-DAG:       %[[C1:.+]] = arith.constant 1 : index
//      CHECK-RED:       %[[V0:.+]] = vector.extract %[[INPUT]][0]
//      CHECK-RED:       %[[ACC0:.+]] = vector.extract %[[ACC]][0]
//      CHECK-RED:       %[[RV0:.+]] = vector.reduction <mul>, %[[V0]], %[[ACC0]] : vector<4xf32> into f32
//      CHECK-RED:       %[[RESULT_VEC_1:.+]] = vector.insertelement %[[RV0:.+]], %[[RESULT_VEC_0]][%[[C0]] : index] : vector<2xf32>
//      CHECK-RED:       %[[V1:.+]] = vector.extract %[[INPUT]][1]
//      CHECK-RED:       %[[ACC1:.+]] = vector.extract %[[ACC]][1]
//      CHECK-RED:       %[[RV1:.+]] = vector.reduction <mul>, %[[V1]], %[[ACC1]] : vector<4xf32> into f32
//      CHECK-RED:       %[[RESULT_VEC:.+]] = vector.insertelement %[[RV1:.+]], %[[RESULT_VEC_1]][%[[C1]] : index] : vector<2xf32>
//      CHECK-RED:       return %[[RESULT_VEC]]

// CHECK-PAR-LABEL: func @vector_multi_reduction
//  CHECK-PAR-SAME:   %[[INPUT:.+]]: vector<2x4xf32>, %[[ACC:.*]]: vector<2xf32>)
//       CHECK-PAR:   %[[TRANSPOSED:.+]] = vector.transpose %[[INPUT]], [1, 0] : vector<2x4xf32> to vector<4x2xf32>
//       CHECK-PAR:   %[[V0:.+]] = vector.extract %[[TRANSPOSED]][0] : vector<2xf32> from vector<4x2xf32>
//       CHECK-PAR:   %[[RV0:.+]] = arith.mulf %[[V0]], %[[ACC]] : vector<2xf32>
//       CHECK-PAR:   %[[V1:.+]] = vector.extract %[[TRANSPOSED]][1] : vector<2xf32> from vector<4x2xf32>
//       CHECK-PAR:   %[[RV01:.+]] = arith.mulf %[[V1]], %[[RV0]] : vector<2xf32>
//       CHECK-PAR:   %[[V2:.+]] = vector.extract %[[TRANSPOSED]][2] : vector<2xf32> from vector<4x2xf32>
//       CHECK-PAR:   %[[RV012:.+]] = arith.mulf %[[V2]], %[[RV01]] : vector<2xf32>
//       CHECK-PAR:   %[[V3:.+]] = vector.extract %[[TRANSPOSED]][3] : vector<2xf32> from vector<4x2xf32>
//       CHECK-PAR:   %[[RESULT_VEC:.+]] = arith.mulf %[[V3]], %[[RV012]] : vector<2xf32>
//       CHECK-PAR:   return %[[RESULT_VEC]] : vector<2xf32>

func.func @vector_multi_reduction_parallel_middle(%arg0: vector<3x4x5xf32>, %acc: vector<4xf32>) -> vector<4xf32> {
    %0 = vector.multi_reduction <add>, %arg0, %acc [0, 2] : vector<3x4x5xf32> to vector<4xf32>
    return %0 : vector<4xf32>
}

// CHECK-RED-LABEL: func @vector_multi_reduction_parallel_middle
//  CHECK-RED-SAME:   %[[INPUT:.+]]: vector<3x4x5xf32>, %[[ACC:.+]]: vector<4xf32>
//       CHECK-RED: vector.transpose %[[INPUT]], [1, 0, 2] : vector<3x4x5xf32> to vector<4x3x5xf32>

// CHECK-PAR-LABEL: func @vector_multi_reduction_parallel_middle
//  CHECK-PAR-SAME:   %[[INPUT:.+]]: vector<3x4x5xf32>, %[[ACC:.+]]: vector<4xf32>
//       CHECK-PAR: vector.transpose %[[INPUT]], [0, 2, 1] : vector<3x4x5xf32> to vector<3x5x4xf32>
