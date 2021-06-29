// RUN: mlir-opt %s -test-vector-multi-reduction-lowering-patterns | FileCheck %s

func @vector_multi_reduction(%arg0: vector<2x4xf32>) -> vector<2xf32> {
    %0 = vector.multi_reduction #vector.kind<mul>, %arg0 [1] : vector<2x4xf32> to vector<2xf32>
    return %0 : vector<2xf32>
}
// CHECK-LABEL: func @vector_multi_reduction
//  CHECK-SAME:   %[[INPUT:.+]]: vector<2x4xf32>
//       CHECK:       %[[RESULT_VEC_0:.+]] = constant dense<{{.*}}> : vector<2xf32>
//       CHECK:       %[[C0:.+]] = constant 0 : i32
//       CHECK:       %[[C1:.+]] = constant 1 : i32
//       CHECK:       %[[V0:.+]] = vector.extract %[[INPUT]][0]
//       CHECK:       %[[RV0:.+]] = vector.reduction "mul", %[[V0]] : vector<4xf32> into f32
//       CHECK:       %[[RESULT_VEC_1:.+]] = vector.insertelement %[[RV0:.+]], %[[RESULT_VEC_0]][%[[C0]] : i32] : vector<2xf32>
//       CHECK:       %[[V1:.+]] = vector.extract %[[INPUT]][1]
//       CHECK:       %[[RV1:.+]] = vector.reduction "mul", %[[V1]] : vector<4xf32> into f32
//       CHECK:       %[[RESULT_VEC:.+]] = vector.insertelement %[[RV1:.+]], %[[RESULT_VEC_1]][%[[C1]] : i32] : vector<2xf32>
//       CHECK:       return %[[RESULT_VEC]]

func @vector_reduction_inner(%arg0: vector<2x3x4x5xi32>) -> vector<2x3xi32> {
    %0 = vector.multi_reduction #vector.kind<add>, %arg0 [2, 3] : vector<2x3x4x5xi32> to vector<2x3xi32>
    return %0 : vector<2x3xi32>
}
// CHECK-LABEL: func @vector_reduction_inner
//  CHECK-SAME:   %[[INPUT:.+]]: vector<2x3x4x5xi32>
//       CHECK:       %[[FLAT_RESULT_VEC_0:.+]] = constant dense<0> : vector<6xi32>
//   CHECK-DAG:       %[[C0:.+]] = constant 0 : i32
//   CHECK-DAG:       %[[C1:.+]] = constant 1 : i32
//   CHECK-DAG:       %[[C2:.+]] = constant 2 : i32
//   CHECK-DAG:       %[[C3:.+]] = constant 3 : i32
//   CHECK-DAG:       %[[C4:.+]] = constant 4 : i32
//   CHECK-DAG:       %[[C5:.+]] = constant 5 : i32
//       CHECK:       %[[RESHAPED_INPUT:.+]] = vector.shape_cast %[[INPUT]] : vector<2x3x4x5xi32> to vector<6x20xi32>
//       CHECK:       %[[V0:.+]] = vector.extract %[[RESHAPED_INPUT]][0] : vector<6x20xi32>
//       CHECK:       %[[V0R:.+]] = vector.reduction "add", %[[V0]] : vector<20xi32> into i32
//       CHECK:       %[[FLAT_RESULT_VEC_1:.+]] = vector.insertelement %[[V0R]], %[[FLAT_RESULT_VEC_0]][%[[C0]] : i32] : vector<6xi32>
//       CHECK:       %[[V1:.+]] = vector.extract %[[RESHAPED_INPUT]][1] : vector<6x20xi32>
//       CHECK:       %[[V1R:.+]] = vector.reduction "add", %[[V1]] : vector<20xi32> into i32
//       CHECK:       %[[FLAT_RESULT_VEC_2:.+]] = vector.insertelement %[[V1R]], %[[FLAT_RESULT_VEC_1]][%[[C1]] : i32] : vector<6xi32>
//       CHECK:       %[[V2:.+]] = vector.extract %[[RESHAPED_INPUT]][2] : vector<6x20xi32>
//       CHECK:       %[[V2R:.+]] = vector.reduction "add", %[[V2]] : vector<20xi32> into i32
//       CHECK:       %[[FLAT_RESULT_VEC_3:.+]] = vector.insertelement %[[V2R]], %[[FLAT_RESULT_VEC_2]][%[[C2]] : i32] : vector<6xi32>
//       CHECK:       %[[V3:.+]] = vector.extract %[[RESHAPED_INPUT]][3] : vector<6x20xi32>
//       CHECK:       %[[V3R:.+]] = vector.reduction "add", %[[V3]] : vector<20xi32> into i32
//       CHECK:       %[[FLAT_RESULT_VEC_4:.+]] = vector.insertelement %[[V3R]], %[[FLAT_RESULT_VEC_3]][%[[C3]] : i32] : vector<6xi32>
//       CHECK:       %[[V4:.+]] = vector.extract %[[RESHAPED_INPUT]][4] : vector<6x20xi32>
//       CHECK:       %[[V4R:.+]] = vector.reduction "add", %[[V4]] : vector<20xi32> into i32
//       CHECK:       %[[FLAT_RESULT_VEC_5:.+]] = vector.insertelement %[[V4R]], %[[FLAT_RESULT_VEC_4]][%[[C4]] : i32] : vector<6xi32>
///       CHECK:      %[[V5:.+]] = vector.extract %[[RESHAPED_INPUT]][5] : vector<6x20xi32>
//       CHECK:       %[[V5R:.+]] = vector.reduction "add", %[[V5]] : vector<20xi32> into i32
//       CHECK:       %[[FLAT_RESULT_VEC:.+]] = vector.insertelement %[[V5R]], %[[FLAT_RESULT_VEC_5]][%[[C5]] : i32] : vector<6xi32>
//       CHECK:       %[[RESULT:.+]] = vector.shape_cast %[[FLAT_RESULT_VEC]] : vector<6xi32> to vector<2x3xi32>
//       CHECK:       return %[[RESULT]]     


func @vector_multi_reduction_transposed(%arg0: vector<2x3x4x5xf32>) -> vector<2x5xf32> {
    %0 = vector.multi_reduction #vector.kind<add>, %arg0 [1, 2] : vector<2x3x4x5xf32> to vector<2x5xf32>
    return %0 : vector<2x5xf32>
}

// CHECK-LABEL: func @vector_multi_reduction_transposed
//  CHECK-SAME:    %[[INPUT:.+]]: vector<2x3x4x5xf32>
//       CHECK:     %[[TRANSPOSED_INPUT:.+]] = vector.transpose %[[INPUT]], [0, 3, 1, 2] : vector<2x3x4x5xf32> to vector<2x5x3x4xf32>
//       CHECK:     vector.shape_cast %[[TRANSPOSED_INPUT]] : vector<2x5x3x4xf32> to vector<10x12xf32>
//       CHECK:     %[[RESULT:.+]] = vector.shape_cast %{{.*}} : vector<10xf32> to vector<2x5xf32>
//       CHECK:       return %[[RESULT]]     

func @vector_multi_reduction_ordering(%arg0: vector<3x2x4xf32>) -> vector<2x4xf32> {
    %0 = vector.multi_reduction #vector.kind<mul>, %arg0 [0] : vector<3x2x4xf32> to vector<2x4xf32>
    return %0 : vector<2x4xf32>
}
// CHECK-LABEL: func @vector_multi_reduction_ordering
//  CHECK-SAME:   %[[INPUT:.+]]: vector<3x2x4xf32>
//       CHECK:       %[[RESULT_VEC_0:.+]] = constant dense<{{.*}}> : vector<8xf32>
//       CHECK:       %[[C0:.+]] = constant 0 : i32
//       CHECK:       %[[C1:.+]] = constant 1 : i32
//       CHECK:       %[[C2:.+]] = constant 2 : i32
//       CHECK:       %[[C3:.+]] = constant 3 : i32
//       CHECK:       %[[C4:.+]] = constant 4 : i32
//       CHECK:       %[[C5:.+]] = constant 5 : i32
//       CHECK:       %[[C6:.+]] = constant 6 : i32
//       CHECK:       %[[C7:.+]] = constant 7 : i32
//       CHECK:       %[[TRANSPOSED_INPUT:.+]] = vector.transpose %[[INPUT]], [1, 2, 0] : vector<3x2x4xf32> to vector<2x4x3xf32>
//       CHECK:       %[[V0:.+]] = vector.extract %[[TRANSPOSED_INPUT]][0, 0]
//       CHECK:       %[[RV0:.+]] = vector.reduction "mul", %[[V0]] : vector<3xf32> into f32
//       CHECK:       %[[RESULT_VEC_1:.+]] = vector.insertelement %[[RV0:.+]], %[[RESULT_VEC_0]][%[[C0]] : i32] : vector<8xf32>
//       CHECK:       %[[V1:.+]] = vector.extract %[[TRANSPOSED_INPUT]][0, 1]
//       CHECK:       %[[RV1:.+]] = vector.reduction "mul", %[[V1]] : vector<3xf32> into f32
//       CHECK:       %[[RESULT_VEC_2:.+]] = vector.insertelement %[[RV1:.+]], %[[RESULT_VEC_1]][%[[C1]] : i32] : vector<8xf32>
//       CHECK:       %[[V2:.+]] = vector.extract %[[TRANSPOSED_INPUT]][0, 2]
//       CHECK:       %[[RV2:.+]] = vector.reduction "mul", %[[V2]] : vector<3xf32> into f32
//       CHECK:       %[[RESULT_VEC_3:.+]] = vector.insertelement %[[RV2:.+]], %[[RESULT_VEC_2]][%[[C2]] : i32] : vector<8xf32>
//       CHECK:       %[[V3:.+]] = vector.extract %[[TRANSPOSED_INPUT]][0, 3]
//       CHECK:       %[[RV3:.+]] = vector.reduction "mul", %[[V3]] : vector<3xf32> into f32
//       CHECK:       %[[RESULT_VEC_4:.+]] = vector.insertelement %[[RV3:.+]], %[[RESULT_VEC_3]][%[[C3]] : i32] : vector<8xf32>
//       CHECK:       %[[V4:.+]] = vector.extract %[[TRANSPOSED_INPUT]][1, 0]
//       CHECK:       %[[RV4:.+]] = vector.reduction "mul", %[[V4]] : vector<3xf32> into f32
//       CHECK:       %[[RESULT_VEC_5:.+]] = vector.insertelement %[[RV4:.+]], %[[RESULT_VEC_4]][%[[C4]] : i32] : vector<8xf32>
//       CHECK:       %[[V5:.+]] = vector.extract %[[TRANSPOSED_INPUT]][1, 1]
//       CHECK:       %[[RV5:.+]] = vector.reduction "mul", %[[V5]] : vector<3xf32> into f32
//       CHECK:       %[[RESULT_VEC_6:.+]] = vector.insertelement %[[RV5:.+]], %[[RESULT_VEC_5]][%[[C5]] : i32] : vector<8xf32>
//       CHECK:       %[[V6:.+]] = vector.extract %[[TRANSPOSED_INPUT]][1, 2]
//       CHECK:       %[[RV6:.+]] = vector.reduction "mul", %[[V6]] : vector<3xf32> into f32
//       CHECK:       %[[RESULT_VEC_7:.+]] = vector.insertelement %[[RV6:.+]], %[[RESULT_VEC_6]][%[[C6]] : i32] : vector<8xf32>
//       CHECK:       %[[V7:.+]] = vector.extract %[[TRANSPOSED_INPUT]][1, 3]
//       CHECK:       %[[RV7:.+]] = vector.reduction "mul", %[[V7]] : vector<3xf32> into f32
//       CHECK:       %[[RESULT_VEC:.+]] = vector.insertelement %[[RV7:.+]], %[[RESULT_VEC_7]][%[[C7]] : i32] : vector<8xf32>
//       CHECK:       %[[RESHAPED_VEC:.+]] = vector.shape_cast %[[RESULT_VEC]] : vector<8xf32> to vector<2x4xf32>
//       CHECK:       return %[[RESHAPED_VEC]]
