// RUN: mlir-opt -lower-vector-multi-reduction="lowering-strategy=inner-reduction" -split-input-file %s | FileCheck %s

// -----
func.func @vector_multi_reduction(%arg0: vector<2x4xf32>, %acc: vector<2xf32>) -> vector<2xf32> {
    %0 = vector.multi_reduction <mul>, %arg0, %acc [1] : vector<2x4xf32> to vector<2xf32>
    return %0 : vector<2xf32>
}
// CHECK-LABEL: func @vector_multi_reduction
//  CHECK-SAME:   %[[INPUT:.+]]: vector<2x4xf32>, %[[ACC:.*]]: vector<2xf32>)
//   CHECK-DAG:       %[[RESULT_VEC_0:.+]] = arith.constant dense<{{.*}}> : vector<2xf32>
//   CHECK-DAG:       %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:       %[[C1:.+]] = arith.constant 1 : index
//       CHECK:       %[[V0:.+]] = vector.extract %[[INPUT]][0]
//       CHECK:       %[[ACC0:.+]] = vector.extract %[[ACC]][0]
//       CHECK:       %[[RV0:.+]] = vector.reduction <mul>, %[[V0]], %[[ACC0]] : vector<4xf32> into f32
//       CHECK:       %[[RESULT_VEC_1:.+]] = vector.insertelement %[[RV0:.+]], %[[RESULT_VEC_0]][%[[C0]] : index] : vector<2xf32>
//       CHECK:       %[[V1:.+]] = vector.extract %[[INPUT]][1]
//       CHECK:       %[[ACC1:.+]] = vector.extract %[[ACC]][1]
//       CHECK:       %[[RV1:.+]] = vector.reduction <mul>, %[[V1]], %[[ACC1]] : vector<4xf32> into f32
//       CHECK:       %[[RESULT_VEC:.+]] = vector.insertelement %[[RV1:.+]], %[[RESULT_VEC_1]][%[[C1]] : index] : vector<2xf32>
//       CHECK:       return %[[RESULT_VEC]]

// -----
func.func @vector_multi_reduction_to_scalar(%arg0: vector<2x4xf32>, %acc: f32) -> f32 {
    %0 = vector.multi_reduction <mul>, %arg0, %acc [0, 1] : vector<2x4xf32> to f32
    return %0 : f32
}
// CHECK-LABEL: func @vector_multi_reduction_to_scalar
//  CHECK-SAME:   %[[INPUT:.+]]: vector<2x4xf32>, %[[ACC:.*]]: f32)
//       CHECK:   %[[CASTED:.*]] = vector.shape_cast %[[INPUT]] : vector<2x4xf32> to vector<8xf32>
//       CHECK:   %[[REDUCED:.*]] = vector.reduction <mul>, %[[CASTED]], %[[ACC]] : vector<8xf32> into f32
//       CHECK:   %[[INSERTED:.*]] = vector.insertelement %[[REDUCED]], {{.*}} : vector<1xf32>
//       CHECK:   %[[RES:.*]] = vector.extract %[[INSERTED]][0] : f32 from vector<1xf32>
//       CHECK:   return %[[RES]]
