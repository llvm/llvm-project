// RUN: mlir-opt -lower-vector-multi-reduction="lowering-strategy=inner-parallel" -split-input-file %s | FileCheck %s

// -----
func.func @vector_multi_reduction(%arg0: vector<2x4xf32>, %acc: vector<2xf32>) -> vector<2xf32> {
    %0 = vector.multi_reduction <mul>, %arg0, %acc [1] : vector<2x4xf32> to vector<2xf32>
    return %0 : vector<2xf32>
}

// CHECK-LABEL: func @vector_multi_reduction
//  CHECK-SAME:   %[[INPUT:.+]]: vector<2x4xf32>, %[[ACC:.*]]: vector<2xf32>
//       CHECK:   %[[TRANSPOSED:.+]] = vector.transpose %[[INPUT]], [1, 0] : vector<2x4xf32> to vector<4x2xf32>
//       CHECK:   %[[V0:.+]] = vector.extract %[[TRANSPOSED]][0] : vector<2xf32> from vector<4x2xf32>
//       CHECK:   %[[RV0:.+]] = arith.mulf %[[V0]], %[[ACC]] : vector<2xf32>
//       CHECK:   %[[V1:.+]] = vector.extract %[[TRANSPOSED]][1] : vector<2xf32> from vector<4x2xf32>
//       CHECK:   %[[RV01:.+]] = arith.mulf %[[V1]], %[[RV0]] : vector<2xf32>
//       CHECK:   %[[V2:.+]] = vector.extract %[[TRANSPOSED]][2] : vector<2xf32> from vector<4x2xf32>
//       CHECK:   %[[RV012:.+]] = arith.mulf %[[V2]], %[[RV01]] : vector<2xf32>
//       CHECK:   %[[V3:.+]] = vector.extract %[[TRANSPOSED]][3] : vector<2xf32> from vector<4x2xf32>
//       CHECK:   %[[RESULT_VEC:.+]] = arith.mulf %[[V3]], %[[RV012]] : vector<2xf32>
//       CHECK:   return %[[RESULT_VEC]] : vector<2xf32>

// -----
func.func @vector_multi_reduction_min(%arg0: vector<2x4xf32>, %acc: vector<2xf32>) -> vector<2xf32> {
    %0 = vector.multi_reduction <minnumf>, %arg0, %acc [1] : vector<2x4xf32> to vector<2xf32>
    return %0 : vector<2xf32>
}

// CHECK-LABEL: func @vector_multi_reduction_min
//  CHECK-SAME:   %[[INPUT:.+]]: vector<2x4xf32>, %[[ACC:.*]]: vector<2xf32>
//       CHECK:   %[[TRANSPOSED:.+]] = vector.transpose %[[INPUT]], [1, 0] : vector<2x4xf32> to vector<4x2xf32>
//       CHECK:   %[[V0:.+]] = vector.extract %[[TRANSPOSED]][0] : vector<2xf32> from vector<4x2xf32>
//       CHECK:   %[[RV0:.+]] = arith.minnumf %[[V0]], %[[ACC]] : vector<2xf32>
//       CHECK:   %[[V1:.+]] = vector.extract %[[TRANSPOSED]][1] : vector<2xf32> from vector<4x2xf32>
//       CHECK:   %[[RV01:.+]] = arith.minnumf %[[V1]], %[[RV0]] : vector<2xf32>
//       CHECK:   %[[V2:.+]] = vector.extract %[[TRANSPOSED]][2] : vector<2xf32> from vector<4x2xf32>
//       CHECK:   %[[RV012:.+]] = arith.minnumf %[[V2]], %[[RV01]] : vector<2xf32>
//       CHECK:   %[[V3:.+]] = vector.extract %[[TRANSPOSED]][3] : vector<2xf32> from vector<4x2xf32>
//       CHECK:   %[[RESULT_VEC:.+]] = arith.minnumf %[[V3]], %[[RV012]] : vector<2xf32>
//       CHECK:   return %[[RESULT_VEC]] : vector<2xf32>