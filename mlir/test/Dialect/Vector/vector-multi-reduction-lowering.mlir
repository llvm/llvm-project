// RUN: mlir-opt %s -test-vector-multi-reduction-lowering-patterns -split-input-file | FileCheck %s

func.func @vector_multi_reduction(%arg0: vector<2x4xf32>, %acc: vector<2xf32>) -> vector<2xf32> {
    %0 = vector.multi_reduction <mul>, %arg0, %acc [1] : vector<2x4xf32> to vector<2xf32>
    return %0 : vector<2xf32>
}
// CHECK-LABEL: func @vector_multi_reduction
//  CHECK-SAME:   %[[INPUT:.+]]: vector<2x4xf32>, %[[ACC:.*]]: vector<2xf32>)
//       CHECK:       %[[RESULT_VEC_0:.+]] = arith.constant dense<{{.*}}> : vector<2xf32>
//       CHECK:       %[[C0:.+]] = arith.constant 0 : index
//       CHECK:       %[[C1:.+]] = arith.constant 1 : index
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
//       CHECK:   %[[RES:.*]] = vector.extract %[[INSERTED]][0] : vector<1xf32>
//       CHECK:   return %[[RES]]

// -----

func.func @vector_reduction_inner(%arg0: vector<2x3x4x5xi32>, %acc: vector<2x3xi32>) -> vector<2x3xi32> {
    %0 = vector.multi_reduction <add>, %arg0, %acc [2, 3] : vector<2x3x4x5xi32> to vector<2x3xi32>
    return %0 : vector<2x3xi32>
}
// CHECK-LABEL: func @vector_reduction_inner
//  CHECK-SAME:   %[[INPUT:.+]]: vector<2x3x4x5xi32>, %[[ACC:.*]]: vector<2x3xi32>
//       CHECK:       %[[FLAT_RESULT_VEC_0:.+]] = arith.constant dense<0> : vector<6xi32>
//   CHECK-DAG:       %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:       %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:       %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:       %[[C3:.+]] = arith.constant 3 : index
//   CHECK-DAG:       %[[C4:.+]] = arith.constant 4 : index
//   CHECK-DAG:       %[[C5:.+]] = arith.constant 5 : index
//       CHECK:       %[[RESHAPED_INPUT:.+]] = vector.shape_cast %[[INPUT]] : vector<2x3x4x5xi32> to vector<6x20xi32>
//       CHECK:       %[[V0:.+]] = vector.extract %[[RESHAPED_INPUT]][0] : vector<6x20xi32>
//       CHECK:       %[[ACC0:.+]] = vector.extract %[[ACC]][0, 0] : vector<2x3xi32>
//       CHECK:       %[[V0R:.+]] = vector.reduction <add>, %[[V0]], %[[ACC0]] : vector<20xi32> into i32
//       CHECK:       %[[FLAT_RESULT_VEC_1:.+]] = vector.insertelement %[[V0R]], %[[FLAT_RESULT_VEC_0]][%[[C0]] : index] : vector<6xi32>
//       CHECK:       %[[V1:.+]] = vector.extract %[[RESHAPED_INPUT]][1] : vector<6x20xi32>
//       CHECK:       %[[ACC1:.+]] = vector.extract %[[ACC]][0, 1] : vector<2x3xi32>
//       CHECK:       %[[V1R:.+]] = vector.reduction <add>, %[[V1]], %[[ACC1]] : vector<20xi32> into i32
//       CHECK:       %[[FLAT_RESULT_VEC_2:.+]] = vector.insertelement %[[V1R]], %[[FLAT_RESULT_VEC_1]][%[[C1]] : index] : vector<6xi32>
//       CHECK:       %[[V2:.+]] = vector.extract %[[RESHAPED_INPUT]][2] : vector<6x20xi32>
//       CHECK:       %[[ACC2:.+]] = vector.extract %[[ACC]][0, 2] : vector<2x3xi32>
//       CHECK:       %[[V2R:.+]] = vector.reduction <add>, %[[V2]], %[[ACC2]] : vector<20xi32> into i32
//       CHECK:       %[[FLAT_RESULT_VEC_3:.+]] = vector.insertelement %[[V2R]], %[[FLAT_RESULT_VEC_2]][%[[C2]] : index] : vector<6xi32>
//       CHECK:       %[[V3:.+]] = vector.extract %[[RESHAPED_INPUT]][3] : vector<6x20xi32>
//       CHECK:       %[[ACC3:.+]] = vector.extract %[[ACC]][1, 0] : vector<2x3xi32>
//       CHECK:       %[[V3R:.+]] = vector.reduction <add>, %[[V3]], %[[ACC3]] : vector<20xi32> into i32
//       CHECK:       %[[FLAT_RESULT_VEC_4:.+]] = vector.insertelement %[[V3R]], %[[FLAT_RESULT_VEC_3]][%[[C3]] : index] : vector<6xi32>
//       CHECK:       %[[V4:.+]] = vector.extract %[[RESHAPED_INPUT]][4] : vector<6x20xi32>
//       CHECK:       %[[ACC4:.+]] = vector.extract %[[ACC]][1, 1] : vector<2x3xi32>
//       CHECK:       %[[V4R:.+]] = vector.reduction <add>, %[[V4]], %[[ACC4]] : vector<20xi32> into i32
//       CHECK:       %[[FLAT_RESULT_VEC_5:.+]] = vector.insertelement %[[V4R]], %[[FLAT_RESULT_VEC_4]][%[[C4]] : index] : vector<6xi32>
///       CHECK:      %[[V5:.+]] = vector.extract %[[RESHAPED_INPUT]][5] : vector<6x20xi32>
//       CHECK:       %[[ACC5:.+]] = vector.extract %[[ACC]][1, 2] : vector<2x3xi32>
//       CHECK:       %[[V5R:.+]] = vector.reduction <add>, %[[V5]], %[[ACC5]] : vector<20xi32> into i32
//       CHECK:       %[[FLAT_RESULT_VEC:.+]] = vector.insertelement %[[V5R]], %[[FLAT_RESULT_VEC_5]][%[[C5]] : index] : vector<6xi32>
//       CHECK:       %[[RESULT:.+]] = vector.shape_cast %[[FLAT_RESULT_VEC]] : vector<6xi32> to vector<2x3xi32>
//       CHECK:       return %[[RESULT]]

// -----

func.func @vector_multi_reduction_transposed(%arg0: vector<2x3x4x5xf32>, %acc: vector<2x5xf32>) -> vector<2x5xf32> {
    %0 = vector.multi_reduction <add>, %arg0, %acc [1, 2] : vector<2x3x4x5xf32> to vector<2x5xf32>
    return %0 : vector<2x5xf32>
}

// CHECK-LABEL: func @vector_multi_reduction_transposed
//  CHECK-SAME:    %[[INPUT:.+]]: vector<2x3x4x5xf32>
//       CHECK:     %[[TRANSPOSED_INPUT:.+]] = vector.transpose %[[INPUT]], [0, 3, 1, 2] : vector<2x3x4x5xf32> to vector<2x5x3x4xf32>
//       CHECK:     vector.shape_cast %[[TRANSPOSED_INPUT]] : vector<2x5x3x4xf32> to vector<10x12xf32>
//       CHECK:     %[[RESULT:.+]] = vector.shape_cast %{{.*}} : vector<10xf32> to vector<2x5xf32>
//       CHECK:       return %[[RESULT]]

// -----

func.func @vector_multi_reduction_ordering(%arg0: vector<3x2x4xf32>, %acc: vector<2x4xf32>) -> vector<2x4xf32> {
    %0 = vector.multi_reduction <mul>, %arg0, %acc [0] : vector<3x2x4xf32> to vector<2x4xf32>
    return %0 : vector<2x4xf32>
}
// CHECK-LABEL: func @vector_multi_reduction_ordering
//  CHECK-SAME:   %[[INPUT:.+]]: vector<3x2x4xf32>, %[[ACC:.*]]: vector<2x4xf32>)
//       CHECK:       %[[RESULT_VEC_0:.+]] = arith.constant dense<{{.*}}> : vector<8xf32>
//       CHECK:       %[[C0:.+]] = arith.constant 0 : index
//       CHECK:       %[[C1:.+]] = arith.constant 1 : index
//       CHECK:       %[[C2:.+]] = arith.constant 2 : index
//       CHECK:       %[[C3:.+]] = arith.constant 3 : index
//       CHECK:       %[[C4:.+]] = arith.constant 4 : index
//       CHECK:       %[[C5:.+]] = arith.constant 5 : index
//       CHECK:       %[[C6:.+]] = arith.constant 6 : index
//       CHECK:       %[[C7:.+]] = arith.constant 7 : index
//       CHECK:       %[[TRANSPOSED_INPUT:.+]] = vector.transpose %[[INPUT]], [1, 2, 0] : vector<3x2x4xf32> to vector<2x4x3xf32>
//       CHECK:       %[[V0:.+]] = vector.extract %[[TRANSPOSED_INPUT]][0, 0]
//       CHECK:       %[[ACC0:.+]] = vector.extract %[[ACC]][0, 0] : vector<2x4xf32>
//       CHECK:       %[[RV0:.+]] = vector.reduction <mul>, %[[V0]], %[[ACC0]] : vector<3xf32> into f32
//       CHECK:       %[[RESULT_VEC_1:.+]] = vector.insertelement %[[RV0:.+]], %[[RESULT_VEC_0]][%[[C0]] : index] : vector<8xf32>
//       CHECK:       %[[V1:.+]] = vector.extract %[[TRANSPOSED_INPUT]][0, 1]
//       CHECK:       %[[ACC1:.+]] = vector.extract %[[ACC]][0, 1] : vector<2x4xf32>
//       CHECK:       %[[RV1:.+]] = vector.reduction <mul>, %[[V1]], %[[ACC1]] : vector<3xf32> into f32
//       CHECK:       %[[RESULT_VEC_2:.+]] = vector.insertelement %[[RV1:.+]], %[[RESULT_VEC_1]][%[[C1]] : index] : vector<8xf32>
//       CHECK:       %[[V2:.+]] = vector.extract %[[TRANSPOSED_INPUT]][0, 2]
//       CHECK:       %[[ACC2:.+]] = vector.extract %[[ACC]][0, 2] : vector<2x4xf32>
//       CHECK:       %[[RV2:.+]] = vector.reduction <mul>, %[[V2]], %[[ACC2]] : vector<3xf32> into f32
//       CHECK:       %[[RESULT_VEC_3:.+]] = vector.insertelement %[[RV2:.+]], %[[RESULT_VEC_2]][%[[C2]] : index] : vector<8xf32>
//       CHECK:       %[[V3:.+]] = vector.extract %[[TRANSPOSED_INPUT]][0, 3]
//       CHECK:       %[[ACC3:.+]] = vector.extract %[[ACC]][0, 3] : vector<2x4xf32>
//       CHECK:       %[[RV3:.+]] = vector.reduction <mul>, %[[V3]], %[[ACC3]] : vector<3xf32> into f32
//       CHECK:       %[[RESULT_VEC_4:.+]] = vector.insertelement %[[RV3:.+]], %[[RESULT_VEC_3]][%[[C3]] : index] : vector<8xf32>
//       CHECK:       %[[V4:.+]] = vector.extract %[[TRANSPOSED_INPUT]][1, 0]
//       CHECK:       %[[ACC4:.+]] = vector.extract %[[ACC]][1, 0] : vector<2x4xf32>
//       CHECK:       %[[RV4:.+]] = vector.reduction <mul>, %[[V4]], %[[ACC4]] : vector<3xf32> into f32
//       CHECK:       %[[RESULT_VEC_5:.+]] = vector.insertelement %[[RV4:.+]], %[[RESULT_VEC_4]][%[[C4]] : index] : vector<8xf32>
//       CHECK:       %[[V5:.+]] = vector.extract %[[TRANSPOSED_INPUT]][1, 1]
//       CHECK:       %[[ACC5:.+]] = vector.extract %[[ACC]][1, 1] : vector<2x4xf32>
//       CHECK:       %[[RV5:.+]] = vector.reduction <mul>, %[[V5]], %[[ACC5]] : vector<3xf32> into f32
//       CHECK:       %[[RESULT_VEC_6:.+]] = vector.insertelement %[[RV5:.+]], %[[RESULT_VEC_5]][%[[C5]] : index] : vector<8xf32>
//       CHECK:       %[[V6:.+]] = vector.extract %[[TRANSPOSED_INPUT]][1, 2]
//       CHECK:       %[[ACC6:.+]] = vector.extract %[[ACC]][1, 2] : vector<2x4xf32>
//       CHECK:       %[[RV6:.+]] = vector.reduction <mul>, %[[V6]], %[[ACC6]] : vector<3xf32> into f32
//       CHECK:       %[[RESULT_VEC_7:.+]] = vector.insertelement %[[RV6:.+]], %[[RESULT_VEC_6]][%[[C6]] : index] : vector<8xf32>
//       CHECK:       %[[V7:.+]] = vector.extract %[[TRANSPOSED_INPUT]][1, 3]
//       CHECK:       %[[ACC7:.+]] = vector.extract %[[ACC]][1, 3] : vector<2x4xf32>
//       CHECK:       %[[RV7:.+]] = vector.reduction <mul>, %[[V7]], %[[ACC7]] : vector<3xf32> into f32
//       CHECK:       %[[RESULT_VEC:.+]] = vector.insertelement %[[RV7:.+]], %[[RESULT_VEC_7]][%[[C7]] : index] : vector<8xf32>
//       CHECK:       %[[RESHAPED_VEC:.+]] = vector.shape_cast %[[RESULT_VEC]] : vector<8xf32> to vector<2x4xf32>
//       CHECK:       return %[[RESHAPED_VEC]]

// -----

func.func @vectorize_dynamic_reduction(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
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

// Verify that the original 2-D mask is sliced and propagated properly to the
// vector.reduction instances.

// CHECK-LABEL:   func.func @vectorize_dynamic_reduction
// CHECK:           %[[VAL_8:.*]] = tensor.dim
// CHECK:           %[[VAL_9:.*]] = tensor.dim
// CHECK:           %[[VAL_10:.*]] = vector.create_mask %[[VAL_8]], %[[VAL_9]] : vector<4x8xi1>

// CHECK:           %[[VAL_16:.*]] = vector.extract %[[VAL_10]][0] : vector<4x8xi1>
// CHECK:           %[[VAL_17:.*]] = vector.mask %[[VAL_16]] { vector.reduction <add>, %{{.*}} : vector<8xf32> into f32 } : vector<8xi1> -> f32
// CHECK:           %[[VAL_18:.*]] = vector.insertelement

// CHECK:           %[[VAL_21:.*]] = vector.extract %[[VAL_10]][1] : vector<4x8xi1>
// CHECK:           %[[VAL_22:.*]] = vector.mask %[[VAL_21]] { vector.reduction <add>, %{{.*}} : vector<8xf32> into f32 } : vector<8xi1> -> f32
// CHECK:           %[[VAL_23:.*]] = vector.insertelement

// CHECK:           %[[VAL_26:.*]] = vector.extract %[[VAL_10]][2] : vector<4x8xi1>
// CHECK:           %[[VAL_27:.*]] = vector.mask %[[VAL_26]] { vector.reduction <add>, %{{.*}} : vector<8xf32> into f32 } : vector<8xi1> -> f32
// CHECK:           %[[VAL_28:.*]] = vector.insertelement

// CHECK:           %[[VAL_31:.*]] = vector.extract %[[VAL_10]][3] : vector<4x8xi1>
// CHECK:           %[[VAL_32:.*]] = vector.mask %[[VAL_31]] { vector.reduction <add>, %{{.*}} : vector<8xf32> into f32 } : vector<8xi1> -> f32
// CHECK:           %[[VAL_33:.*]] = vector.insertelement

// -----

func.func @vectorize_dynamic_transpose_reduction(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %c1 = arith.constant 1 : index
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %c2 = arith.constant 2 : index
  %dim_1 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %c0_2 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = vector.create_mask %dim, %dim_0, %dim_1 : vector<4x8x16xi1>
  %1 = vector.mask %0 { vector.transfer_read %arg0[%c0_2, %c0_2, %c0_2], %cst {in_bounds = [true, true, true]} : tensor<?x?x?xf32>, vector<4x8x16xf32> } : vector<4x8x16xi1> -> vector<4x8x16xf32>
  %cst_3 = arith.constant 0.000000e+00 : f32
  %2 = vector.create_mask %dim_1, %dim_0 : vector<16x8xi1>
  %3 = vector.mask %2 { vector.transfer_read %arg1[%c0_2, %c0_2], %cst_3 {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : tensor<?x?xf32>, vector<8x16xf32> } : vector<16x8xi1> -> vector<8x16xf32>
  %4 = vector.mask %0 { vector.multi_reduction <add>, %1, %3 [0] : vector<4x8x16xf32> to vector<8x16xf32> } : vector<4x8x16xi1> -> vector<8x16xf32>
  %c0_4 = arith.constant 0 : index
  %5 = vector.mask %2 { vector.transfer_write %4, %arg1[%c0_4, %c0_4] {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : vector<8x16xf32>, tensor<?x?xf32> } : vector<16x8xi1> -> tensor<?x?xf32>
  return %5 : tensor<?x?xf32>
}

// CHECK-LABEL:   func.func @vectorize_dynamic_transpose_reduction
// CHECK:           %[[VAL_6:.*]] = tensor.dim
// CHECK:           %[[VAL_7:.*]] = tensor.dim
// CHECK:           %[[VAL_8:.*]] = tensor.dim
// CHECK:           %[[VAL_135:.*]] = vector.create_mask %{{.*}}, %{{.*}}, %{{.*}} : vector<4x8x16xi1>
// CHECK:           %[[VAL_139:.*]] = vector.transpose %[[VAL_135]], [1, 2, 0] : vector<4x8x16xi1> to vector<8x16x4xi1>

// Just checking a few instances to make sure the vector mask is properly propagated:

// CHECK:           %[[VAL_143:.*]] = vector.extract %[[VAL_139]][0, 0] : vector<8x16x4xi1>
// CHECK:           %[[VAL_144:.*]] = vector.mask %[[VAL_143]] { vector.reduction <add>
// CHECK:           %[[VAL_145:.*]] = vector.insertelement %[[VAL_144]]

// CHECK:           %[[VAL_148:.*]] = vector.extract %[[VAL_139]][0, 1] : vector<8x16x4xi1>
// CHECK:           %[[VAL_149:.*]] = vector.mask %[[VAL_148]] { vector.reduction <add>
// CHECK:           %[[VAL_150:.*]] = vector.insertelement %[[VAL_149]]

// CHECK:           %[[VAL_153:.*]] = vector.extract %[[VAL_139]][0, 2] : vector<8x16x4xi1>
// CHECK:           %[[VAL_154:.*]] = vector.mask %[[VAL_153]] { vector.reduction <add>
// CHECK:           %[[VAL_155:.*]] = vector.insertelement %[[VAL_154]]

// CHECK:           %[[VAL_158:.*]] = vector.extract %[[VAL_139]][0, 3] : vector<8x16x4xi1>
// CHECK:           %[[VAL_159:.*]] = vector.mask %[[VAL_158]] { vector.reduction <add>
// CHECK:           %[[VAL_160:.*]] = vector.insertelement %[[VAL_159]]

// -----

func.func @vector_multi_reduction_parallel_middle(%arg0: vector<3x4x5xf32>, %acc: vector<4xf32>) -> vector<4xf32> {
    %0 = vector.multi_reduction <add>, %arg0, %acc [0, 2] : vector<3x4x5xf32> to vector<4xf32>
    return %0 : vector<4xf32>
}

// CHECK-LABEL: func @vector_multi_reduction_parallel_middle
//  CHECK-SAME:   %[[INPUT:.+]]: vector<3x4x5xf32>, %[[ACC:.+]]: vector<4xf32>
//       CHECK: vector.transpose %[[INPUT]], [1, 0, 2] : vector<3x4x5xf32> to vector<4x3x5xf32>
