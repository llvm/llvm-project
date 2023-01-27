// RUN: mlir-opt %s -test-vector-multi-reduction-lowering-patterns="use-outer-reductions" | FileCheck %s

func.func @vector_multi_reduction(%arg0: vector<2x4xf32>, %acc: vector<2xf32>) -> vector<2xf32> {
    %0 = vector.multi_reduction <mul>, %arg0, %acc [1] : vector<2x4xf32> to vector<2xf32>
    return %0 : vector<2xf32>
}

// CHECK-LABEL: func @vector_multi_reduction
//  CHECK-SAME:   %[[INPUT:.+]]: vector<2x4xf32>, %[[ACC:.*]]: vector<2xf32>
//       CHECK:   %[[TRANSPOSED:.+]] = vector.transpose %[[INPUT]], [1, 0] : vector<2x4xf32> to vector<4x2xf32>
//       CHECK:   %[[V0:.+]] = vector.extract %[[TRANSPOSED]][0] : vector<4x2xf32>
//       CHECK:   %[[RV0:.+]] = arith.mulf %[[V0]], %[[ACC]] : vector<2xf32>
//       CHECK:   %[[V1:.+]] = vector.extract %[[TRANSPOSED]][1] : vector<4x2xf32>
//       CHECK:   %[[RV01:.+]] = arith.mulf %[[V1]], %[[RV0]] : vector<2xf32>
//       CHECK:   %[[V2:.+]] = vector.extract %[[TRANSPOSED]][2] : vector<4x2xf32>
//       CHECK:   %[[RV012:.+]] = arith.mulf %[[V2]], %[[RV01]] : vector<2xf32>
//       CHECK:   %[[V3:.+]] = vector.extract %[[TRANSPOSED]][3] : vector<4x2xf32>
//       CHECK:   %[[RESULT_VEC:.+]] = arith.mulf %[[V3]], %[[RV012]] : vector<2xf32>
//       CHECK:   return %[[RESULT_VEC]] : vector<2xf32>

func.func @vector_multi_reduction_min(%arg0: vector<2x4xf32>, %acc: vector<2xf32>) -> vector<2xf32> {
    %0 = vector.multi_reduction <minf>, %arg0, %acc [1] : vector<2x4xf32> to vector<2xf32>
    return %0 : vector<2xf32>
}

// CHECK-LABEL: func @vector_multi_reduction_min
//  CHECK-SAME:   %[[INPUT:.+]]: vector<2x4xf32>, %[[ACC:.*]]: vector<2xf32>
//       CHECK:   %[[TRANSPOSED:.+]] = vector.transpose %[[INPUT]], [1, 0] : vector<2x4xf32> to vector<4x2xf32>
//       CHECK:   %[[V0:.+]] = vector.extract %[[TRANSPOSED]][0] : vector<4x2xf32>
//       CHECK:   %[[RV0:.+]] = arith.minf %[[V0]], %[[ACC]] : vector<2xf32>
//       CHECK:   %[[V1:.+]] = vector.extract %[[TRANSPOSED]][1] : vector<4x2xf32>
//       CHECK:   %[[RV01:.+]] = arith.minf %[[V1]], %[[RV0]] : vector<2xf32>
//       CHECK:   %[[V2:.+]] = vector.extract %[[TRANSPOSED]][2] : vector<4x2xf32>
//       CHECK:   %[[RV012:.+]] = arith.minf %[[V2]], %[[RV01]] : vector<2xf32>
//       CHECK:   %[[V3:.+]] = vector.extract %[[TRANSPOSED]][3] : vector<4x2xf32>
//       CHECK:   %[[RESULT_VEC:.+]] = arith.minf %[[V3]], %[[RV012]] : vector<2xf32>
//       CHECK:   return %[[RESULT_VEC]] : vector<2xf32>

func.func @vector_multi_reduction_max(%arg0: vector<2x4xf32>, %acc: vector<2xf32>) -> vector<2xf32> {
    %0 = vector.multi_reduction <maxf>, %arg0, %acc [1] : vector<2x4xf32> to vector<2xf32>
    return %0 : vector<2xf32>
}

// CHECK-LABEL: func @vector_multi_reduction_max
//  CHECK-SAME:   %[[INPUT:.+]]: vector<2x4xf32>, %[[ACC:.*]]: vector<2xf32>
//       CHECK:   %[[TRANSPOSED:.+]] = vector.transpose %[[INPUT]], [1, 0] : vector<2x4xf32> to vector<4x2xf32>
//       CHECK:   %[[V0:.+]] = vector.extract %[[TRANSPOSED]][0] : vector<4x2xf32>
//       CHECK:   %[[RV0:.+]] = arith.maxf %[[V0]], %[[ACC]] : vector<2xf32>
//       CHECK:   %[[V1:.+]] = vector.extract %[[TRANSPOSED]][1] : vector<4x2xf32>
//       CHECK:   %[[RV01:.+]] = arith.maxf %[[V1]], %[[RV0]] : vector<2xf32>
//       CHECK:   %[[V2:.+]] = vector.extract %[[TRANSPOSED]][2] : vector<4x2xf32>
//       CHECK:   %[[RV012:.+]] = arith.maxf %[[V2]], %[[RV01]] : vector<2xf32>
//       CHECK:   %[[V3:.+]] = vector.extract %[[TRANSPOSED]][3] : vector<4x2xf32>
//       CHECK:   %[[RESULT_VEC:.+]] = arith.maxf %[[V3]], %[[RV012]] : vector<2xf32>
//       CHECK:   return %[[RESULT_VEC]] : vector<2xf32>

func.func @vector_multi_reduction_and(%arg0: vector<2x4xi32>, %acc: vector<2xi32>) -> vector<2xi32> {
    %0 = vector.multi_reduction <and>, %arg0, %acc [1] : vector<2x4xi32> to vector<2xi32>
    return %0 : vector<2xi32>
}

// CHECK-LABEL: func @vector_multi_reduction_and
//  CHECK-SAME:   %[[INPUT:.+]]: vector<2x4xi32>, %[[ACC:.*]]: vector<2xi32>
//       CHECK:   %[[TRANSPOSED:.+]] = vector.transpose %[[INPUT]], [1, 0] : vector<2x4xi32> to vector<4x2xi32>
//       CHECK:   %[[V0:.+]] = vector.extract %[[TRANSPOSED]][0] : vector<4x2xi32>
//       CHECK:   %[[RV0:.+]] = arith.andi %[[V0]], %[[ACC]] : vector<2xi32>
//       CHECK:   %[[V1:.+]] = vector.extract %[[TRANSPOSED]][1] : vector<4x2xi32>
//       CHECK:   %[[RV01:.+]] = arith.andi %[[V1]], %[[RV0]] : vector<2xi32>
//       CHECK:   %[[V2:.+]] = vector.extract %[[TRANSPOSED]][2] : vector<4x2xi32>
//       CHECK:   %[[RV012:.+]] = arith.andi %[[V2]], %[[RV01]] : vector<2xi32>
//       CHECK:   %[[V3:.+]] = vector.extract %[[TRANSPOSED]][3] : vector<4x2xi32>
//       CHECK:   %[[RESULT_VEC:.+]] = arith.andi %[[V3]], %[[RV012]] : vector<2xi32>
//       CHECK:   return %[[RESULT_VEC]] : vector<2xi32>

func.func @vector_multi_reduction_or(%arg0: vector<2x4xi32>, %acc: vector<2xi32>) -> vector<2xi32> {
    %0 = vector.multi_reduction <or>, %arg0, %acc [1] : vector<2x4xi32> to vector<2xi32>
    return %0 : vector<2xi32>
}

// CHECK-LABEL: func @vector_multi_reduction_or
//  CHECK-SAME:   %[[INPUT:.+]]: vector<2x4xi32>, %[[ACC:.*]]: vector<2xi32>
//       CHECK:   %[[TRANSPOSED:.+]] = vector.transpose %[[INPUT]], [1, 0] : vector<2x4xi32> to vector<4x2xi32>
//       CHECK:   %[[V0:.+]] = vector.extract %[[TRANSPOSED]][0] : vector<4x2xi32>
//       CHECK:   %[[RV0:.+]] = arith.ori %[[V0]], %[[ACC]] : vector<2xi32>
//       CHECK:   %[[V1:.+]] = vector.extract %[[TRANSPOSED]][1] : vector<4x2xi32>
//       CHECK:   %[[RV01:.+]] = arith.ori %[[V1]], %[[RV0]] : vector<2xi32>
//       CHECK:   %[[V2:.+]] = vector.extract %[[TRANSPOSED]][2] : vector<4x2xi32>
//       CHECK:   %[[RV012:.+]] = arith.ori %[[V2]], %[[RV01]] : vector<2xi32>
//       CHECK:   %[[V3:.+]] = vector.extract %[[TRANSPOSED]][3] : vector<4x2xi32>
//       CHECK:   %[[RESULT_VEC:.+]] = arith.ori %[[V3]], %[[RV012]] : vector<2xi32>
//       CHECK:   return %[[RESULT_VEC]] : vector<2xi32>

func.func @vector_multi_reduction_xor(%arg0: vector<2x4xi32>, %acc: vector<2xi32>) -> vector<2xi32> {
    %0 = vector.multi_reduction <xor>, %arg0, %acc [1] : vector<2x4xi32> to vector<2xi32>
    return %0 : vector<2xi32>
}

// CHECK-LABEL: func @vector_multi_reduction_xor
//  CHECK-SAME:   %[[INPUT:.+]]: vector<2x4xi32>, %[[ACC:.*]]: vector<2xi32>
//       CHECK:   %[[TRANSPOSED:.+]] = vector.transpose %[[INPUT]], [1, 0] : vector<2x4xi32> to vector<4x2xi32>
//       CHECK:   %[[V0:.+]] = vector.extract %[[TRANSPOSED]][0] : vector<4x2xi32>
//       CHECK:   %[[RV0:.+]] = arith.xori %[[V0]], %[[ACC]] : vector<2xi32>
//       CHECK:   %[[V1:.+]] = vector.extract %[[TRANSPOSED]][1] : vector<4x2xi32>
//       CHECK:   %[[RV01:.+]] = arith.xori %[[V1]], %[[RV0]] : vector<2xi32>
//       CHECK:   %[[V2:.+]] = vector.extract %[[TRANSPOSED]][2] : vector<4x2xi32>
//       CHECK:   %[[RV012:.+]] = arith.xori %[[V2]], %[[RV01]] : vector<2xi32>
//       CHECK:   %[[V3:.+]] = vector.extract %[[TRANSPOSED]][3] : vector<4x2xi32>
//       CHECK:   %[[RESULT_VEC:.+]] = arith.xori %[[V3]], %[[RV012]] : vector<2xi32>
//       CHECK:   return %[[RESULT_VEC]] : vector<2xi32>


func.func @vector_reduction_outer(%arg0: vector<2x3x4x5xi32>, %acc: vector<2x3xi32>) -> vector<2x3xi32> {
    %0 = vector.multi_reduction <add>, %arg0, %acc [2, 3] : vector<2x3x4x5xi32> to vector<2x3xi32>
    return %0 : vector<2x3xi32>
}

// CHECK-LABEL: func @vector_reduction_outer
//  CHECK-SAME:   %[[INPUT:.+]]: vector<2x3x4x5xi32>, %[[ACC:.*]]: vector<2x3xi32>
//       CHECK:   %[[TRANSPOSED:.+]] = vector.transpose %[[INPUT]], [2, 3, 0, 1] : vector<2x3x4x5xi32> to vector<4x5x2x3xi32>
//       CHECK:   %[[RESHAPED:.+]] = vector.shape_cast %[[TRANSPOSED]] : vector<4x5x2x3xi32> to vector<20x6xi32>
//       CHECK:   %[[FACC:.+]] = vector.shape_cast %[[ACC]] : vector<2x3xi32> to vector<6xi32>
//       CHECK:   %[[V0:.+]] = vector.extract %[[RESHAPED]][0] : vector<20x6xi32>
//       CHECK:   %[[R:.+]] = arith.addi %[[V0]], %[[FACC]] : vector<6xi32>
//       CHECK:   %[[V1:.+]] = vector.extract %[[RESHAPED]][1] : vector<20x6xi32>
//       CHECK:   %[[R0:.+]] = arith.addi %[[V1]], %[[R]] : vector<6xi32>
//       CHECK:   %[[V2:.+]] = vector.extract %[[RESHAPED]][2] : vector<20x6xi32>
//       CHECK:   %[[R1:.+]] = arith.addi %[[V2]], %[[R0]] : vector<6xi32>
//       CHECK:   %[[V3:.+]] = vector.extract %[[RESHAPED]][3] : vector<20x6xi32>
//       CHECK:   %[[R2:.+]] = arith.addi %[[V3]], %[[R1]] : vector<6xi32>
//       CHECK:   %[[V4:.+]] = vector.extract %[[RESHAPED]][4] : vector<20x6xi32>
//       CHECK:   %[[R3:.+]] = arith.addi %[[V4]], %[[R2]] : vector<6xi32>
//       CHECK:   %[[V5:.+]] = vector.extract %[[RESHAPED]][5] : vector<20x6xi32>
//       CHECK:   %[[R4:.+]] = arith.addi %[[V5]], %[[R3]] : vector<6xi32>
//       CHECK:   %[[V6:.+]] = vector.extract %[[RESHAPED]][6] : vector<20x6xi32>
//       CHECK:   %[[R5:.+]] = arith.addi %[[V6]], %[[R4]] : vector<6xi32>
//       CHECK:   %[[V7:.+]] = vector.extract %[[RESHAPED]][7] : vector<20x6xi32>
//       CHECK:   %[[R6:.+]] = arith.addi %[[V7]], %[[R5]] : vector<6xi32>
//       CHECK:   %[[V8:.+]] = vector.extract %[[RESHAPED]][8] : vector<20x6xi32>
//       CHECK:   %[[R7:.+]] = arith.addi %[[V8]], %[[R6]] : vector<6xi32>
//       CHECK:   %[[V9:.+]] = vector.extract %[[RESHAPED]][9] : vector<20x6xi32>
//       CHECK:   %[[R8:.+]] = arith.addi %[[V9]], %[[R7]] : vector<6xi32>
//       CHECK:   %[[V10:.+]] = vector.extract %[[RESHAPED]][10] : vector<20x6xi32>
//       CHECK:   %[[R9:.+]] = arith.addi %[[V10]], %[[R8]] : vector<6xi32>
//       CHECK:   %[[V11:.+]] = vector.extract %[[RESHAPED]][11] : vector<20x6xi32>
//       CHECK:   %[[R10:.+]] = arith.addi %[[V11]], %[[R9]] : vector<6xi32>
//       CHECK:   %[[V12:.+]] = vector.extract %[[RESHAPED]][12] : vector<20x6xi32>
//       CHECK:   %[[R11:.+]] = arith.addi %[[V12]], %[[R10]] : vector<6xi32>
//       CHECK:   %[[V13:.+]] = vector.extract %[[RESHAPED]][13] : vector<20x6xi32>
//       CHECK:   %[[R12:.+]] = arith.addi %[[V13]], %[[R11]] : vector<6xi32>
//       CHECK:   %[[V14:.+]] = vector.extract %[[RESHAPED]][14] : vector<20x6xi32>
//       CHECK:   %[[R13:.+]] = arith.addi %[[V14]], %[[R12]] : vector<6xi32>
//       CHECK:   %[[V15:.+]] = vector.extract %[[RESHAPED]][15] : vector<20x6xi32>
//       CHECK:   %[[R14:.+]] = arith.addi %[[V15]], %[[R13]] : vector<6xi32>
//       CHECK:   %[[V16:.+]] = vector.extract %[[RESHAPED]][16] : vector<20x6xi32>
//       CHECK:   %[[R15:.+]] = arith.addi %[[V16]], %[[R14]] : vector<6xi32>
//       CHECK:   %[[V17:.+]] = vector.extract %[[RESHAPED]][17] : vector<20x6xi32>
//       CHECK:   %[[R16:.+]] = arith.addi %[[V17]], %[[R15]] : vector<6xi32>
//       CHECK:   %[[V18:.+]] = vector.extract %[[RESHAPED]][18] : vector<20x6xi32>
//       CHECK:   %[[R17:.+]] = arith.addi %[[V18]], %[[R16]] : vector<6xi32>
//       CHECK:   %[[V19:.+]] = vector.extract %[[RESHAPED]][19] : vector<20x6xi32>
//       CHECK:   %[[R18:.+]] = arith.addi %[[V19]], %[[R17]] : vector<6xi32>
//       CHECK:   %[[RESULT_VEC:.+]] = vector.shape_cast %[[R18]] : vector<6xi32> to vector<2x3xi32>
//       CHECK:   return %[[RESULT_VEC]] : vector<2x3xi32>

func.func @vector_multi_reduction_parallel_middle(%arg0: vector<3x4x5xf32>, %acc: vector<4xf32>) -> vector<4xf32> {
    %0 = vector.multi_reduction <add>, %arg0, %acc [0, 2] : vector<3x4x5xf32> to vector<4xf32>
    return %0 : vector<4xf32>
}

// CHECK-LABEL: func @vector_multi_reduction_parallel_middle
//  CHECK-SAME:   %[[INPUT:.+]]: vector<3x4x5xf32>, %[[ACC:.+]]: vector<4xf32>
//       CHECK: vector.transpose %[[INPUT]], [0, 2, 1] : vector<3x4x5xf32> to vector<3x5x4xf32>

// This test is mainly to catch a bug that running
// `InnerOuterDimReductionConversion` on this function results in an
// infinite loop. So just check that some value is returned.
func.func @vector_reduction_1D(%arg0 : vector<2xf32>, %acc: f32) -> f32 {
  %0 = vector.multi_reduction #vector.kind<maxf>, %arg0, %acc [0] : vector<2xf32> to f32
  return %0 : f32
}
// CHECK-LABEL: func @vector_reduction_1D
//       CHECK:   return %{{.+}}

func.func @vector_multi_reduction_to_scalar(%arg0: vector<2x3xf32>, %acc: f32) -> f32 {
  %0 = vector.multi_reduction <add>, %arg0, %acc [0, 1] : vector<2x3xf32> to f32
  return %0 : f32
}
// CHECK-LABEL: func @vector_multi_reduction_to_scalar
//       CHECK:   return %{{.+}}
