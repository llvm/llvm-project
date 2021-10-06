// RUN: mlir-opt %s -test-vector-multi-reduction-lowering-patterns="use-outer-reductions" | FileCheck %s

func @vector_multi_reduction(%arg0: vector<2x4xf32>) -> vector<2xf32> {
    %0 = vector.multi_reduction #vector.kind<mul>, %arg0 [1] : vector<2x4xf32> to vector<2xf32>
    return %0 : vector<2xf32>
}

// CHECK-LABEL: func @vector_multi_reduction
//  CHECK-SAME:   %[[INPUT:.+]]: vector<2x4xf32>
//       CHECK:   %[[TRANSPOSED:.+]] = vector.transpose %[[INPUT]], [1, 0] : vector<2x4xf32> to vector<4x2xf32>
//       CHECK:   %[[V0:.+]] = vector.extract %[[TRANSPOSED]][0] : vector<4x2xf32>
//       CHECK:   %[[V1:.+]] = vector.extract %[[TRANSPOSED]][1] : vector<4x2xf32>
//       CHECK:   %[[RV01:.+]] = mulf %[[V1]], %[[V0]] : vector<2xf32>
//       CHECK:   %[[V2:.+]] = vector.extract %[[TRANSPOSED]][2] : vector<4x2xf32>
//       CHECK:   %[[RV012:.+]] = mulf %[[V2]], %[[RV01]] : vector<2xf32>
//       CHECK:   %[[V3:.+]] = vector.extract %[[TRANSPOSED]][3] : vector<4x2xf32>
//       CHECK:   %[[RESULT_VEC:.+]] = mulf %[[V3]], %[[RV012]] : vector<2xf32>
//       CHECK:   return %[[RESULT_VEC]] : vector<2xf32>

func @vector_multi_reduction_min(%arg0: vector<2x4xf32>) -> vector<2xf32> {
    %0 = vector.multi_reduction #vector.kind<minf>, %arg0 [1] : vector<2x4xf32> to vector<2xf32>
    return %0 : vector<2xf32>
}

// CHECK-LABEL: func @vector_multi_reduction_min
//  CHECK-SAME:   %[[INPUT:.+]]: vector<2x4xf32>
//       CHECK:   %[[TRANSPOSED:.+]] = vector.transpose %[[INPUT]], [1, 0] : vector<2x4xf32> to vector<4x2xf32>
//       CHECK:   %[[V0:.+]] = vector.extract %[[TRANSPOSED]][0] : vector<4x2xf32>
//       CHECK:   %[[V1:.+]] = vector.extract %[[TRANSPOSED]][1] : vector<4x2xf32>
//       CHECK:   %[[RV01:.+]] = minf %[[V1]], %[[V0]] : vector<2xf32>
//       CHECK:   %[[V2:.+]] = vector.extract %[[TRANSPOSED]][2] : vector<4x2xf32>
//       CHECK:   %[[RV012:.+]] = minf %[[V2]], %[[RV01]] : vector<2xf32>
//       CHECK:   %[[V3:.+]] = vector.extract %[[TRANSPOSED]][3] : vector<4x2xf32>
//       CHECK:   %[[RESULT_VEC:.+]] = minf %[[V3]], %[[RV012]] : vector<2xf32>
//       CHECK:   return %[[RESULT_VEC]] : vector<2xf32>

func @vector_multi_reduction_max(%arg0: vector<2x4xf32>) -> vector<2xf32> {
    %0 = vector.multi_reduction #vector.kind<maxf>, %arg0 [1] : vector<2x4xf32> to vector<2xf32>
    return %0 : vector<2xf32>
}

// CHECK-LABEL: func @vector_multi_reduction_max
//  CHECK-SAME:   %[[INPUT:.+]]: vector<2x4xf32>
//       CHECK:   %[[TRANSPOSED:.+]] = vector.transpose %[[INPUT]], [1, 0] : vector<2x4xf32> to vector<4x2xf32>
//       CHECK:   %[[V0:.+]] = vector.extract %[[TRANSPOSED]][0] : vector<4x2xf32>
//       CHECK:   %[[V1:.+]] = vector.extract %[[TRANSPOSED]][1] : vector<4x2xf32>
//       CHECK:   %[[RV01:.+]] = maxf %[[V1]], %[[V0]] : vector<2xf32>
//       CHECK:   %[[V2:.+]] = vector.extract %[[TRANSPOSED]][2] : vector<4x2xf32>
//       CHECK:   %[[RV012:.+]] = maxf %[[V2]], %[[RV01]] : vector<2xf32>
//       CHECK:   %[[V3:.+]] = vector.extract %[[TRANSPOSED]][3] : vector<4x2xf32>
//       CHECK:   %[[RESULT_VEC:.+]] = maxf %[[V3]], %[[RV012]] : vector<2xf32>
//       CHECK:   return %[[RESULT_VEC]] : vector<2xf32>

func @vector_multi_reduction_and(%arg0: vector<2x4xi32>) -> vector<2xi32> {
    %0 = vector.multi_reduction #vector.kind<and>, %arg0 [1] : vector<2x4xi32> to vector<2xi32>
    return %0 : vector<2xi32>
}

// CHECK-LABEL: func @vector_multi_reduction_and
//  CHECK-SAME:   %[[INPUT:.+]]: vector<2x4xi32>
//       CHECK:   %[[TRANSPOSED:.+]] = vector.transpose %[[INPUT]], [1, 0] : vector<2x4xi32> to vector<4x2xi32>
//       CHECK:   %[[V0:.+]] = vector.extract %[[TRANSPOSED]][0] : vector<4x2xi32>
//       CHECK:   %[[V1:.+]] = vector.extract %[[TRANSPOSED]][1] : vector<4x2xi32>
//       CHECK:   %[[RV01:.+]] = and %[[V1]], %[[V0]] : vector<2xi32>
//       CHECK:   %[[V2:.+]] = vector.extract %[[TRANSPOSED]][2] : vector<4x2xi32>
//       CHECK:   %[[RV012:.+]] = and %[[V2]], %[[RV01]] : vector<2xi32>
//       CHECK:   %[[V3:.+]] = vector.extract %[[TRANSPOSED]][3] : vector<4x2xi32>
//       CHECK:   %[[RESULT_VEC:.+]] = and %[[V3]], %[[RV012]] : vector<2xi32>
//       CHECK:   return %[[RESULT_VEC]] : vector<2xi32>

func @vector_multi_reduction_or(%arg0: vector<2x4xi32>) -> vector<2xi32> {
    %0 = vector.multi_reduction #vector.kind<or>, %arg0 [1] : vector<2x4xi32> to vector<2xi32>
    return %0 : vector<2xi32>
}

// CHECK-LABEL: func @vector_multi_reduction_or
//  CHECK-SAME:   %[[INPUT:.+]]: vector<2x4xi32>
//       CHECK:   %[[TRANSPOSED:.+]] = vector.transpose %[[INPUT]], [1, 0] : vector<2x4xi32> to vector<4x2xi32>
//       CHECK:   %[[V0:.+]] = vector.extract %[[TRANSPOSED]][0] : vector<4x2xi32>
//       CHECK:   %[[V1:.+]] = vector.extract %[[TRANSPOSED]][1] : vector<4x2xi32>
//       CHECK:   %[[RV01:.+]] = or %[[V1]], %[[V0]] : vector<2xi32>
//       CHECK:   %[[V2:.+]] = vector.extract %[[TRANSPOSED]][2] : vector<4x2xi32>
//       CHECK:   %[[RV012:.+]] = or %[[V2]], %[[RV01]] : vector<2xi32>
//       CHECK:   %[[V3:.+]] = vector.extract %[[TRANSPOSED]][3] : vector<4x2xi32>
//       CHECK:   %[[RESULT_VEC:.+]] = or %[[V3]], %[[RV012]] : vector<2xi32>
//       CHECK:   return %[[RESULT_VEC]] : vector<2xi32>

func @vector_multi_reduction_xor(%arg0: vector<2x4xi32>) -> vector<2xi32> {
    %0 = vector.multi_reduction #vector.kind<xor>, %arg0 [1] : vector<2x4xi32> to vector<2xi32>
    return %0 : vector<2xi32>
}

// CHECK-LABEL: func @vector_multi_reduction_xor
//  CHECK-SAME:   %[[INPUT:.+]]: vector<2x4xi32>
//       CHECK:   %[[TRANSPOSED:.+]] = vector.transpose %[[INPUT]], [1, 0] : vector<2x4xi32> to vector<4x2xi32>
//       CHECK:   %[[V0:.+]] = vector.extract %[[TRANSPOSED]][0] : vector<4x2xi32>
//       CHECK:   %[[V1:.+]] = vector.extract %[[TRANSPOSED]][1] : vector<4x2xi32>
//       CHECK:   %[[RV01:.+]] = xor %[[V1]], %[[V0]] : vector<2xi32>
//       CHECK:   %[[V2:.+]] = vector.extract %[[TRANSPOSED]][2] : vector<4x2xi32>
//       CHECK:   %[[RV012:.+]] = xor %[[V2]], %[[RV01]] : vector<2xi32>
//       CHECK:   %[[V3:.+]] = vector.extract %[[TRANSPOSED]][3] : vector<4x2xi32>
//       CHECK:   %[[RESULT_VEC:.+]] = xor %[[V3]], %[[RV012]] : vector<2xi32>
//       CHECK:   return %[[RESULT_VEC]] : vector<2xi32>


func @vector_reduction_outer(%arg0: vector<2x3x4x5xi32>) -> vector<2x3xi32> {
    %0 = vector.multi_reduction #vector.kind<add>, %arg0 [2, 3] : vector<2x3x4x5xi32> to vector<2x3xi32>
    return %0 : vector<2x3xi32>
}

// CHECK-LABEL: func @vector_reduction_outer
//  CHECK-SAME:   %[[INPUT:.+]]: vector<2x3x4x5xi32>
//       CHECK:   %[[TRANSPOSED:.+]] = vector.transpose %[[INPUT]], [2, 3, 0, 1] : vector<2x3x4x5xi32> to vector<4x5x2x3xi32>
//       CHECK:   %[[RESHAPED:.+]] = vector.shape_cast %[[TRANSPOSED]] : vector<4x5x2x3xi32> to vector<20x6xi32>
//       CHECK:   %[[V0:.+]] = vector.extract %[[RESHAPED]][0] : vector<20x6xi32>
//       CHECK:   %[[V1:.+]] = vector.extract %[[RESHAPED]][1] : vector<20x6xi32>
//       CHECK:   %[[R0:.+]] = addi %[[V1]], %[[V0]] : vector<6xi32>
//       CHECK:   %[[V2:.+]] = vector.extract %[[RESHAPED]][2] : vector<20x6xi32>
//       CHECK:   %[[R1:.+]] = addi %[[V2]], %[[R0]] : vector<6xi32>
//       CHECK:   %[[V3:.+]] = vector.extract %[[RESHAPED]][3] : vector<20x6xi32>
//       CHECK:   %[[R2:.+]] = addi %[[V3]], %[[R1]] : vector<6xi32>
//       CHECK:   %[[V4:.+]] = vector.extract %[[RESHAPED]][4] : vector<20x6xi32>
//       CHECK:   %[[R3:.+]] = addi %[[V4]], %[[R2]] : vector<6xi32>
//       CHECK:   %[[V5:.+]] = vector.extract %[[RESHAPED]][5] : vector<20x6xi32>
//       CHECK:   %[[R4:.+]] = addi %[[V5]], %[[R3]] : vector<6xi32>
//       CHECK:   %[[V6:.+]] = vector.extract %[[RESHAPED]][6] : vector<20x6xi32>
//       CHECK:   %[[R5:.+]] = addi %[[V6]], %[[R4]] : vector<6xi32>
//       CHECK:   %[[V7:.+]] = vector.extract %[[RESHAPED]][7] : vector<20x6xi32>
//       CHECK:   %[[R6:.+]] = addi %[[V7]], %[[R5]] : vector<6xi32>
//       CHECK:   %[[V8:.+]] = vector.extract %[[RESHAPED]][8] : vector<20x6xi32>
//       CHECK:   %[[R7:.+]] = addi %[[V8]], %[[R6]] : vector<6xi32>
//       CHECK:   %[[V9:.+]] = vector.extract %[[RESHAPED]][9] : vector<20x6xi32>
//       CHECK:   %[[R8:.+]] = addi %[[V9]], %[[R7]] : vector<6xi32>
//       CHECK:   %[[V10:.+]] = vector.extract %[[RESHAPED]][10] : vector<20x6xi32>
//       CHECK:   %[[R9:.+]] = addi %[[V10]], %[[R8]] : vector<6xi32>
//       CHECK:   %[[V11:.+]] = vector.extract %[[RESHAPED]][11] : vector<20x6xi32>
//       CHECK:   %[[R10:.+]] = addi %[[V11]], %[[R9]] : vector<6xi32>
//       CHECK:   %[[V12:.+]] = vector.extract %[[RESHAPED]][12] : vector<20x6xi32>
//       CHECK:   %[[R11:.+]] = addi %[[V12]], %[[R10]] : vector<6xi32>
//       CHECK:   %[[V13:.+]] = vector.extract %[[RESHAPED]][13] : vector<20x6xi32>
//       CHECK:   %[[R12:.+]] = addi %[[V13]], %[[R11]] : vector<6xi32>
//       CHECK:   %[[V14:.+]] = vector.extract %[[RESHAPED]][14] : vector<20x6xi32>
//       CHECK:   %[[R13:.+]] = addi %[[V14]], %[[R12]] : vector<6xi32>
//       CHECK:   %[[V15:.+]] = vector.extract %[[RESHAPED]][15] : vector<20x6xi32>
//       CHECK:   %[[R14:.+]] = addi %[[V15]], %[[R13]] : vector<6xi32>
//       CHECK:   %[[V16:.+]] = vector.extract %[[RESHAPED]][16] : vector<20x6xi32>
//       CHECK:   %[[R15:.+]] = addi %[[V16]], %[[R14]] : vector<6xi32>
//       CHECK:   %[[V17:.+]] = vector.extract %[[RESHAPED]][17] : vector<20x6xi32>
//       CHECK:   %[[R16:.+]] = addi %[[V17]], %[[R15]] : vector<6xi32>
//       CHECK:   %[[V18:.+]] = vector.extract %[[RESHAPED]][18] : vector<20x6xi32>
//       CHECK:   %[[R17:.+]] = addi %[[V18]], %[[R16]] : vector<6xi32>
//       CHECK:   %[[V19:.+]] = vector.extract %[[RESHAPED]][19] : vector<20x6xi32>
//       CHECK:   %[[R18:.+]] = addi %[[V19]], %[[R17]] : vector<6xi32>
//       CHECK:   %[[RESULT_VEC:.+]] = vector.shape_cast %[[R18]] : vector<6xi32> to vector<2x3xi32>
//       CHECK:   return %[[RESULT_VEC]] : vector<2x3xi32>
