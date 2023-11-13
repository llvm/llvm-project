// RUN: mlir-opt -lower-vector-mask -split-input-file %s | FileCheck %s

func.func @vector_transfer_read(%t0: tensor<?xf32>, %idx: index, %m0: vector<16xi1>) -> vector<16xf32> {
  %ft0 = arith.constant 0.0 : f32
  %0 = vector.mask %m0 { vector.transfer_read %t0[%idx], %ft0 : tensor<?xf32>, vector<16xf32> } : vector<16xi1> -> vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL:   func.func @vector_transfer_read(
// CHECK-SAME:                                    %[[VAL_0:.*]]: tensor<?xf32>,
// CHECK-SAME:                                    %[[VAL_1:.*]]: index,
// CHECK-SAME:                                    %[[VAL_2:.*]]: vector<16xi1>) -> vector<16xf32> {
// CHECK-NOT:       vector.mask
// CHECK:           %[[VAL_4:.*]] = vector.transfer_read {{.*}}, %[[VAL_2]] : tensor<?xf32>, vector<16xf32>
// CHECK:           return %[[VAL_4]] : vector<16xf32>
// CHECK:         }

// -----

func.func @vector_transfer_write_on_memref(%val: vector<16xf32>, %t0: memref<?xf32>, %idx: index, %m0: vector<16xi1>) {
  vector.mask %m0 { vector.transfer_write %val, %t0[%idx] : vector<16xf32>, memref<?xf32> } : vector<16xi1>
  return
}

// CHECK-LABEL:   func.func @vector_transfer_write_on_memref(
// CHECK-SAME:                                               %[[VAL_0:.*]]: vector<16xf32>,
// CHECK-SAME:                                               %[[VAL_1:.*]]: memref<?xf32>,
// CHECK-SAME:                                               %[[VAL_2:.*]]: index,
// CHECK-SAME:                                               %[[VAL_3:.*]]: vector<16xi1>) {
  //CHECK-NOT:      vector.mask
// CHECK:           vector.transfer_write %[[VAL_0]], {{.*}}, %[[VAL_3]] : vector<16xf32>, memref<?xf32>
// CHECK:           return
// CHECK:         }

// -----

func.func @vector_transfer_write_on_tensor(%val: vector<16xf32>, %t0: tensor<?xf32>, %idx: index, %m0: vector<16xi1>) -> tensor<?xf32> {
  %res = vector.mask %m0 { vector.transfer_write %val, %t0[%idx] : vector<16xf32>, tensor<?xf32> } : vector<16xi1> -> tensor<?xf32>
  return %res : tensor<?xf32>
}

// CHECK-LABEL:   func.func @vector_transfer_write_on_tensor(
// CHECK-SAME:                                               %[[VAL_0:.*]]: vector<16xf32>,
// CHECK-SAME:                                               %[[VAL_1:.*]]: tensor<?xf32>,
// CHECK-SAME:                                               %[[VAL_2:.*]]: index,
// CHECK-SAME:                                               %[[VAL_3:.*]]: vector<16xi1>) -> tensor<?xf32> {
// CHECK:           %[[VAL_4:.*]] = vector.transfer_write %[[VAL_0]], {{.*}}, %[[VAL_3]] : vector<16xf32>, tensor<?xf32>
// CHECK:           return %[[VAL_4]] : tensor<?xf32>
// CHECK:         }

// -----

func.func @vector_gather(%arg0: tensor<64xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c3 = arith.constant 3 : index
  %0 = vector.create_mask %c3 : vector<4xi1>
  %1 = vector.mask %0 { vector.transfer_read %arg1[%c0], %cst {in_bounds = [true]} : tensor<3xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
  %cst_0 = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
  %cst_1 = arith.constant dense<true> : vector<4xi1>
  %cst_2 = arith.constant dense<0.000000e+00> : vector<4xf32>
  %c0_3 = arith.constant 0 : index
  %2 = vector.mask %0 { vector.gather %arg0[%c0_3] [%cst_0], %cst_1, %cst_2 : tensor<64xf32>, vector<4xindex>, vector<4xi1>, vector<4xf32> into vector<4xf32> } : vector<4xi1> -> vector<4xf32>
  %c0_4 = arith.constant 0 : index
  %3 = vector.mask %0 { vector.transfer_write %2, %arg1[%c0_4] {in_bounds = [true]} : vector<4xf32>, tensor<3xf32> } : vector<4xi1> -> tensor<3xf32>
  return %3 : tensor<3xf32>
}

// CHECK-LABEL:   func.func @vector_gather(
// CHECK-SAME:                             %[[VAL_0:.*]]: tensor<64xf32>,
// CHECK-SAME:                             %[[VAL_1:.*]]: tensor<3xf32>) -> tensor<3xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK:           %[[VAL_3:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_6:.*]] = vector.create_mask %[[VAL_5]] : vector<4xi1>
// CHECK:           %[[VAL_7:.*]] = vector.gather %[[VAL_0]][%[[VAL_4]]] [%[[VAL_3]]], %[[VAL_6]], %[[VAL_2]] : tensor<64xf32>, vector<4xindex>, vector<4xi1>, vector<4xf32> into vector<4xf32>
// CHECK:           %[[VAL_8:.*]] = vector.transfer_write %[[VAL_7]], %[[VAL_1]][%[[VAL_4]]], %[[VAL_6]] {in_bounds = [true]} : vector<4xf32>, tensor<3xf32>

// -----

// CHECK-LABEL: func @empty_vector_mask_with_return
//  CHECK-SAME:     %[[IN:.*]]: vector<8xf32>
func.func @empty_vector_mask_with_return(%a : vector<8xf32>, %mask : vector<8xi1>) -> vector<8xf32> {
//   CHECK-NOT:   vector.mask
//       CHECK:   return %[[IN]] : vector<8xf32>
  %0 = vector.mask %mask { vector.yield %a : vector<8xf32> } : vector<8xi1> -> vector<8xf32>
  return %0 : vector<8xf32>
}

