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

