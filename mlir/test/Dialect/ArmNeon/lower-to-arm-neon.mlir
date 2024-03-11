// RUN: mlir-opt -test-lower-to-arm-neon -verify-diagnostics -split-input-file %s | FileCheck %s

// CHECK-LABEL: test_lower_vector_arm_neon_mixed_types
// CHECK-SAME:    %[[A0:.*]]: vector<2x8xi8>, %[[A1:.*]]: vector<2x8xi4>, %[[A2:.*]]: vector<2x2xi32>
// CHECK-DAG: %[[D0:.*]] = arith.extsi %[[A1]] : vector<2x8xi4> to vector<2x8xi8>
// CHECK-DAG: %[[D1:.*]] = vector.shape_cast %[[A0]] : vector<2x8xi8> to vector<16xi8>
// CHECK-DAG: %[[D2:.*]] = vector.shape_cast %[[D0]] : vector<2x8xi8> to vector<16xi8>
// CHECK-DAG: %[[D3:.*]] = vector.shape_cast %[[A2]] : vector<2x2xi32> to vector<4xi32>
// CHECK-DAG: %[[D4:.*]] = arm_neon.intr.smmla %[[D3]], %[[D1]], %[[D2]] : vector<16xi8> to vector<4xi32>
// CHECK-DAG: %[[D5:.*]] = vector.shape_cast %[[D4]] : vector<4xi32> to vector<2x2xi32>
func.func @test_lower_vector_arm_neon_mixed_types(%lhs: vector<2x8xi8>, %rhs: vector<2x8xi4>, %acc : vector<2x2xi32>) -> vector<2x2xi32> {
  %lhs_extsi = arith.extsi %lhs : vector<2x8xi8> to vector<2x8xi32>
  %rhs_extsi = arith.extsi %rhs : vector<2x8xi4> to vector<2x8xi32>
  %res = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs_extsi, %rhs_extsi, %acc : vector<2x8xi32>, vector<2x8xi32> into vector<2x2xi32>
  return %res : vector<2x2xi32>
}

// -----

// CHECK-LABEL: test_lower_vector_arm_neon_same_types
// CHECK-SAME:    %[[A0:.*]]: vector<2x8xi8>, %[[A1:.*]]: vector<2x8xi8>, %[[A2:.*]]: vector<2x2xi32>
// CHECK-DAG: %[[D0:.*]] = vector.shape_cast %[[A0]] : vector<2x8xi8> to vector<16xi8>
// CHECK-DAG: %[[D1:.*]] = vector.shape_cast %[[A1]] : vector<2x8xi8> to vector<16xi8>
// CHECK-DAG: %[[D2:.*]] = vector.shape_cast %[[A2]] : vector<2x2xi32> to vector<4xi32>
// CHECK-DAG: %[[D3:.*]] = arm_neon.intr.smmla %[[D2]], %[[D0]], %[[D1]] : vector<16xi8> to vector<4xi32>
// CHECK-DAG: %[[D4:.*]] = vector.shape_cast %[[D3]] : vector<4xi32> to vector<2x2xi32>
func.func @test_lower_vector_arm_neon_same_types(%lhs: vector<2x8xi8>, %rhs: vector<2x8xi8>, %acc : vector<2x2xi32>) -> vector<2x2xi32> {
  %lhs_extsi = arith.extsi %lhs : vector<2x8xi8> to vector<2x8xi32>
  %rhs_extsi = arith.extsi %rhs : vector<2x8xi8> to vector<2x8xi32>
  %res = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs_extsi, %rhs_extsi, %acc : vector<2x8xi32>, vector<2x8xi32> into vector<2x2xi32>
  return %res : vector<2x2xi32>
}

// -----

// CHECK-LABEL: test_lower_vector_arm_neon_without_extsi
// CHECK-SAME:    %[[A0:.*]]: vector<2x8xi32>, %[[A1:.*]]: vector<2x8xi32>, %[[A2:.*]]: vector<2x2xi32>
// CHECK-DAG: %[[D0:.*]] = vector.contract
func.func @test_lower_vector_arm_neon_without_extsi(%lhs: vector<2x8xi32>, %rhs: vector<2x8xi32>, %acc : vector<2x2xi32>) -> vector<2x2xi32> {
  %res = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %rhs, %acc : vector<2x8xi32>, vector<2x8xi32> into vector<2x2xi32>
  return %res : vector<2x2xi32>
}

// -----

// CHECK-LABEL: test_lower_vector_arm_neon_unroll
// CHECK-SAME: %[[VAL_0:.*]]: vector<4x8xi8>, %[[VAL_1:.*]]: vector<4x8xi8>, %[[VAL_2:.*]]: vector<4x4xi32>
// CHECK:  %[[VAL_3:.*]] = arith.constant dense<0> : vector<4x4xi32>
// CHECK:  %[[VAL_4:.*]] = vector.extract_strided_slice %[[VAL_0]] {offsets = [0, 0], sizes = [2, 8], strides = [1, 1]} : vector<4x8xi8> to vector<2x8xi8>
// CHECK:  %[[VAL_5:.*]] = vector.extract_strided_slice %[[VAL_1]] {offsets = [0, 0], sizes = [2, 8], strides = [1, 1]} : vector<4x8xi8> to vector<2x8xi8>
// CHECK:  %[[VAL_6:.*]] = vector.extract_strided_slice %[[VAL_2]] {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xi32> to vector<2x2xi32>
// CHECK:  %[[VAL_7:.*]] = vector.shape_cast %[[VAL_4]] : vector<2x8xi8> to vector<16xi8>
// CHECK:  %[[VAL_8:.*]] = vector.shape_cast %[[VAL_5]] : vector<2x8xi8> to vector<16xi8>
// CHECK:  %[[VAL_9:.*]] = vector.shape_cast %[[VAL_6]] : vector<2x2xi32> to vector<4xi32>
// CHECK:  %[[VAL_10:.*]] = arm_neon.intr.smmla %[[VAL_9]], %[[VAL_7]], %[[VAL_8]] : vector<16xi8> to vector<4xi32>
// CHECK:  %[[VAL_11:.*]] = vector.shape_cast %[[VAL_10]] : vector<4xi32> to vector<2x2xi32>
// CHECK:  %[[VAL_12:.*]] = vector.insert_strided_slice %[[VAL_11]], %[[VAL_3]] {offsets = [0, 0], strides = [1, 1]} : vector<2x2xi32> into vector<4x4xi32>
// CHECK:  %[[VAL_13:.*]] = vector.extract_strided_slice %[[VAL_0]] {offsets = [0, 0], sizes = [2, 8], strides = [1, 1]} : vector<4x8xi8> to vector<2x8xi8>
// CHECK:  %[[VAL_14:.*]] = vector.extract_strided_slice %[[VAL_1]] {offsets = [2, 0], sizes = [2, 8], strides = [1, 1]} : vector<4x8xi8> to vector<2x8xi8>
// CHECK:  %[[VAL_15:.*]] = vector.extract_strided_slice %[[VAL_2]] {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xi32> to vector<2x2xi32>
// CHECK:  %[[VAL_16:.*]] = vector.shape_cast %[[VAL_13]] : vector<2x8xi8> to vector<16xi8>
// CHECK:  %[[VAL_17:.*]] = vector.shape_cast %[[VAL_14]] : vector<2x8xi8> to vector<16xi8>
// CHECK:  %[[VAL_18:.*]] = vector.shape_cast %[[VAL_15]] : vector<2x2xi32> to vector<4xi32>
// CHECK:  %[[VAL_19:.*]] = arm_neon.intr.smmla %[[VAL_18]], %[[VAL_16]], %[[VAL_17]] : vector<16xi8> to vector<4xi32>
// CHECK:  %[[VAL_20:.*]] = vector.shape_cast %[[VAL_19]] : vector<4xi32> to vector<2x2xi32>
// CHECK:  %[[VAL_21:.*]] = vector.insert_strided_slice %[[VAL_20]], %[[VAL_12]] {offsets = [0, 2], strides = [1, 1]} : vector<2x2xi32> into vector<4x4xi32>
// CHECK:  %[[VAL_22:.*]] = vector.extract_strided_slice %[[VAL_0]] {offsets = [2, 0], sizes = [2, 8], strides = [1, 1]} : vector<4x8xi8> to vector<2x8xi8>
// CHECK:  %[[VAL_23:.*]] = vector.extract_strided_slice %[[VAL_1]] {offsets = [0, 0], sizes = [2, 8], strides = [1, 1]} : vector<4x8xi8> to vector<2x8xi8>
// CHECK:  %[[VAL_24:.*]] = vector.extract_strided_slice %[[VAL_2]] {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xi32> to vector<2x2xi32>
// CHECK:  %[[VAL_25:.*]] = vector.shape_cast %[[VAL_22]] : vector<2x8xi8> to vector<16xi8>
// CHECK:  %[[VAL_26:.*]] = vector.shape_cast %[[VAL_23]] : vector<2x8xi8> to vector<16xi8>
// CHECK:  %[[VAL_27:.*]] = vector.shape_cast %[[VAL_24]] : vector<2x2xi32> to vector<4xi32>
// CHECK:  %[[VAL_28:.*]] = arm_neon.intr.smmla %[[VAL_27]], %[[VAL_25]], %[[VAL_26]] : vector<16xi8> to vector<4xi32>
// CHECK:  %[[VAL_29:.*]] = vector.shape_cast %[[VAL_28]] : vector<4xi32> to vector<2x2xi32>
// CHECK:  %[[VAL_30:.*]] = vector.insert_strided_slice %[[VAL_29]], %[[VAL_21]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xi32> into vector<4x4xi32>
// CHECK:  %[[VAL_31:.*]] = vector.extract_strided_slice %[[VAL_0]] {offsets = [2, 0], sizes = [2, 8], strides = [1, 1]} : vector<4x8xi8> to vector<2x8xi8>
// CHECK:  %[[VAL_32:.*]] = vector.extract_strided_slice %[[VAL_1]] {offsets = [2, 0], sizes = [2, 8], strides = [1, 1]} : vector<4x8xi8> to vector<2x8xi8>
// CHECK:  %[[VAL_33:.*]] = vector.extract_strided_slice %[[VAL_2]] {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xi32> to vector<2x2xi32>
// CHECK:  %[[VAL_34:.*]] = vector.shape_cast %[[VAL_31]] : vector<2x8xi8> to vector<16xi8>
// CHECK:  %[[VAL_35:.*]] = vector.shape_cast %[[VAL_32]] : vector<2x8xi8> to vector<16xi8>
// CHECK:  %[[VAL_36:.*]] = vector.shape_cast %[[VAL_33]] : vector<2x2xi32> to vector<4xi32>
// CHECK:  %[[VAL_37:.*]] = arm_neon.intr.smmla %[[VAL_36]], %[[VAL_34]], %[[VAL_35]] : vector<16xi8> to vector<4xi32>
// CHECK:  %[[VAL_38:.*]] = vector.shape_cast %[[VAL_37]] : vector<4xi32> to vector<2x2xi32>
// CHECK:  %[[VAL_39:.*]] = vector.insert_strided_slice %[[VAL_38]], %[[VAL_30]] {offsets = [2, 2], strides = [1, 1]} : vector<2x2xi32> into vector<4x4xi32>
// CHECK:  return %[[VAL_39]] : vector<4x4xi32>
// CHECK:  }
func.func @test_lower_vector_arm_neon_unroll(%lhs: vector<4x8xi8>, %rhs: vector<4x8xi8>, %acc : vector<4x4xi32>) -> vector<4x4xi32> {
  %lhs_extsi = arith.extsi %lhs : vector<4x8xi8> to vector<4x8xi32>
  %rhs_extsi = arith.extsi %rhs : vector<4x8xi8> to vector<4x8xi32>
  %res = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs_extsi, %rhs_extsi, %acc : vector<4x8xi32>, vector<4x8xi32> into vector<4x4xi32>
  return %res : vector<4x4xi32>
}
