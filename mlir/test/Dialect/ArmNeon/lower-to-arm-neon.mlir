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
// CHECK-DAG:  %[[VAL_3:.*]] = arith.constant dense<0> : vector<4x4xi32>
// CHECK-DAG:  %[[VAL_4:.*]] = vector.extract_strided_slice %[[VAL_0]] {offsets = [0, 0], sizes = [2, 8], strides = [1, 1]} : vector<4x8xi8> to vector<2x8xi8>
// CHECK-DAG:  %[[VAL_5:.*]] = vector.extract_strided_slice %[[VAL_1]] {offsets = [0, 0], sizes = [2, 8], strides = [1, 1]} : vector<4x8xi8> to vector<2x8xi8>
// CHECK-DAG:  %[[VAL_6:.*]] = vector.extract_strided_slice %[[VAL_2]] {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xi32> to vector<2x2xi32>
// CHECK-DAG:  %[[VAL_7:.*]] = vector.shape_cast %[[VAL_4]] : vector<2x8xi8> to vector<16xi8>
// CHECK-DAG:  %[[VAL_8:.*]] = vector.shape_cast %[[VAL_5]] : vector<2x8xi8> to vector<16xi8>
// CHECK-DAG:  %[[VAL_9:.*]] = vector.shape_cast %[[VAL_6]] : vector<2x2xi32> to vector<4xi32>
// CHECK-DAG:  %[[VAL_10:.*]] = arm_neon.intr.smmla %[[VAL_9]], %[[VAL_7]], %[[VAL_8]] : vector<16xi8> to vector<4xi32>
// CHECK-DAG:  %[[VAL_11:.*]] = vector.shape_cast %[[VAL_10]] : vector<4xi32> to vector<2x2xi32>
// CHECK-DAG:  %[[VAL_12:.*]] = vector.insert_strided_slice %[[VAL_11]], %[[VAL_3]] {offsets = [0, 0], strides = [1, 1]} : vector<2x2xi32> into vector<4x4xi32>
// CHECK-DAG:  %[[VAL_13:.*]] = vector.extract_strided_slice %[[VAL_0]] {offsets = [0, 0], sizes = [2, 8], strides = [1, 1]} : vector<4x8xi8> to vector<2x8xi8>
// CHECK-DAG:  %[[VAL_14:.*]] = vector.extract_strided_slice %[[VAL_1]] {offsets = [2, 0], sizes = [2, 8], strides = [1, 1]} : vector<4x8xi8> to vector<2x8xi8>
// CHECK-DAG:  %[[VAL_15:.*]] = vector.extract_strided_slice %[[VAL_2]] {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xi32> to vector<2x2xi32>
// CHECK-DAG:  %[[VAL_16:.*]] = vector.shape_cast %[[VAL_13]] : vector<2x8xi8> to vector<16xi8>
// CHECK-DAG:  %[[VAL_17:.*]] = vector.shape_cast %[[VAL_14]] : vector<2x8xi8> to vector<16xi8>
// CHECK-DAG:  %[[VAL_18:.*]] = vector.shape_cast %[[VAL_15]] : vector<2x2xi32> to vector<4xi32>
// CHECK-DAG:  %[[VAL_19:.*]] = arm_neon.intr.smmla %[[VAL_18]], %[[VAL_16]], %[[VAL_17]] : vector<16xi8> to vector<4xi32>
// CHECK-DAG:  %[[VAL_20:.*]] = vector.shape_cast %[[VAL_19]] : vector<4xi32> to vector<2x2xi32>
// CHECK-DAG:  %[[VAL_21:.*]] = vector.insert_strided_slice %[[VAL_20]], %[[VAL_12]] {offsets = [0, 2], strides = [1, 1]} : vector<2x2xi32> into vector<4x4xi32>
// CHECK-DAG:  %[[VAL_22:.*]] = vector.extract_strided_slice %[[VAL_0]] {offsets = [2, 0], sizes = [2, 8], strides = [1, 1]} : vector<4x8xi8> to vector<2x8xi8>
// CHECK-DAG:  %[[VAL_23:.*]] = vector.extract_strided_slice %[[VAL_1]] {offsets = [0, 0], sizes = [2, 8], strides = [1, 1]} : vector<4x8xi8> to vector<2x8xi8>
// CHECK-DAG:  %[[VAL_24:.*]] = vector.extract_strided_slice %[[VAL_2]] {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xi32> to vector<2x2xi32>
// CHECK-DAG:  %[[VAL_25:.*]] = vector.shape_cast %[[VAL_22]] : vector<2x8xi8> to vector<16xi8>
// CHECK-DAG:  %[[VAL_26:.*]] = vector.shape_cast %[[VAL_23]] : vector<2x8xi8> to vector<16xi8>
// CHECK-DAG:  %[[VAL_27:.*]] = vector.shape_cast %[[VAL_24]] : vector<2x2xi32> to vector<4xi32>
// CHECK-DAG:  %[[VAL_28:.*]] = arm_neon.intr.smmla %[[VAL_27]], %[[VAL_25]], %[[VAL_26]] : vector<16xi8> to vector<4xi32>
// CHECK-DAG:  %[[VAL_29:.*]] = vector.shape_cast %[[VAL_28]] : vector<4xi32> to vector<2x2xi32>
// CHECK-DAG:  %[[VAL_30:.*]] = vector.insert_strided_slice %[[VAL_29]], %[[VAL_21]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xi32> into vector<4x4xi32>
// CHECK-DAG:  %[[VAL_31:.*]] = vector.extract_strided_slice %[[VAL_0]] {offsets = [2, 0], sizes = [2, 8], strides = [1, 1]} : vector<4x8xi8> to vector<2x8xi8>
// CHECK-DAG:  %[[VAL_32:.*]] = vector.extract_strided_slice %[[VAL_1]] {offsets = [2, 0], sizes = [2, 8], strides = [1, 1]} : vector<4x8xi8> to vector<2x8xi8>
// CHECK-DAG:  %[[VAL_33:.*]] = vector.extract_strided_slice %[[VAL_2]] {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xi32> to vector<2x2xi32>
// CHECK-DAG:  %[[VAL_34:.*]] = vector.shape_cast %[[VAL_31]] : vector<2x8xi8> to vector<16xi8>
// CHECK-DAG:  %[[VAL_35:.*]] = vector.shape_cast %[[VAL_32]] : vector<2x8xi8> to vector<16xi8>
// CHECK-DAG:  %[[VAL_36:.*]] = vector.shape_cast %[[VAL_33]] : vector<2x2xi32> to vector<4xi32>
// CHECK-DAG:  %[[VAL_37:.*]] = arm_neon.intr.smmla %[[VAL_36]], %[[VAL_34]], %[[VAL_35]] : vector<16xi8> to vector<4xi32>
// CHECK-DAG:  %[[VAL_38:.*]] = vector.shape_cast %[[VAL_37]] : vector<4xi32> to vector<2x2xi32>
// CHECK-DAG:  %[[VAL_39:.*]] = vector.insert_strided_slice %[[VAL_38]], %[[VAL_30]] {offsets = [2, 2], strides = [1, 1]} : vector<2x2xi32> into vector<4x4xi32>
// CHECK-DAG:  return %[[VAL_39]] : vector<4x4xi32>
// CHECK-DAG:  }
func.func @test_lower_vector_arm_neon_unroll(%lhs: vector<4x8xi8>, %rhs: vector<4x8xi8>, %acc : vector<4x4xi32>) -> vector<4x4xi32> {
  %lhs_extsi = arith.extsi %lhs : vector<4x8xi8> to vector<4x8xi32>
  %rhs_extsi = arith.extsi %rhs : vector<4x8xi8> to vector<4x8xi32>
  %res = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs_extsi, %rhs_extsi, %acc : vector<4x8xi32>, vector<4x8xi32> into vector<4x4xi32>
  return %res : vector<4x4xi32>
}

// -----

// CHECK-LABEL:   func.func @test_lower_vector_arm_neon_mixed_unroll(
// CHECK-SAME:                                                       %[[VAL_0:.*]]: vector<4x8xi8>,
// CHECK-SAME:                                                       %[[VAL_1:.*]]: vector<2x8xi4>,
// CHECK-SAME:                                                       %[[VAL_2:.*]]: vector<4x2xi32>) -> vector<4x2xi32> {
// CHECK-DAG:  %[[VAL_3:.*]] = arith.constant dense<0> : vector<4x2xi32>
// CHECK-DAG:  %[[VAL_4:.*]] = arith.extsi %[[VAL_1]] : vector<2x8xi4> to vector<2x8xi8>
// CHECK-DAG:  %[[VAL_5:.*]] = vector.extract_strided_slice %[[VAL_0]] {offsets = [0, 0], sizes = [2, 8], strides = [1, 1]} : vector<4x8xi8> to vector<2x8xi8>
// CHECK-DAG:  %[[VAL_6:.*]] = vector.extract_strided_slice %[[VAL_2]] {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xi32> to vector<2x2xi32>
// CHECK-DAG:  %[[VAL_7:.*]] = vector.shape_cast %[[VAL_5]] : vector<2x8xi8> to vector<16xi8>
// CHECK-DAG:  %[[VAL_8:.*]] = vector.shape_cast %[[VAL_4]] : vector<2x8xi8> to vector<16xi8>
// CHECK-DAG:  %[[VAL_9:.*]] = vector.shape_cast %[[VAL_6]] : vector<2x2xi32> to vector<4xi32>
// CHECK-DAG:  %[[VAL_10:.*]] = arm_neon.intr.smmla %[[VAL_9]], %[[VAL_7]], %[[VAL_8]] : vector<16xi8> to vector<4xi32>
// CHECK-DAG:  %[[VAL_11:.*]] = vector.shape_cast %[[VAL_10]] : vector<4xi32> to vector<2x2xi32>
// CHECK-DAG:  %[[VAL_12:.*]] = vector.insert_strided_slice %[[VAL_11]], %[[VAL_3]] {offsets = [0, 0], strides = [1, 1]} : vector<2x2xi32> into vector<4x2xi32>
// CHECK-DAG:  %[[VAL_13:.*]] = vector.extract_strided_slice %[[VAL_0]] {offsets = [2, 0], sizes = [2, 8], strides = [1, 1]} : vector<4x8xi8> to vector<2x8xi8>
// CHECK-DAG:  %[[VAL_14:.*]] = vector.extract_strided_slice %[[VAL_2]] {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xi32> to vector<2x2xi32>
// CHECK-DAG:  %[[VAL_15:.*]] = vector.shape_cast %[[VAL_13]] : vector<2x8xi8> to vector<16xi8>
// CHECK-DAG:  %[[VAL_16:.*]] = vector.shape_cast %[[VAL_4]] : vector<2x8xi8> to vector<16xi8>
// CHECK-DAG:  %[[VAL_17:.*]] = vector.shape_cast %[[VAL_14]] : vector<2x2xi32> to vector<4xi32>
// CHECK-DAG:  %[[VAL_18:.*]] = arm_neon.intr.smmla %[[VAL_17]], %[[VAL_15]], %[[VAL_16]] : vector<16xi8> to vector<4xi32>
// CHECK-DAG:  %[[VAL_19:.*]] = vector.shape_cast %[[VAL_18]] : vector<4xi32> to vector<2x2xi32>
// CHECK-DAG:  %[[VAL_20:.*]] = vector.insert_strided_slice %[[VAL_19]], %[[VAL_12]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xi32> into vector<4x2xi32>
// CHECK-DAG:  return %[[VAL_20]] : vector<4x2xi32>
// CHECK-DAG:  }
func.func @test_lower_vector_arm_neon_mixed_unroll(%lhs: vector<4x8xi8>, %rhs: vector<2x8xi4>, %acc : vector<4x2xi32>) -> vector<4x2xi32> {
  %lhs_extsi = arith.extsi %lhs : vector<4x8xi8> to vector<4x8xi32>
  %rhs_extsi = arith.extsi %rhs : vector<2x8xi4> to vector<2x8xi32>
  %res = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs_extsi, %rhs_extsi, %acc : vector<4x8xi32>, vector<2x8xi32> into vector<4x2xi32>
  return %res : vector<4x2xi32>
}

// -----

// CHECK-LABEL:   func.func @test_lower_vector_arm_neon_unroll_incompatible_shape(
// CHECK-DAG:  %[[result:.*]] = vector.contract
func.func @test_lower_vector_arm_neon_unroll_incompatible_shape(%lhs: vector<4x12xi8>, %rhs: vector<4x12xi8>, %acc : vector<4x4xi32>) -> vector<4x4xi32> {
  %lhs_extsi = arith.extsi %lhs : vector<4x12xi8> to vector<4x12xi32>
  %rhs_extsi = arith.extsi %rhs : vector<4x12xi8> to vector<4x12xi32>
  %res = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs_extsi, %rhs_extsi, %acc : vector<4x12xi32>, vector<4x12xi32> into vector<4x4xi32>
  return %res : vector<4x4xi32>
}
