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
