// RUN: mlir-opt -canonicalize -split-input-file -verify-diagnostics %s | mlir-opt | FileCheck %s

// -----

// CHECK-LABEL: @cast_vector_to_tile__cast_tile_to_vector
// CHECK-SAME: %[[TILE_ID:.*]]: i8
func.func @cast_vector_to_tile__cast_tile_to_vector(%tile_id_0 : i8) -> i8 {
  // CHECK-NOT: arm_sme.cast_tile_to_vector
  // CHECK-NOT: arm_sme.cast_vector_to_tile
  // CHECK-NEXT: return %[[TILE_ID]] : i8
  %tile = arm_sme.cast_tile_to_vector %tile_id_0 : i8 to vector<[16]x[16]xi8>
  %tile_id_1 = arm_sme.cast_vector_to_tile %tile : vector<[16]x[16]xi8> to i8
  return %tile_id_1 : i8
}

// -----

// CHECK-LABEL: @cast_tile_to_vector__cast_vector_to_tile
// CHECK-SAME: %[[TILE:.*]]: vector<[16]x[16]xi8>
func.func @cast_tile_to_vector__cast_vector_to_tile(%tile_0 : vector<[16]x[16]xi8>) -> vector<[16]x[16]xi8> {
  // CHECK-NOT: arm_sme.cast_vector_to_tile
  // CHECK-NOT: arm_sme.cast_tile_to_vector
  // CHECK-NEXT: return %[[TILE]] : vector<[16]x[16]xi8>
  %tile_id = arm_sme.cast_vector_to_tile %tile_0 : vector<[16]x[16]xi8> to i8
  %tile_1 = arm_sme.cast_tile_to_vector %tile_id : i8 to vector<[16]x[16]xi8>
  return %tile_1 : vector<[16]x[16]xi8>
}
