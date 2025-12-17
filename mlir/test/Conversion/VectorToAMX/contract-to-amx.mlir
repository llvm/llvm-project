// RUN: mlir-opt %s -convert-vector-to-amx -split-input-file | FileCheck %s

/// VNNI format is Intel's packed data layout.
/// For matrix multiplication, elements from the reduction dimension `k`
/// are packed into 32-bit tuples. Then the appropriate AMX operations can
/// perform tile multiplication directly on the packed data.
///
/// These packed elements are represented in the indexing maps by a separate
/// reduction dimension `vnni`.

#map = affine_map<(m, n, k, vnni) -> (m, k, vnni)>
#map1 = affine_map<(m, n, k, vnni) -> (k, n, vnni)>
#map2 = affine_map<(m, n, k, vnni) -> (m, n)>
func.func @contract_vnni_f16(%A: vector<4x8x2xf16>, %B: vector<8x16x2xf16>,
    %C: vector<4x16xf32>) -> vector<4x16xf32> {
  %0 = vector.contract
    {kind = #vector.kind<add>,
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    %A, %B, %C : vector<4x8x2xf16>, vector<8x16x2xf16> into vector<4x16xf32>
  return %0 : vector<4x16xf32>
}

// CHECK-LABEL: @contract_vnni_f16(
// CHECK-SAME:    %[[A:.+]]: vector<4x8x2xf16>,
// CHECK-SAME:    %[[B:.+]]: vector<8x16x2xf16>,
// CHECK-SAME:    %[[C:.+]]: vector<4x16xf32>

/// AMX hardware has no direct access to the registers. Thus, data must
/// be transfered through intermediate buffers.
///
/// Load A vector into an AMX tile
// CHECK:       %[[A_BUF:.+]] = memref.alloca() : memref<4x8x2xf16>
// CHECK:       vector.transfer_write %[[A]], %[[A_BUF]]
// CHECK:       %[[A_BUF_2D:.+]] = memref.collapse_shape %[[A_BUF]]
// CHECK-SAME:    {{\[}}[0], [1, 2]] : memref<4x8x2xf16> into memref<4x16xf16>
// CHECK:       %[[A_TILE:.+]] = amx.tile_load %[[A_BUF_2D]]

/// Load B vector into an AMX tile
// CHECK:       %[[B_BUF:.+]] = memref.alloca() : memref<8x16x2xf16>
// CHECK:       vector.transfer_write %[[B]], %[[B_BUF]]
// CHECK:       %[[B_BUF_2D:.+]] = memref.collapse_shape %[[B_BUF]]
// CHECK-SAME:    {{\[}}[0], [1, 2]] : memref<8x16x2xf16> into memref<8x32xf16>
// CHECK:       %[[B_TILE:.+]] = amx.tile_load %[[B_BUF_2D]]

/// Load C vector into an AMX tile
// CHECK:       %[[C_BUF:.+]] = memref.alloca() : memref<4x16xf32>
// CHECK:       vector.transfer_write %[[C]], %[[C_BUF]]
// CHECK:       %[[C_TILE:.+]] = amx.tile_load %[[C_BUF]]

/// Perform tile multiplication
// CHECK:       %[[RES:.+]] = amx.tile_mulf
// CHECK-SAME:    %[[A_TILE]], %[[B_TILE]], %[[C_TILE]]

/// Load the result back into a vector
// CHECK:       %[[RES_BUF:.+]] = memref.alloca() : memref<4x16xf32>
// CHECK:       amx.tile_store %[[RES_BUF]]{{.*}}, %[[RES]]
// CHECK:       %[[RES_VEC:.+]] = vector.transfer_read %[[RES_BUF]]

// CHECK:       return %[[RES_VEC]]

// -----

#map = affine_map<(m, n, k, vnni) -> (m, k, vnni)>
#map1 = affine_map<(m, n, k, vnni) -> (k, n, vnni)>
#map2 = affine_map<(m, n, k, vnni) -> (m, n)>
func.func @contract_vnni_bf16(%A: vector<4x8x2xbf16>, %B: vector<8x16x2xbf16>,
    %C: vector<4x16xf32>) -> vector<4x16xf32> {
  %0 = vector.contract
    {kind = #vector.kind<add>,
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    %A, %B, %C : vector<4x8x2xbf16>, vector<8x16x2xbf16> into vector<4x16xf32>
  return %0 : vector<4x16xf32>
}

// CHECK-LABEL: @contract_vnni_bf16(
// CHECK-COUNT-3: amx.tile_load
// CHECK: amx.tile_mulf
// CHECK: amx.tile_store

// -----

#map = affine_map<(m, n, k, vnni) -> (m, k, vnni)>
#map1 = affine_map<(m, n, k, vnni) -> (k, n, vnni)>
#map2 = affine_map<(m, n, k, vnni) -> (m, n)>
func.func @contract_vnni_i8(%A: vector<4x16x4xi8>, %B: vector<16x8x4xi8>,
    %C: vector<4x8xi32>) -> vector<4x8xi32> {
  %0 = vector.contract
    {kind = #vector.kind<add>,
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    %A, %B, %C : vector<4x16x4xi8>, vector<16x8x4xi8> into vector<4x8xi32>
  return %0 : vector<4x8xi32>
}

// CHECK-LABEL: @contract_vnni_i8(
// CHECK-COUNT-3: amx.tile_load
// CHECK: amx.tile_muli
// CHECK: amx.tile_store

// -----

#map = affine_map<(vnni, m, k, n) -> (m, k, vnni)>
#map1 = affine_map<(vnni, m, k, n) -> (k, n, vnni)>
#map2 = affine_map<(vnni, m, k, n) -> (m, n)>
func.func @contract_shuffled_iterators(%A: vector<4x16x4xi8>, %B: vector<16x8x4xi8>,
    %C: vector<4x8xi32>) -> vector<4x8xi32> {
  %0 = vector.contract
    {kind = #vector.kind<add>,
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "reduction", "parallel"]}
    %A, %B, %C : vector<4x16x4xi8>, vector<16x8x4xi8> into vector<4x8xi32>
  return %0 : vector<4x8xi32>
}

// CHECK-LABEL: @contract_shuffled_iterators(
// CHECK-COUNT-3: amx.tile_load
// CHECK: amx.tile_muli
// CHECK: amx.tile_store

// -----

#map = affine_map<(m, n, k, vnni) -> (m, k, vnni)>
#map1 = affine_map<(m, n, k, vnni) -> (k, n, vnni)>
#map2 = affine_map<(m, n, k, vnni) -> (m, n)>
func.func @negative_invalid_kind(%A: vector<4x8x2xf16>, %B: vector<8x16x2xf16>,
    %C: vector<4x16xf32>) -> vector<4x16xf32> {
  %0 = vector.contract
    {kind = #vector.kind<mul>,
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    %A, %B, %C : vector<4x8x2xf16>, vector<8x16x2xf16> into vector<4x16xf32>
  return %0 : vector<4x16xf32>
}

// CHECK-LABEL: @negative_invalid_kind(
// CHECK-NOT: amx
// CHECK: vector.contract

// -----

#map = affine_map<(m, k, vnni) -> (m, k, vnni)>
#map1 = affine_map<(m, k, vnni) -> (k, m, vnni)>
#map2 = affine_map<(m, k, vnni) -> ()>
func.func @negative_non_vector_acc(%A: vector<4x8x2xf16>, %B: vector<8x4x2xf16>,
    %C: f32) -> f32 {
  %0 = vector.contract
    {kind = #vector.kind<add>,
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "reduction", "reduction"]}
    %A, %B, %C : vector<4x8x2xf16>, vector<8x4x2xf16> into f32
  return %0 : f32
}

// CHECK-LABEL: @negative_non_vector_acc(
// CHECK-NOT: amx
// CHECK: vector.contract

// -----

#map = affine_map<(m, n, k, vnni) -> (m, k, vnni)>
#map1 = affine_map<(m, n, k, vnni) -> (k, n, vnni)>
#map2 = affine_map<(m, n, k, vnni) -> (m, n)>
func.func @negative_invalid_operand_types(%A: vector<4x8x2xf32>,
    %B: vector<8x16x2xf32>, %C: vector<4x16xf32>) -> vector<4x16xf32> {
  %0 = vector.contract
    {kind = #vector.kind<add>,
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    %A, %B, %C : vector<4x8x2xf32>, vector<8x16x2xf32> into vector<4x16xf32>
  return %0 : vector<4x16xf32>
}

// CHECK-LABEL: @negative_invalid_operand_types(
// CHECK-NOT: amx
// CHECK: vector.contract

// -----

#map = affine_map<(m, n, k) -> (m, k)>
#map1 = affine_map<(m, n, k) -> (k, n)>
#map2 = affine_map<(m, n, k) -> (m, n)>
func.func @negative_non_packed_layout(%A: vector<4x16xf16>, %B: vector<16x16xf16>,
    %C: vector<4x16xf32>) -> vector<4x16xf32> {
  %0 = vector.contract
    {kind = #vector.kind<add>,
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]}
    %A, %B, %C : vector<4x16xf16>, vector<16x16xf16> into vector<4x16xf32>
  return %0 : vector<4x16xf32>
}

// CHECK-LABEL: @negative_non_packed_layout(
// CHECK-NOT: amx
// CHECK: vector.contract

// -----

#map = affine_map<(m, n, k, vnni) -> (m, k, vnni)>
#map1 = affine_map<(m, n, k, vnni) -> (k, n, vnni)>
#map2 = affine_map<(m, n, k, vnni) -> (m, n)>
func.func @negative_invalid_vnni_factor(%A: vector<4x2x4xf16>, %B: vector<2x2x4xf16>,
    %C: vector<4x2xf32>) -> vector<4x2xf32> {
  %0 = vector.contract
    {kind = #vector.kind<add>,
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    %A, %B, %C : vector<4x2x4xf16>, vector<2x2x4xf16> into vector<4x2xf32>
  return %0 : vector<4x2xf32>
}

// CHECK-LABEL: @negative_invalid_vnni_factor(
// CHECK-NOT: amx
// CHECK: vector.contract

// -----

#map = affine_map<(batch, m, n, k, vnni) -> (batch, m, k, vnni)>
#map1 = affine_map<(batch, m, n, k, vnni) -> (batch, k, n, vnni)>
#map2 = affine_map<(batch, m, n, k, vnni) -> (batch, m, n)>
func.func @negative_invalid_operands_shapes(%A: vector<1x4x8x2xf16>,
    %B: vector<1x8x16x2xf16>, %C: vector<1x4x16xf32>) -> vector<1x4x16xf32> {
  %0 = vector.contract
    {kind = #vector.kind<add>,
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]}
    %A, %B, %C : vector<1x4x8x2xf16>, vector<1x8x16x2xf16> into vector<1x4x16xf32>
  return %0 : vector<1x4x16xf32>
}

// CHECK-LABEL: @negative_invalid_operands_shapes(
// CHECK-NOT: amx
// CHECK: vector.contract

// -----

#map = affine_map<(m, n, k, vnni) -> (m, k, vnni)>
#map1 = affine_map<(m, n, k, vnni) -> (k, n, vnni)>
#map2 = affine_map<(m, n, k, vnni) -> (m, n)>
func.func @negative_too_many_rows(%A: vector<32x8x2xf16>, %B: vector<8x16x2xf16>,
    %C: vector<32x16xf32>) -> vector<32x16xf32> {
  %0 = vector.contract
    {kind = #vector.kind<add>,
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    %A, %B, %C : vector<32x8x2xf16>, vector<8x16x2xf16> into vector<32x16xf32>
  return %0 : vector<32x16xf32>
}

// CHECK-LABEL: @negative_too_many_rows(
// CHECK-NOT: amx
// CHECK: vector.contract

// -----

#map = affine_map<(m, n, k, vnni) -> (m, k, vnni)>
#map1 = affine_map<(m, n, k, vnni) -> (k, n, vnni)>
#map2 = affine_map<(m, n, k, vnni) -> (m, n)>
func.func @negative_too_wide_rows(%A: vector<4x32x2xf16>, %B: vector<32x16x2xf16>,
    %C: vector<4x16xf32>) -> vector<4x16xf32> {
  %0 = vector.contract
    {kind = #vector.kind<add>,
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    %A, %B, %C : vector<4x32x2xf16>, vector<32x16x2xf16> into vector<4x16xf32>
  return %0 : vector<4x16xf32>
}

// CHECK-LABEL: @negative_too_wide_rows(
// CHECK-NOT: amx
// CHECK: vector.contract

// -----

#map = affine_map<(m, n, k, vnni) -> (k, vnni, m)>
#map1 = affine_map<(m, n, k, vnni) -> (n, k, vnni)>
#map2 = affine_map<(m, n, k, vnni) -> (m, n)>
func.func @negative_input_dim_permutation(%A: vector<2x2x2xf16>,
    %B: vector<2x2x2xf16>, %C: vector<2x2xf32>) -> vector<2x2xf32> {
  %0 = vector.contract
    {kind = #vector.kind<add>,
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    %A, %B, %C : vector<2x2x2xf16>, vector<2x2x2xf16> into vector<2x2xf32>
  return %0 : vector<2x2xf32>
}

// CHECK-LABEL: @negative_input_dim_permutation(
// CHECK-NOT: amx
// CHECK: vector.contract

// -----

#map = affine_map<(m, n, k, vnni) -> (m, k, vnni)>
#map1 = affine_map<(m, n, k, vnni) -> (k, n, vnni)>
#map2 = affine_map<(m, n, k, vnni) -> (n, m)>
func.func @negative_output_dim_permutation(%A: vector<4x8x2xf16>,
    %B: vector<8x16x2xf16>, %C: vector<16x4xf32>) -> vector<16x4xf32> {
  %0 = vector.contract
    {kind = #vector.kind<add>,
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    %A, %B, %C : vector<4x8x2xf16>, vector<8x16x2xf16> into vector<16x4xf32>
  return %0 : vector<16x4xf32>
}

// CHECK-LABEL: @negative_output_dim_permutation(
// CHECK-NOT: amx
// CHECK: vector.contract
