// RUN: mlir-opt %s -convert-vector-to-amx -split-input-file | FileCheck %s

/// These test cases validate replacement of vector transfer ops with equivalent
/// AMX tile data transfers.

#map = affine_map<(m, n, k, vnni) -> (m, k, vnni)>
#map1 = affine_map<(m, n, k, vnni) -> (k, n, vnni)>
#map2 = affine_map<(m, n, k, vnni) -> (m, n)>
func.func @transfers_static_dims(%A: memref<64x32x16x2xf16>,
    %B: memref<64x16x32x2xf16>, %C: memref<64x64xf32>, %idx: index) {
  %c0_f16 = arith.constant 0.0 : f16
  %c0_f32 = arith.constant 0.0 : f32
  %vecA = vector.transfer_read %A[%idx, %idx, %idx, %idx], %c0_f16
    {in_bounds = [true, true, true]} : memref<64x32x16x2xf16>, vector<4x8x2xf16>
  %vecB = vector.transfer_read %B[%idx, %idx, %idx, %idx], %c0_f16
    {in_bounds = [true, true, true]} : memref<64x16x32x2xf16>, vector<8x16x2xf16>
  %vecC = vector.transfer_read %C[%idx, %idx], %c0_f32
    {in_bounds = [true, true]} : memref<64x64xf32>, vector<4x16xf32>
  %vecD = vector.contract
    {kind = #vector.kind<add>,
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    %vecA, %vecB, %vecC : vector<4x8x2xf16>, vector<8x16x2xf16> into vector<4x16xf32>
  vector.transfer_write %vecD, %C[%idx, %idx]
    {in_bounds = [true, true]} : vector<4x16xf32>, memref<64x64xf32>
  return
}

// CHECK-LABEL: @transfers_static_dims(
// CHECK-SAME:    %[[A:.+]]: memref<64x32x16x2xf16>,
// CHECK-SAME:    %[[B:.+]]: memref<64x16x32x2xf16>,
// CHECK-SAME:    %[[C:.+]]: memref<64x64xf32>,
// CHECK-SAME:    %[[IDX:.+]]: index
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index

/// Load A into an AMX tile
// CHECK:       %[[A_SUBVIEW:.+]] = memref.subview %[[A]]
// CHECK-SAME:    {{\[}}%[[IDX]], %[[IDX]], %[[IDX]], %[[IDX]]{{\]}}
// CHECK:       %[[A_PACKED_DIM_COLLAPSE:.+]] = memref.collapse_shape %[[A_SUBVIEW]]
// CHECK-SAME:    {{\[}}[0], [1], [2, 3]] : memref<1x4x8x2xf16{{.*}}into memref<1x4x16xf16
// CHECK:       %[[A_TILE:.+]] = amx.tile_load %[[A_PACKED_DIM_COLLAPSE]]
// CHECK-SAME:    {{\[}}%[[C0]], %[[C0]], %[[C0]]{{\]}}
// CHECK-NOT:   vector.transfer_read %[[A]]

/// Load B into an AMX tile
// CHECK:       %[[B_SUBVIEW:.+]] = memref.subview %[[B]]
// CHECK-SAME:    {{\[}}%[[IDX]], %[[IDX]], %[[IDX]], %[[IDX]]{{\]}}
// CHECK:       %[[B_PACKED_DIM_COLLAPSE:.+]] = memref.collapse_shape %[[B_SUBVIEW]]
// CHECK-SAME:    {{\[}}[0], [1], [2, 3]] : memref<1x8x16x2xf16{{.*}}into memref<1x8x32xf16
// CHECK:       %[[B_TILE:.+]] = amx.tile_load %[[B_PACKED_DIM_COLLAPSE]]
// CHECK-SAME:    {{\[}}%[[C0]], %[[C0]], %[[C0]]{{\]}}
// CHECK-NOT:   vector.transfer_read %[[B]]

/// Load C into an AMX tile
// CHECK:       %[[C_SUBVIEW:.+]] = memref.subview %[[C]]
// CHECK-SAME:    {{\[}}%[[IDX]], %[[IDX]]{{\]}}
// CHECK:       %[[C_TILE:.+]] = amx.tile_load %[[C_SUBVIEW]]
// CHECK-SAME:    {{\[}}%[[C0]], %[[C0]]{{\]}}
// CHECK-NOT:   vector.transfer_read %[[C]]

/// Perform tile multiplication
// CHECK:       %[[RES:.+]] = amx.tile_mulf
// CHECK-SAME:    %[[A_TILE]], %[[B_TILE]], %[[C_TILE]]

/// Store the result back
// CHECK:       %[[RES_SUBVIEW:.+]] = memref.subview %[[C]]
// CHECK-SAME:    {{\[}}%[[IDX]], %[[IDX]]{{\]}}
// CHECK:       amx.tile_store %[[RES_SUBVIEW]]{{\[}}%[[C0]], %[[C0]]{{\]}}, %[[RES]]
// CHECK-NOT:   vector.transfer_write{{.*}}%[[C]]

// -----

#map = affine_map<(m, n, k, vnni) -> (m, k, vnni)>
#map1 = affine_map<(m, n, k, vnni) -> (k, n, vnni)>
#map2 = affine_map<(m, n, k, vnni) -> (m, n)>
func.func @transfers_dynamic_outer_dims(%A: memref<?x?x16x2xf16>,
    %B: memref<?x?x32x2xf16>, %C: memref<?x64xf32>, %idx: index) {
  %c0_f16 = arith.constant 0.0 : f16
  %c0_f32 = arith.constant 0.0 : f32
  %vecA = vector.transfer_read %A[%idx, %idx, %idx, %idx], %c0_f16
    {in_bounds = [true, true, true]} : memref<?x?x16x2xf16>, vector<4x8x2xf16>
  %vecB = vector.transfer_read %B[%idx, %idx, %idx, %idx], %c0_f16
    {in_bounds = [true, true, true]} : memref<?x?x32x2xf16>, vector<8x16x2xf16>
  %vecC = vector.transfer_read %C[%idx, %idx], %c0_f32
    {in_bounds = [true, true]} : memref<?x64xf32>, vector<4x16xf32>
  %vecD = vector.contract
    {kind = #vector.kind<add>,
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    %vecA, %vecB, %vecC : vector<4x8x2xf16>, vector<8x16x2xf16> into vector<4x16xf32>
  vector.transfer_write %vecD, %C[%idx, %idx]
    {in_bounds = [true, true]} : vector<4x16xf32>, memref<?x64xf32>
  return
}

// CHECK-LABEL: @transfers_dynamic_outer_dims(
// CHECK-SAME:    %[[A:.+]]: memref<?x?x16x2xf16>,
// CHECK-SAME:    %[[B:.+]]: memref<?x?x32x2xf16>,
// CHECK-SAME:    %[[C:.+]]: memref<?x64xf32>
// CHECK-NOT:  vector.transfer_read %[[A]]
// CHECK-NOT:  vector.transfer_read %[[B]]
// CHECK-NOT:  vector.transfer_read %[[C]]
// CHECK-NOT:  vector.transfer_write{{.*}}%[[C]]

// -----

/// AMX tile can be loaded directly from the buffer. However, vector transfer
/// has to remain due to other users that require data in registers.

#map = affine_map<(m, n, k, vnni) -> (m, k, vnni)>
#map1 = affine_map<(m, n, k, vnni) -> (k, n, vnni)>
#map2 = affine_map<(m, n, k, vnni) -> (m, n)>
func.func @transfer_read_multiple_users(%C: memref<64x64xf32>,
    %vecA: vector<4x8x2xf16>, %vecB: vector<8x16x2xf16>,
    %idx: index) -> vector<4x16xf32> {
  %c0_f32 = arith.constant 0.0 : f32
  %vecC = vector.transfer_read %C[%idx, %idx], %c0_f32
    {in_bounds = [true, true]} : memref<64x64xf32>, vector<4x16xf32>
  %vecD = vector.contract
    {kind = #vector.kind<add>,
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    %vecA, %vecB, %vecC : vector<4x8x2xf16>, vector<8x16x2xf16> into vector<4x16xf32>
  %mul = arith.mulf %vecC, %vecD : vector<4x16xf32>
  return %mul : vector<4x16xf32>
}

// CHECK-LABEL: @transfer_read_multiple_users(
// CHECK-SAME:    %[[C:.+]]: memref<64x64xf32>,

/// Load to AMX tile directly from buffer.
// CHECK: %[[C_SUBVIEW:.+]] = memref.subview %[[C]]
// CHECK: %[[C_TILE:.+]] = amx.tile_load %[[C_SUBVIEW]]

/// Vector read remains to load data for the other non-AMX consumer.
// CHECK: %[[C_VEC:.+]] = vector.transfer_read %[[C]]

/// Contraction uses the directly loaded tile.
// CHECK: %[[TILE_MUL:.+]] = amx.tile_mulf{{.*}}%[[C_TILE]]

/// Consumer uses original C value and the updated one after contraction.
// CHECK: %[[RES_BUF:.+]] = memref.alloca
// CHECK: amx.tile_store %[[RES_BUF]]
// CHECK: %[[RES_VEC:.+]] = vector.transfer_read %[[RES_BUF]]
// CHECK: %[[VEC_MUL:.+]] = arith.mulf %[[C_VEC]], %[[RES_VEC]]

// -----

/// As contraction has multiple users, the results have to loaded back
/// from AMX tile into registers.

#map = affine_map<(m, n, k, vnni) -> (m, k, vnni)>
#map1 = affine_map<(m, n, k, vnni) -> (k, n, vnni)>
#map2 = affine_map<(m, n, k, vnni) -> (m, n)>
func.func @negative_contract_multiple_users(%C: memref<64x64xf32>,
    %vecA: vector<4x8x2xf16>, %vecB: vector<8x16x2xf16>,
    %vecC: vector<4x16xf32>, %idx: index) -> vector<4x16xf32> {
  %vecD = vector.contract
    {kind = #vector.kind<add>,
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    %vecA, %vecB, %vecC : vector<4x8x2xf16>, vector<8x16x2xf16> into vector<4x16xf32>
  vector.transfer_write %vecD, %C[%idx, %idx]
    {in_bounds = [true, true]} : vector<4x16xf32>, memref<64x64xf32>
  %mul = arith.mulf %vecC, %vecD : vector<4x16xf32>
  return %mul : vector<4x16xf32>
}

// CHECK-LABEL: @negative_contract_multiple_users(
// CHECK-SAME:    %[[C:.+]]: memref<64x64xf32>
// CHECK:     %[[TILE_MUL:.+]] = amx.tile_mulf
// CHECK: vector.transfer_write{{.*}}%[[C]]

// -----

#map = affine_map<(m, n, k, vnni) -> (m, k, vnni)>
#map1 = affine_map<(m, n, k, vnni) -> (k, n, vnni)>
#map2 = affine_map<(m, n, k, vnni) -> (m, n)>
func.func @negative_out_of_bounds(%C: memref<64x64xf32>,
    %vecA: vector<4x8x2xf16>, %vecB: vector<8x16x2xf16>,
    %vecC: vector<4x16xf32>, %idx: index) {
  %vecD = vector.contract
    {kind = #vector.kind<add>,
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    %vecA, %vecB, %vecC : vector<4x8x2xf16>, vector<8x16x2xf16> into vector<4x16xf32>
  vector.transfer_write %vecD, %C[%idx, %idx]
    {in_bounds = [true, false]} : vector<4x16xf32>, memref<64x64xf32>
  return
}

// CHECK-LABEL: @negative_out_of_bounds(
// CHECK-SAME:    %[[C:.+]]: memref<64x64xf32>
// CHECK: vector.transfer_write{{.*}}%[[C]]

// -----

#map = affine_map<(m, n, k, vnni) -> (m, k, vnni)>
#map1 = affine_map<(m, n, k, vnni) -> (k, n, vnni)>
#map2 = affine_map<(m, n, k, vnni) -> (m, n)>
func.func @negative_non_identity_map(%C: memref<64x64xf32>,
    %vecA: vector<4x8x2xf16>, %vecB: vector<8x16x2xf16>,
    %vecC: vector<4x16xf32>, %idx: index) {
  %vecD = vector.contract
    {kind = #vector.kind<add>,
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    %vecA, %vecB, %vecC : vector<4x8x2xf16>, vector<8x16x2xf16> into vector<4x16xf32>
  vector.transfer_write %vecD, %C[%idx, %idx]
    {permutation_map = affine_map<(d0, d1) -> (d1, d0)>,
    in_bounds = [true, true]} : vector<4x16xf32>, memref<64x64xf32>
  return
}

// CHECK-LABEL: @negative_non_identity_map(
// CHECK-SAME:    %[[C:.+]]: memref<64x64xf32>
// CHECK: vector.transfer_write{{.*}}%[[C]]

// -----

/// AMX tile transfers require row elements to be contiguous

#map = affine_map<(m, n, k, vnni) -> (m, k, vnni)>
#map1 = affine_map<(m, n, k, vnni) -> (k, n, vnni)>
#map2 = affine_map<(m, n, k, vnni) -> (m, n)>
func.func @negative_non_contiguous_row(
    %A: memref<8x128x2xf16, strided<[256, 4, 1]>>,
    %vecB: vector<8x16x2xf16>, %vecC: vector<4x16xf32>,
    %idx: index) -> vector<4x16xf32> {
  %c0_f16 = arith.constant 0.0 : f16
  %vecA = vector.transfer_read %A[%idx, %idx, %idx], %c0_f16
    {in_bounds = [true, true, true]}
    : memref<8x128x2xf16, strided<[256, 4, 1]>>, vector<4x8x2xf16>
  %vecD = vector.contract
    {kind = #vector.kind<add>,
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    %vecA, %vecB, %vecC : vector<4x8x2xf16>, vector<8x16x2xf16> into vector<4x16xf32>
  return %vecD : vector<4x16xf32>
}

// CHECK-LABEL: @negative_non_contiguous_row(
// CHECK-SAME:    %[[A:.+]]: memref<8x128x2xf16, strided<[256, 4, 1]>>
// CHECK: vector.transfer_read %[[A]]

// -----

/// Buffer shape checks are conservative to avoid problems with deriving
/// stride for AMX tile rows.
/// When in doubt, vector operations are left to perform initial transfers.
/// Afterwards, data can be placed in a contiguous temporary buffer which
/// ensures correct layout for AMX transfers.

#map = affine_map<(m, n, k, vnni) -> (m, k, vnni)>
#map1 = affine_map<(m, n, k, vnni) -> (k, n, vnni)>
#map2 = affine_map<(m, n, k, vnni) -> (m, n)>
func.func @negative_1D_buffer(%C: memref<512xf32>,
    %vecA: vector<4x8x2xf16>, %vecB: vector<8x16x2xf16>,
    %idx: index) -> vector<4x16xf32> {
  %c0_f32 = arith.constant 0.0 : f32
  %vecC = vector.transfer_read %C[%idx], %c0_f32
    {permutation_map = affine_map<(d0) -> (0, d0)>,
    in_bounds = [true, true]} : memref<512xf32>, vector<4x16xf32>
  %vecD = vector.contract
    {kind = #vector.kind<add>,
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    %vecA, %vecB, %vecC : vector<4x8x2xf16>, vector<8x16x2xf16> into vector<4x16xf32>
  return %vecD : vector<4x16xf32>
}

// CHECK-LABEL: @negative_1D_buffer(
// CHECK-SAME:    %[[C:.+]]: memref<512xf32>
// CHECK: vector.transfer_read %[[C]]

// -----

#map = affine_map<(m, n, k, vnni) -> (m, k, vnni)>
#map1 = affine_map<(m, n, k, vnni) -> (k, n, vnni)>
#map2 = affine_map<(m, n, k, vnni) -> (m, n)>
func.func @negative_dynamic_shapes(%A: memref<?x?x?x2xf16>,
    %B: memref<?x?x2xf16>, %C: memref<?x?xf32>, %idx: index) {
  %c0_f16 = arith.constant 0.0 : f16
  %c0_f32 = arith.constant 0.0 : f32
  %vecA = vector.transfer_read %A[%idx, %idx, %idx, %idx], %c0_f16
    {in_bounds = [true, true, true]} : memref<?x?x?x2xf16>, vector<4x8x2xf16>
  %vecB = vector.transfer_read %B[%idx, %idx, %idx], %c0_f16
    {in_bounds = [true, true, true]} : memref<?x?x2xf16>, vector<8x16x2xf16>
  %vecC = vector.transfer_read %C[%idx, %idx], %c0_f32
    {in_bounds = [true, true]} : memref<?x?xf32>, vector<4x16xf32>
  %vecD = vector.contract
    {kind = #vector.kind<add>,
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    %vecA, %vecB, %vecC : vector<4x8x2xf16>, vector<8x16x2xf16> into vector<4x16xf32>
  vector.transfer_write %vecD, %C[%idx, %idx]
    {in_bounds = [true, true]} : vector<4x16xf32>, memref<?x?xf32>
  return
}

// CHECK-LABEL: @negative_dynamic_shapes(
// CHECK-SAME:    %[[A:.+]]: memref<?x?x?x2xf16>,
// CHECK-SAME:    %[[B:.+]]: memref<?x?x2xf16>,
// CHECK-SAME:    %[[C:.+]]: memref<?x?xf32>
// CHECK:  vector.transfer_read %[[A]]
// CHECK:  vector.transfer_read %[[B]]
// CHECK:  vector.transfer_read %[[C]]
// CHECK:  vector.transfer_write{{.*}}%[[C]]

// -----


#map = affine_map<(m, n, k, vnni) -> (m, k, vnni)>
#map1 = affine_map<(m, n, k, vnni) -> (k, n, vnni)>
#map2 = affine_map<(m, n, k, vnni) -> (m, n)>
func.func @negative_invalid_buffer_row_shape(%C: memref<5x2x4x4xf32>,
    %vecA: vector<4x8x2xf16>, %vecB: vector<8x16x2xf16>,
    %idx: index) -> vector<4x16xf32> {
  %c0_f32 = arith.constant 0.0 : f32
  %vecC = vector.transfer_read %C[%idx, %idx, %idx, %idx], %c0_f32
    {in_bounds = [true, true]} : memref<5x2x4x4xf32>, vector<4x16xf32>
  %vecD = vector.contract
    {kind = #vector.kind<add>,
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    %vecA, %vecB, %vecC : vector<4x8x2xf16>, vector<8x16x2xf16> into vector<4x16xf32>
  return %vecD : vector<4x16xf32>
}

// CHECK-LABEL: @negative_invalid_buffer_row_shape(
// CHECK-SAME:    %[[C:.+]]: memref<5x2x4x4xf32>
// CHECK: vector.transfer_read %[[C]]

// -----

#map = affine_map<(m, n, k, vnni) -> (m, k, vnni)>
#map1 = affine_map<(m, n, k, vnni) -> (k, n, vnni)>
#map2 = affine_map<(m, n, k, vnni) -> (m, n)>
func.func @negative_buffer_non_packed_source_shape(%A: memref<8x64x64xf16>,
    %vecB: vector<8x16x2xf16>, %vecC: vector<4x16xf32>,
    %idx: index) -> vector<4x16xf32> {
  %c0_f16 = arith.constant 0.0 : f16
  %vecA = vector.transfer_read %A[%idx, %idx, %idx], %c0_f16
    {in_bounds = [true, true, true]} : memref<8x64x64xf16>, vector<4x8x2xf16>
  %vecD = vector.contract
    {kind = #vector.kind<add>,
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    %vecA, %vecB, %vecC : vector<4x8x2xf16>, vector<8x16x2xf16> into vector<4x16xf32>
  return %vecD : vector<4x16xf32>
}

// CHECK-LABEL: @negative_buffer_non_packed_source_shape(
// CHECK-SAME:    %[[A:.+]]: memref<8x64x64xf16>
// CHECK: vector.transfer_read %[[A]]
