// RUN: mlir-opt %s -convert-vector-to-xegpu -split-input-file | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @dpas_gemm_f16(%lhs: vector<8x16xf16>, %rhs: vector<16x16xf16>,
    %acc: vector<8x16xf32>) -> vector<8x16xf32> {
  %3 = vector.contract
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>} %lhs, %rhs, %acc
    : vector<8x16xf16>, vector<16x16xf16> into vector<8x16xf32>
  return %3 : vector<8x16xf32>
}

// CHECK-LABEL: @dpas_gemm_f16(
// CHECK-SAME:  %[[LHS:.+]]: vector<8x16xf16>,
// CHECK-SAME:  %[[RHS:.+]]: vector<16x16xf16>,
// CHECK-SAME:  %[[ACC:.+]]: vector<8x16xf32>
// CHECK:       %[[DPAS:.+]] = xegpu.dpas
// CHECK-SAME:    %[[LHS]], %[[RHS]], %[[ACC]]
// CHECK-SAME:    {{.*}}-> vector<8x16xf32>
// CHECK:       return %[[DPAS]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @dpas_gemm_i8(%lhs: vector<8x32xi8>, %rhs: vector<32x16xi8>,
    %acc: vector<8x16xi32>) -> vector<8x16xi32> {
  %3 = vector.contract
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>} %lhs, %rhs, %acc
    : vector<8x32xi8>, vector<32x16xi8> into vector<8x16xi32>
  return %3 : vector<8x16xi32>
}

// CHECK-LABEL: @dpas_gemm_i8(
// CHECK-SAME:  %[[LHS:.+]]: vector<8x32xi8>,
// CHECK-SAME:  %[[RHS:.+]]: vector<32x16xi8>,
// CHECK-SAME:  %[[ACC:.+]]: vector<8x16xi32>
// CHECK:       %[[DPAS:.+]] = xegpu.dpas
// CHECK-SAME:    %[[LHS]], %[[RHS]], %[[ACC]]
// CHECK-SAME:    {{.*}}-> vector<8x16xi32>
// CHECK:       return %[[DPAS]]

// -----

// No restriction on vector sizes to allow capturing workgroup-sized operations.
// The operations can then be progressively resized through distribution down
// to hardware compatible sizes.

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @dpas_large_dims(%lhs: vector<128x512xf16>, %rhs: vector<512x256xf16>,
    %acc: vector<128x256xf32>) -> vector<128x256xf32> {
  %3 = vector.contract
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>} %lhs, %rhs, %acc
    : vector<128x512xf16>, vector<512x256xf16> into vector<128x256xf32>
  return %3 : vector<128x256xf32>
}

// CHECK-LABEL: @dpas_large_dims(
// CHECK-SAME:  %[[LHS:.+]]: vector<128x512xf16>,
// CHECK-SAME:  %[[RHS:.+]]: vector<512x256xf16>,
// CHECK-SAME:  %[[ACC:.+]]: vector<128x256xf32>
// CHECK:       %[[DPAS:.+]] = xegpu.dpas
// CHECK-SAME:    %[[LHS]], %[[RHS]], %[[ACC]]
// CHECK-SAME:    {{.*}}-> vector<128x256xf32>
// CHECK:       return %[[DPAS]]

// -----

#map = affine_map<(d0, d1, d2) -> (d2, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @gemm_transpose_a(%lhs: vector<16x8xf16>, %rhs: vector<16x16xf16>,
    %acc: vector<8x16xf32>) -> vector<8x16xf32> {
  %3 = vector.contract
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>} %lhs, %rhs, %acc
    : vector<16x8xf16>, vector<16x16xf16> into vector<8x16xf32>
  return %3 : vector<8x16xf32>
}

// CHECK-LABEL: @gemm_transpose_a(
// CHECK-SAME:  %[[LHS:.+]]: vector<16x8xf16>,
// CHECK-SAME:  %[[RHS:.+]]: vector<16x16xf16>,
// CHECK-SAME:  %[[ACC:.+]]: vector<8x16xf32>
// CHECK:       %[[LHS_TRANSPOSED:.+]] = vector.transpose %[[LHS]], [1, 0] : vector<16x8xf16> to vector<8x16xf16>
// CHECK:       %[[DPAS:.+]] = xegpu.dpas
// CHECK-SAME:    %[[LHS_TRANSPOSED]], %[[RHS]], %[[ACC]]
// CHECK-SAME:    {{.*}}-> vector<8x16xf32>
// CHECK:       return %[[DPAS]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @gemm_transpose_b(%lhs: vector<8x16xf16>, %rhs: vector<16x16xf16>,
    %acc: vector<8x16xf32>) -> vector<8x16xf32> {
  %3 = vector.contract
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>} %lhs, %rhs, %acc
    : vector<8x16xf16>, vector<16x16xf16> into vector<8x16xf32>
  return %3 : vector<8x16xf32>
}

// CHECK-LABEL: @gemm_transpose_b(
// CHECK-SAME:  %[[LHS:.+]]: vector<8x16xf16>,
// CHECK-SAME:  %[[RHS:.+]]: vector<16x16xf16>,
// CHECK-SAME:  %[[ACC:.+]]: vector<8x16xf32>
// CHECK:       %[[RHS_TRANSPOSED:.+]] = vector.transpose %[[RHS]], [1, 0] : vector<16x16xf16> to vector<16x16xf16>
// CHECK:       %[[DPAS:.+]] = xegpu.dpas
// CHECK-SAME:    %[[LHS]], %[[RHS_TRANSPOSED]], %[[ACC]]
// CHECK-SAME:    {{.*}}-> vector<8x16xf32>
// CHECK:       return %[[DPAS]]

// -----

// For simplicity, only plain data layouts are currently supported.
// VNNI packing is applied later as a separate lowering step.

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
func.func @negative_vnni_packed(%lhs: vector<8x8x2xf16>, %rhs: vector<8x16x2xf16>,
    %acc: vector<8x16xf32>) -> vector<8x16xf32> {
  %3 = vector.contract
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"],
    kind = #vector.kind<add>} %lhs, %rhs, %acc
    : vector<8x8x2xf16>, vector<8x16x2xf16> into vector<8x16xf32>
  return %3 : vector<8x16xf32>
}

// CHECK-LABEL: @negative_vnni_packed(
// CHECK: vector.contract

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @negative_combining_kind(%lhs: vector<8x16xf16>, %rhs: vector<16x16xf16>,
    %acc: vector<8x16xf32>) -> vector<8x16xf32> {
  %3 = vector.contract
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<mul>} %lhs, %rhs, %acc
    : vector<8x16xf16>, vector<16x16xf16> into vector<8x16xf32>
  return %3 : vector<8x16xf32>
}

// CHECK-LABEL: @negative_combining_kind(
// CHECK:       vector.contract

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> ()>
func.func @negative_accumulator_shape(%lhs: vector<8x16xf16>, %rhs: vector<16x16xf16>,
    %acc: vector<f32>) -> vector<f32> {
  %3 = vector.contract
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "reduction", "reduction"],
    kind = #vector.kind<add>} %lhs, %rhs, %acc
    : vector<8x16xf16>, vector<16x16xf16> into vector<f32>
  return %3 : vector<f32>
}

// CHECK-LABEL: @negative_accumulator_shape(
// CHECK:       vector.contract

// -----

// A batched matmul whose leading dimensions are batch dimensions shared across
// lhs, rhs, and acc lowers to a batched xegpu.dpas. Two batch dims exercise the
// upper bound (rank 4 = 2 batch + 2 core dims).

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
func.func @dpas_batched_gemm(%lhs: vector<2x4x8x16xf16>, %rhs: vector<2x4x16x16xf16>,
    %acc: vector<2x4x8x16xf32>) -> vector<2x4x8x16xf32> {
  %3 = vector.contract
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>} %lhs, %rhs, %acc
    : vector<2x4x8x16xf16>, vector<2x4x16x16xf16> into vector<2x4x8x16xf32>
  return %3 : vector<2x4x8x16xf32>
}

// CHECK-LABEL: @dpas_batched_gemm(
// CHECK-SAME:  %[[LHS:.+]]: vector<2x4x8x16xf16>,
// CHECK-SAME:  %[[RHS:.+]]: vector<2x4x16x16xf16>,
// CHECK-SAME:  %[[ACC:.+]]: vector<2x4x8x16xf32>
// CHECK:       %[[DPAS:.+]] = xegpu.dpas
// CHECK-SAME:    %[[LHS]], %[[RHS]], %[[ACC]]
// CHECK-SAME:    {{.*}}-> vector<2x4x8x16xf32>
// CHECK:       return %[[DPAS]]

// -----

// An N-D contraction whose leading dimension is not a batch dimension shared
// across lhs, rhs, and acc does not map to a (batched) row-major matmul.

#map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
func.func @negative_non_batched_nd(%lhs: vector<128x64x64xf16>, %rhs: vector<64x64xf16>,
    %acc: vector<128x64xf16>) -> vector<128x64xf16> {
  %3 = vector.contract
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>} %lhs, %rhs, %acc
    : vector<128x64x64xf16>, vector<64x64xf16> into vector<128x64xf16>
  return %3 : vector<128x64xf16>
}

// CHECK-LABEL: @negative_non_batched_nd(
// CHECK:       vector.contract

// -----

// A row-major batched matmul with more than 2 batch dims (rank 5) exceeds the
// xegpu.dpas limit of 2 batch + 2 core dims and is not lowered.

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>
func.func @negative_too_many_batch_dims(%lhs: vector<2x2x4x8x16xf16>, %rhs: vector<2x2x4x16x16xf16>,
    %acc: vector<2x2x4x8x16xf32>) -> vector<2x2x4x8x16xf32> {
  %3 = vector.contract
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>} %lhs, %rhs, %acc
    : vector<2x2x4x8x16xf16>, vector<2x2x4x16x16xf16> into vector<2x2x4x8x16xf32>
  return %3 : vector<2x2x4x8x16xf32>
}

// CHECK-LABEL: @negative_too_many_batch_dims(
// CHECK:       vector.contract
