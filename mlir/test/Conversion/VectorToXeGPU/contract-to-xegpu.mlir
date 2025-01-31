// RUN: mlir-opt %s -convert-vector-to-xegpu -split-input-file | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @dpas_gemm_f32(%lhs: vector<8x8xf32>, %rhs: vector<8x16xf32>,
    %acc: vector<8x16xf32>) -> vector<8x16xf32> {
  %3 = vector.contract
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>} %lhs, %rhs, %acc
    : vector<8x8xf32>, vector<8x16xf32> into vector<8x16xf32>
  return %3 : vector<8x16xf32>
}

// CHECK-LABEL: @dpas_gemm_f32(
// CHECK-SAME:  %[[LHS:.+]]: vector<8x8xf32>,
// CHECK-SAME:  %[[RHS:.+]]: vector<8x16xf32>,
// CHECK-SAME:  %[[ACC:.+]]: vector<8x16xf32>
// CHECK:       %[[DPAS:.+]] = xegpu.dpas
// CHECK-SAME:    %[[LHS]], %[[RHS]], %[[ACC]]
// CHECK-SAME:    {{.*}}-> vector<8x16xf32>
// CHECK:       return %[[DPAS]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @dpas_gemm_f16(%lhs: vector<8x16xf16>, %rhs: vector<16x16xf16>,
    %acc: vector<8x16xf16>) -> vector<8x16xf16> {
  %3 = vector.contract
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>} %lhs, %rhs, %acc
    : vector<8x16xf16>, vector<16x16xf16> into vector<8x16xf16>
  return %3 : vector<8x16xf16>
}

// CHECK-LABEL: @dpas_gemm_f16(
// CHECK-SAME:  %[[LHS:.+]]: vector<8x16xf16>,
// CHECK-SAME:  %[[RHS:.+]]: vector<16x16xf16>,
// CHECK-SAME:  %[[ACC:.+]]: vector<8x16xf16>
// CHECK:       %[[DPAS:.+]] = xegpu.dpas
// CHECK-SAME:    %[[LHS]], %[[RHS]], %[[ACC]]
// CHECK-SAME:    {{.*}}-> vector<8x16xf16>
// CHECK:       return %[[DPAS]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
func.func @dpas_gemm_f16_vnni(%lhs: vector<8x8x2xf16>, %rhs: vector<8x16x2xf16>,
    %acc: vector<8x16xf16>) -> vector<8x16xf16> {
  %3 = vector.contract
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"],
    kind = #vector.kind<add>} %lhs, %rhs, %acc
    : vector<8x8x2xf16>, vector<8x16x2xf16> into vector<8x16xf16>
  return %3 : vector<8x16xf16>
}

// CHECK-LABEL: @dpas_gemm_f16_vnni(
// CHECK-SAME:  %[[LHS:.+]]: vector<8x8x2xf16>,
// CHECK-SAME:  %[[RHS:.+]]: vector<8x16x2xf16>,
// CHECK-SAME:  %[[ACC:.+]]: vector<8x16xf16>
// CHECK:       %[[CAST_LHS:.+]] = vector.shape_cast %[[LHS]]
// CHECK-SAME:    vector<8x8x2xf16> to vector<8x16xf16>
// CHECK:       %[[DPAS:.+]] = xegpu.dpas
// CHECK-SAME:    %[[CAST_LHS]], %[[RHS]], %[[ACC]]
// CHECK-SAME:    {{.*}}-> vector<8x16xf16>
// CHECK:       return %[[DPAS]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @dpas_gemm_mixed_types(%lhs: vector<8x16xi16>, %rhs: vector<16x16xi16>,
    %acc: vector<8x16xf16>) -> vector<8x16xf16> {
  %3 = vector.contract
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>} %lhs, %rhs, %acc
    : vector<8x16xi16>, vector<16x16xi16> into vector<8x16xf16>
  return %3 : vector<8x16xf16>
}

// CHECK-LABEL: @dpas_gemm_mixed_types(
// CHECK-SAME:  %[[LHS:.+]]: vector<8x16xi16>,
// CHECK-SAME:  %[[RHS:.+]]: vector<16x16xi16>,
// CHECK-SAME:  %[[ACC:.+]]: vector<8x16xf16>
// CHECK:       %[[DPAS:.+]] = xegpu.dpas
// CHECK-SAME:    %[[LHS]], %[[RHS]], %[[ACC]]
// CHECK-SAME:    {{.*}}-> vector<8x16xf16>
// CHECK:       return %[[DPAS]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @invalid_combining_type(%lhs: vector<8x16xf16>, %rhs: vector<16x16xf16>,
    %acc: vector<8x16xf16>) -> vector<8x16xf16> {
  %3 = vector.contract
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<mul>} %lhs, %rhs, %acc
    : vector<8x16xf16>, vector<16x16xf16> into vector<8x16xf16>
  return %3 : vector<8x16xf16>
}

// CHECK-LABEL: @invalid_combining_type(
// CHECK:       vector.contract

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> ()>
func.func @invalid_accumulator_shape(%lhs: vector<8x16xf16>, %rhs: vector<16x16xf16>,
    %acc: vector<f16>) -> vector<f16> {
  %3 = vector.contract
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "reduction", "reduction"],
    kind = #vector.kind<add>} %lhs, %rhs, %acc
    : vector<8x16xf16>, vector<16x16xf16> into vector<f16>
  return %3 : vector<f16>
}

// CHECK-LABEL: @invalid_accumulator_shape(
// CHECK:       vector.contract

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2)>
func.func @invalid_high_dim_reduction(%lhs: vector<3x8x8x2xf16>, %rhs: vector<3x8x16x2xf16>,
    %acc: vector<8x16xf16>) -> vector<8x16xf16> {
  %3 = vector.contract
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction", "reduction"],
    kind = #vector.kind<add>} %lhs, %rhs, %acc
    : vector<3x8x8x2xf16>, vector<3x8x16x2xf16> into vector<8x16xf16>
  return %3 : vector<8x16xf16>
}

// CHECK-LABEL: @invalid_high_dim_reduction(
// CHECK:       vector.contract

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
func.func @invalid_indexing_maps(%lhs: vector<3x8x16xf16>, %rhs: vector<3x16x16xf16>,
    %acc: vector<8x16xf16>) -> vector<8x16xf16> {
  %3 = vector.contract
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>} %lhs, %rhs, %acc
    : vector<3x8x16xf16>, vector<3x16x16xf16> into vector<8x16xf16>
  return %3 : vector<8x16xf16>
}

// CHECK-LABEL: @invalid_indexing_maps(
// CHECK:       vector.contract

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
func.func @not_vnni_layout(%lhs: vector<8x8x2xf16>, %rhs: vector<16x8x2xf16>,
    %acc: vector<8x16xf16>) -> vector<8x16xf16> {
  %3 = vector.contract
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"],
    kind = #vector.kind<add>} %lhs, %rhs, %acc
    : vector<8x8x2xf16>, vector<16x8x2xf16> into vector<8x16xf16>
  return %3 : vector<8x16xf16>
}

// CHECK-LABEL: @not_vnni_layout(
// CHECK:       vector.contract

// -----

#map = affine_map<(d0, d1, d2) -> (d2, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @invalid_gemm_transpose_a(%lhs: vector<8x8xf32>, %rhs: vector<8x16xf32>,
    %acc: vector<8x16xf32>) -> vector<8x16xf32> {
  %3 = vector.contract
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>} %lhs, %rhs, %acc
    : vector<8x8xf32>, vector<8x16xf32> into vector<8x16xf32>
  return %3 : vector<8x16xf32>
}

// CHECK-LABEL: @invalid_gemm_transpose_a(
// CHECK:       vector.contract

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @invalid_gemm_transpose_b(%lhs: vector<8x8xf32>, %rhs: vector<16x8xf32>,
    %acc: vector<8x16xf32>) -> vector<8x16xf32> {
  %3 = vector.contract
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>} %lhs, %rhs, %acc
    : vector<8x8xf32>, vector<16x8xf32> into vector<8x16xf32>
  return %3 : vector<8x16xf32>
}

// CHECK-LABEL: @invalid_gemm_transpose_b(
// CHECK:       vector.contract

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
func.func @invalid_k_dim_size(%lhs: vector<8x4x2xf16>, %rhs: vector<4x16x2xf16>,
    %acc: vector<8x16xf16>) -> vector<8x16xf16> {
  %3 = vector.contract
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"],
    kind = #vector.kind<add>} %lhs, %rhs, %acc
    : vector<8x4x2xf16>, vector<4x16x2xf16> into vector<8x16xf16>
  return %3 : vector<8x16xf16>
}

// CHECK-LABEL: @invalid_k_dim_size(
// CHECK:       vector.contract

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
func.func @invalid_vnni_factor(%lhs: vector<8x4x4xf16>, %rhs: vector<4x16x4xf16>,
    %acc: vector<8x16xf16>) -> vector<8x16xf16> {
  %3 = vector.contract
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"],
    kind = #vector.kind<add>} %lhs, %rhs, %acc
    : vector<8x4x4xf16>, vector<4x16x4xf16> into vector<8x16xf16>
  return %3 : vector<8x16xf16>
}

// CHECK-LABEL: @invalid_vnni_factor(
// CHECK:       vector.contract
