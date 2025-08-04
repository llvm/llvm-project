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

#map = affine_map<(d0, d1, d2) -> (d2, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @negative_gemm_transpose_a(%lhs: vector<16x8xf16>, %rhs: vector<16x16xf16>,
    %acc: vector<8x16xf32>) -> vector<8x16xf32> {
  %3 = vector.contract
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>} %lhs, %rhs, %acc
    : vector<16x8xf16>, vector<16x16xf16> into vector<8x16xf32>
  return %3 : vector<8x16xf32>
}

// CHECK-LABEL: @negative_gemm_transpose_a(
// CHECK:       vector.contract

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @negative_gemm_transpose_b(%lhs: vector<8x16xf16>, %rhs: vector<16x16xf16>,
    %acc: vector<8x16xf32>) -> vector<8x16xf32> {
  %3 = vector.contract
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>} %lhs, %rhs, %acc
    : vector<8x16xf16>, vector<16x16xf16> into vector<8x16xf32>
  return %3 : vector<8x16xf32>
}

// CHECK-LABEL: @negative_gemm_transpose_b(
// CHECK:       vector.contract
