// RUN: mlir-opt %s -test-vector-unrolling-patterns=unroll-based-on-type | FileCheck %s

func @vector_contract_f32(%lhs : vector<8x8xf32>, %rhs : vector<8x8xf32>,
                          %init : vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = vector.contract
         {indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                           affine_map<(i, j, k) -> (j, k)>,
                           affine_map<(i, j, k) -> (i, j)>],
          iterator_types = ["parallel", "parallel", "reduction"]}
       %lhs, %rhs, %init : vector<8x8xf32>, vector<8x8xf32> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}
// CHECK-LABEL: func @vector_contract_f32
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   return

func @vector_contract_f16(%lhs : vector<8x8xf16>, %rhs : vector<8x8xf16>,
                          %init : vector<8x8xf16>) -> vector<8x8xf16> {
  %0 = vector.contract
         {indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                           affine_map<(i, j, k) -> (j, k)>,
                           affine_map<(i, j, k) -> (i, j)>],
          iterator_types = ["parallel", "parallel", "reduction"]}
       %lhs, %rhs, %init : vector<8x8xf16>, vector<8x8xf16> into vector<8x8xf16>
  return %0 : vector<8x8xf16>
}
// CHECK-LABEL: func @vector_contract_f16
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x4xf16>, vector<4x4xf16> into vector<4x4xf16>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x4xf16>, vector<4x4xf16> into vector<4x4xf16>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x4xf16>, vector<4x4xf16> into vector<4x4xf16>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x4xf16>, vector<4x4xf16> into vector<4x4xf16>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x4xf16>, vector<4x4xf16> into vector<4x4xf16>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x4xf16>, vector<4x4xf16> into vector<4x4xf16>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x4xf16>, vector<4x4xf16> into vector<4x4xf16>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x4xf16>, vector<4x4xf16> into vector<4x4xf16>
//       CHECK:   return
