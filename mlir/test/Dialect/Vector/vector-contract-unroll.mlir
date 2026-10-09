// RUN: mlir-opt %s -transform-preload-library='transform-library-paths=%p/td/unroll-contract.mlir' \
// RUN: -transform-interpreter=entry-point=unroll_contract --split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Test  UnrollContractionPattern
//===----------------------------------------------------------------------===//
// CHECK-LABEL: @matmul
func.func @matmul(%mask: vector<8x8x4xi1>, %lhs: vector<8x4xf32>, %rhs: vector<4x8xf32>, %acc: vector<8x8xf32>) -> vector<8x8xf32> {

// CHECK-REPEAT-8: vector.contract  {{.*}} : vector<1x4xf32>, vector<4x8xf32> into vector<1x8xf32>
  %res = vector.contract {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %lhs, %rhs, %acc : vector<8x4xf32>, vector<4x8xf32> into vector<8x8xf32>

  return %res : vector<8x8xf32>
}

// -----

// CHECK-LABEL: @negative_batched_matmul
func.func @negative_batched_matmul(%mask: vector<1x8x8x4xi1>, %lhs: vector<1x8x4xf32>, %rhs: vector<1x4x8xf32>, %acc: vector<1x8x8xf32>) -> vector<1x8x8xf32> {

// CHECK: vector.contract  {{.*}} : vector<1x8x4xf32>, vector<1x4x8xf32> into vector<1x8x8xf32>
  %res = vector.contract {
    indexing_maps = [affine_map<(b, m, n, k) -> (b, m, k)>, affine_map<(b, m, n, k) -> (b, k, n)>, affine_map<(b, m, n, k) -> (b, m, n)>],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %lhs, %rhs, %acc : vector<1x8x4xf32>, vector<1x4x8xf32> into vector<1x8x8xf32>

  return %res : vector<1x8x8xf32>
}

// -----

// CHECK-LABEL: @negative_matmul_transpose_a
func.func @negative_matmul_transpose_a(%mask: vector<8x8x4xi1>, %lhs: vector<4x8xf32>, %rhs: vector<4x8xf32>, %acc: vector<8x8xf32>) -> vector<8x8xf32> {

// CHECK: vector.contract  {{.*}} : vector<4x8xf32>, vector<4x8xf32> into vector<8x8xf32>
  %res = vector.contract {
    indexing_maps = [affine_map<(k, m, n) -> (k, m)>, affine_map<(k, m, n) -> (k, n)>, affine_map<(k, m, n) -> (m, n)>],
    iterator_types = ["reduction", "parallel", "parallel"],
    kind = #vector.kind<add>}
    %lhs, %rhs, %acc : vector<4x8xf32>, vector<4x8xf32> into vector<8x8xf32>

  return %res : vector<8x8xf32>
}
