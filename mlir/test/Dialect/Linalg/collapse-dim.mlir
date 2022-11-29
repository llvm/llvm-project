// RUN: mlir-opt %s -test-linalg-elementwise-fusion-patterns=collapse-dimensions-control=2,3 -split-input-file | FileCheck %s

func.func @collapse_reduction(
    %arg0: tensor<2x32x10x4096xf32>, %arg1: tensor<2x32xf32>) -> tensor<2x32xf32> {
  %0 = linalg.generic {
    indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1)>],
  iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
  ins(%arg0 : tensor<2x32x10x4096xf32>) outs(%arg1 : tensor<2x32xf32>) {
  ^bb0(%arg3: f32, %arg4: f32):
    %1 = arith.addf %arg3, %arg4 : f32
    linalg.yield %1 : f32
  } -> tensor<2x32xf32>
  return %0 : tensor<2x32xf32>
}

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @collapse_reduction
//       CHECK:   %[[T:.*]] = tensor.collapse_shape %{{.*}} {{\[}}[0], [1], [2, 3]] : tensor<2x32x10x4096xf32> into tensor<2x32x40960xf32>
//       CHECK:   linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]],
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "reduction"]}
//  CHECK-SAME:     ins(%[[T]] : tensor<2x32x40960xf32>) outs(%{{.*}} : tensor<2x32xf32>) {
//       CHECK:   } -> tensor<2x32xf32>

// -----

func.func @collapse_parallel(
    %arg0: tensor<32x2x10x4096xf32>, %arg1: tensor<2x32x10x4096xf32>) -> tensor<2x32x10x4096xf32> {
  %0 = linalg.generic {
    indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  ins(%arg0 : tensor<32x2x10x4096xf32>) outs(%arg1 : tensor<2x32x10x4096xf32>) {
  ^bb0(%arg3: f32, %arg4: f32):
    %1 = arith.addf %arg3, %arg4 : f32
    linalg.yield %1 : f32
  } -> tensor<2x32x10x4096xf32>
  return %0 : tensor<2x32x10x4096xf32>
}

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: func @collapse_parallel
//   CHECK-DAG:  %[[S:.*]] = tensor.collapse_shape %{{.*}} {{\[}}[0], [1], [2, 3]] : tensor<32x2x10x4096xf32> into tensor<32x2x40960xf32>
//   CHECK-DAG:  %[[D:.*]] = tensor.collapse_shape %{{.*}} {{\[}}[0], [1], [2, 3]] : tensor<2x32x10x4096xf32> into tensor<2x32x40960xf32>
//       CHECK:  %[[R:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]],
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel"]}
//  CHECK-SAME:     ins(%[[S]] : tensor<32x2x40960xf32>) outs(%[[D]] : tensor<2x32x40960xf32>) {
//       CHECK:   } -> tensor<2x32x40960xf32>
//       CHECK:  tensor.expand_shape %[[R]] {{\[}}[0], [1], [2, 3]] : tensor<2x32x40960xf32> into tensor<2x32x10x4096xf32>
