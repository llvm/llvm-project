// RUN: mlir-opt -split-input-file -test-tensor-transform-patterns="test-simplify-pack-unpack-patterns" %s | FileCheck %s

// CHECK-LABEL: func.func @single_dim_packing(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<256xf32>)
// CHECK:         %[[EXPANDED:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1]] : tensor<256xf32> into tensor<8x32xf32>
// CHECK:         return %[[EXPANDED]] : tensor<8x32xf32>
func.func @single_dim_packing(%arg0: tensor<256xf32>) -> tensor<8x32xf32> {
  %empty = tensor.empty() : tensor<8x32xf32>
  %0 = tensor.pack %arg0 inner_dims_pos = [0] inner_tiles = [32] into %empty : tensor<256xf32> -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// -----

// CHECK-LABEL: func.func @single_dim_packing_with_padding(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<255xf32>)
// CHECK-NOT:     tensor.expand_shape
// CHECK:         tensor.pack
func.func @single_dim_packing_with_padding(%arg0: tensor<255xf32>) -> tensor<8x32xf32> {
  %empty = tensor.empty() : tensor<8x32xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.pack %arg0 padding_value(%cst : f32) inner_dims_pos = [0] inner_tiles = [32] into %empty : tensor<255xf32> -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// -----

// CHECK-LABEL: func.func @single_last_inner_dim_packing(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<5x256xf32>)
// CHECK:         %[[EXPANDED:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0], [1, 2]] : tensor<5x256xf32> into tensor<5x8x32xf32>
// CHECK:         return %[[EXPANDED]] : tensor<5x8x32xf32>
func.func @single_last_inner_dim_packing(%arg0: tensor<5x256xf32>) -> tensor<5x8x32xf32> {
  %empty = tensor.empty() : tensor<5x8x32xf32>
  %0 = tensor.pack %arg0 inner_dims_pos = [1] inner_tiles = [32] into %empty : tensor<5x256xf32> -> tensor<5x8x32xf32>
  return %0 : tensor<5x8x32xf32>
}

// -----

// CHECK-LABEL: func.func @packing_with_outer_dims_perm(
// CHECK-NOT:     tensor.expand_shape
// CHECK:         tensor.pack
func.func @packing_with_outer_dims_perm(%arg0: tensor<5x256xf32>) -> tensor<8x5x32xf32> {
  %empty = tensor.empty() : tensor<8x5x32xf32>
  %0 = tensor.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [1] inner_tiles = [32] into %empty : tensor<5x256xf32> -> tensor<8x5x32xf32>
  return %0 : tensor<8x5x32xf32>
}

// -----

// CHECK-LABEL: func.func @single_first_inner_dim_packing(
// CHECK-NOT:     tensor.expand_shape
// CHECK:         tensor.pack
func.func @single_first_inner_dim_packing(%arg0: tensor<256x5xf32>) -> tensor<8x5x32xf32> {
  %empty = tensor.empty() : tensor<8x5x32xf32>
  %0 = tensor.pack %arg0 inner_dims_pos = [0] inner_tiles = [32] into %empty : tensor<256x5xf32> -> tensor<8x5x32xf32>
  return %0 : tensor<8x5x32xf32>
}
