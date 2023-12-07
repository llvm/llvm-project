// RUN: mlir-opt -split-input-file -test-tensor-transform-patterns="test-simplify-pack-patterns" %s | FileCheck %s

// CHECK: func.func @single_dim_packing(
// CHECK-SAME: %[[ARG0:.+]]: tensor<256xf32>)
// CHECK: %[[EXPANDED:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1]] : tensor<256xf32> into tensor<8x32xf32>
// CHECK: return %[[EXPANDED]] : tensor<8x32xf32>
func.func @single_dim_packing(%arg0: tensor<256xf32>) -> tensor<8x32xf32> {
  %empty = tensor.empty() : tensor<8x32xf32>
  %0 = tensor.pack %arg0 inner_dims_pos = [0] inner_tiles = [32] into %empty : tensor<256xf32> -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// -----

// CHECK: func.func @single_dim_packing_with_padding(
// CHECK-SAME: %[[ARG0:.+]]: tensor<255xf32>)
// CHECK-NOT: tensor.expand_shape
// CHECK: tensor.pack
func.func @single_dim_packing_with_padding(%arg0: tensor<255xf32>) -> tensor<8x32xf32> {
  %empty = tensor.empty() : tensor<8x32xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.pack %arg0 padding_value(%cst : f32) inner_dims_pos = [0] inner_tiles = [32] into %empty : tensor<255xf32> -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}
