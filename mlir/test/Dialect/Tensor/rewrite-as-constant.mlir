// RUN: mlir-opt -split-input-file -transform-interpreter %s | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.tensor.rewrite_as_constant
    } : !transform.op<"func.func">
    transform.yield
  }
}

// CHECK-LABEL: func @tensor_generate_constant(
//       CHECK:   %[[cst:.*]] = arith.constant dense<5.000000e+00> : tensor<2x3x5xf32>
//       CHECK:   return %[[cst]]
func.func @tensor_generate_constant() -> tensor<2x3x5xf32> {
  %cst = arith.constant 5.0 : f32
  %0 = tensor.generate {
    ^bb0(%arg0: index, %arg1: index, %arg2: index):
    tensor.yield %cst : f32
  } : tensor<2x3x5xf32>
  return %0 : tensor<2x3x5xf32>
}

// CHECK-LABEL: func.func @fold_pack_with_splat
// CHECK: %[[CST:.+]] = arith.constant dense<1> : tensor<8x2x1x1x32x32xi64>
// CHECK-NEXT: return %[[CST]] : tensor<8x2x1x1x32x32xi64>
func.func @fold_pack_with_splat() ->  tensor<8x2x1x1x32x32xi64> {
  %cst = arith.constant dense<1> : tensor<1x1x64x256xi64>
  %0 = tensor.empty() : tensor<8x2x1x1x32x32xi64>
  %pack = tensor.pack %cst outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32]
    into %0 : tensor<1x1x64x256xi64> -> tensor<8x2x1x1x32x32xi64>
  return  %pack : tensor<8x2x1x1x32x32xi64>
}

// CHECK-LABEL: func.func @fold_pack_with_non_splat
// CHECK: %[[CST:.+]] = arith.constant
// CHECK-SAME: [0.000000e+00, 1.000000e+00], [8.000000e+00, 9.000000e+00], [1.600000e+01, 1.700000e+01], [2.400000e+01, 2.500000e+01]
// CHECK-SAME: [2.000000e+00, 3.000000e+00], [1.000000e+01, 1.100000e+01], [1.800000e+01, 1.900000e+01], [2.600000e+01, 2.700000e+01]
// CHECK-SAME: [4.000000e+00, 5.000000e+00], [1.200000e+01, 1.300000e+01], [2.000000e+01, 2.100000e+01], [2.800000e+01, 2.900000e+01]
// CHECK-SAME: [6.000000e+00, 7.000000e+00], [1.400000e+01, 1.500000e+01], [2.200000e+01, 2.300000e+01], [3.000000e+01, 3.100000e+01]
// CHECK-SAME: [3.200000e+01, 3.300000e+01], [4.000000e+01, 4.100000e+01], [4.900000e+01, 5.000000e+01], [5.700000e+01, 5.800000e+01]
// CHECK-SAME: [3.400000e+01, 3.500000e+01], [4.200000e+01, 4.300000e+01], [5.100000e+01, 5.200000e+01], [5.900000e+01, 6.000000e+01]
// CHECK-SAME: [3.600000e+01, 3.700000e+01], [4.400000e+01, 4.500000e+01], [5.300000e+01, 5.400000e+01], [6.100000e+01, 6.200000e+01]
// CHECK-SAME: [3.800000e+01, 3.900000e+01], [4.600000e+01, 4.700000e+01], [5.500000e+01, 5.600000e+01], [6.300000e+01, 6.400000e+01]
// CHECK-NOT: tensor.pack
// CHECK: return %[[CST]] : tensor<2x4x4x2xf32>
func.func @fold_pack_with_non_splat() -> tensor<2x4x4x2xf32> {
  %cst = arith.constant dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                               [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                               [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
                               [24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0],
                               [32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0],
                               [40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0],
                               [49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0],
                               [57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0]]> : tensor<8x8xf32>
  %0 = tensor.empty() : tensor<2x4x4x2xf32>
  %pack = tensor.pack %cst inner_dims_pos = [0, 1] inner_tiles = [4, 2]
    into %0 : tensor<8x8xf32> -> tensor<2x4x4x2xf32>
  return %pack : tensor<2x4x4x2xf32>
}

// CHECK-LABEL: func.func @fold_pack_with_non_splat_with_inner_dims_reordered
// CHECK: %[[CST:.+]] = arith.constant
// CHECK-SAME: [0.000000e+00, 8.000000e+00, 1.600000e+01, 2.400000e+01], [1.000000e+00, 9.000000e+00, 1.700000e+01, 2.500000e+01]
// CHECK-SAME: [4.000000e+00, 1.200000e+01, 2.000000e+01, 2.800000e+01], [5.000000e+00, 1.300000e+01, 2.100000e+01, 2.900000e+01]
// CHECK-SAME: [8.000000e+00, 1.600000e+01, 2.400000e+01, 3.200000e+01], [9.000000e+00, 1.700000e+01, 2.500000e+01, 3.300000e+01]
// CHECK-SAME: [1.200000e+01, 2.000000e+01, 2.800000e+01, 3.600000e+01], [1.300000e+01, 2.100000e+01, 2.900000e+01, 3.700000e+01]
// CHECK-SAME: [1.600000e+01, 2.400000e+01, 3.200000e+01, 4.000000e+01], [1.700000e+01, 2.500000e+01, 3.300000e+01, 4.100000e+01]
// CHECK-SAME: [2.000000e+01, 2.800000e+01, 3.600000e+01, 4.400000e+01], [2.100000e+01, 2.900000e+01, 3.700000e+01, 4.500000e+01]
// CHECK-SAME: [2.400000e+01, 3.200000e+01, 4.000000e+01, 4.900000e+01], [2.500000e+01, 3.300000e+01, 4.100000e+01, 5.000000e+01]
// CHECK-SAME: [2.800000e+01, 3.600000e+01, 4.400000e+01, 5.300000e+01], [2.900000e+01, 3.700000e+01, 4.500000e+01, 5.400000e+01]
// CHECK-NOT: tensor.pack
// CHECK: return %[[CST]] : tensor<2x4x2x4xf32>
func.func @fold_pack_with_non_splat_with_inner_dims_reordered() -> tensor<2x4x2x4xf32> {
  %cst = arith.constant dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                               [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                               [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
                               [24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0],
                               [32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0],
                               [40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0],
                               [49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0],
                               [57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0]]> : tensor<8x8xf32>
  %0 = tensor.empty() : tensor<2x4x2x4xf32>
  %pack = tensor.pack %cst inner_dims_pos = [1, 0] inner_tiles = [2, 4]
    into %0 : tensor<8x8xf32> -> tensor<2x4x2x4xf32>
  return %pack : tensor<2x4x2x4xf32>
}

// CHECK-LABEL: func.func @fold_pack_with_non_splat_with_inner_tiles_reordered
// CHECK: %[[CST:.+]] = arith.constant
// CHECK-SAME: [0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01]
// CHECK-SAME: [4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00], [1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01]
// CHECK-SAME: [1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01], [2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01]
// CHECK-SAME: [2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01], [2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01]
// CHECK-SAME: [3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01], [4.000000e+01, 4.100000e+01, 4.200000e+01, 4.300000e+01]
// CHECK-SAME: [3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01], [4.400000e+01, 4.500000e+01, 4.600000e+01, 4.700000e+01]
// CHECK-SAME: [4.900000e+01, 5.000000e+01, 5.100000e+01, 5.200000e+01], [5.700000e+01, 5.800000e+01, 5.900000e+01, 6.000000e+01]
// CHECK-SAME: [5.300000e+01, 5.400000e+01, 5.500000e+01, 5.600000e+01], [6.100000e+01, 6.200000e+01, 6.300000e+01, 6.400000e+01]
// CHECK-NOT: tensor.pack
// CHECK: return %[[CST]] : tensor<4x2x2x4xf32>
func.func @fold_pack_with_non_splat_with_inner_tiles_reordered() -> tensor<4x2x2x4xf32> {
  %cst = arith.constant dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                               [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                               [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
                               [24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0],
                               [32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0],
                               [40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0],
                               [49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0],
                               [57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0]]> : tensor<8x8xf32>
  %0 = tensor.empty() : tensor<4x2x2x4xf32>
  %pack = tensor.pack %cst inner_dims_pos = [0, 1] inner_tiles = [2, 4]
    into %0 : tensor<8x8xf32> -> tensor<4x2x2x4xf32>
  return %pack : tensor<4x2x2x4xf32>
}

// CHECK-LABEL: func.func @fold_pack_with_non_splat_with_outer_permutation
// CHECK: %[[CST:.+]] = arith.constant
// CHECK-SAME: [0.000000e+00, 1.000000e+00], [8.000000e+00, 9.000000e+00], [1.600000e+01, 1.700000e+01], [2.400000e+01, 2.500000e+01]
// CHECK-SAME: [3.200000e+01, 3.300000e+01], [4.000000e+01, 4.100000e+01], [4.900000e+01, 5.000000e+01], [5.700000e+01, 5.800000e+01]
// CHECK-SAME: [2.000000e+00, 3.000000e+00], [1.000000e+01, 1.100000e+01], [1.800000e+01, 1.900000e+01], [2.600000e+01, 2.700000e+01]
// CHECK-SAME: [3.400000e+01, 3.500000e+01], [4.200000e+01, 4.300000e+01], [5.100000e+01, 5.200000e+01], [5.900000e+01, 6.000000e+01]
// CHECK-SAME: [4.000000e+00, 5.000000e+00], [1.200000e+01, 1.300000e+01], [2.000000e+01, 2.100000e+01], [2.800000e+01, 2.900000e+01]
// CHECK-SAME: [3.600000e+01, 3.700000e+01], [4.400000e+01, 4.500000e+01], [5.300000e+01, 5.400000e+01], [6.100000e+01, 6.200000e+01]
// CHECK-SAME: [6.000000e+00, 7.000000e+00], [1.400000e+01, 1.500000e+01], [2.200000e+01, 2.300000e+01], [3.000000e+01, 3.100000e+01]
// CHECK-SAME: [3.800000e+01, 3.900000e+01], [4.600000e+01, 4.700000e+01], [5.500000e+01, 5.600000e+01], [6.300000e+01, 6.400000e+01]
// CHECK-NOT: tensor.pack
// CHECK: return %[[CST]] : tensor<4x2x4x2xf32>
func.func @fold_pack_with_non_splat_with_outer_permutation() -> tensor<4x2x4x2xf32> {
  %cst = arith.constant dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                               [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                               [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
                               [24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0],
                               [32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0],
                               [40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0],
                               [49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0],
                               [57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0]]> : tensor<8x8xf32>
  %0 = tensor.empty() : tensor<4x2x4x2xf32>
  %pack = tensor.pack %cst outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [4, 2]
    into %0 : tensor<8x8xf32> -> tensor<4x2x4x2xf32>
  return %pack : tensor<4x2x4x2xf32>
}

// CHECK-LABEL: func.func @fold_pack_with_non_splat_with_inner_and_outer
// CHECK: %[[CST:.+]] = arith.constant
// CHECK-SAME: [0.000000e+00, 1.000000e+00], [4.000000e+00, 5.000000e+00]
// CHECK-SAME: [8.000000e+00, 9.000000e+00], [1.200000e+01, 1.300000e+01]
// CHECK-SAME: [2.000000e+00, 3.000000e+00], [6.000000e+00, 7.000000e+00]
// CHECK-SAME: [1.000000e+01, 1.100000e+01], [1.400000e+01, 1.500000e+01]
// CHECK-NOT: tensor.pack
// CHECK: return %[[CST]] : tensor<1x2x2x2x2xf32>
func.func @fold_pack_with_non_splat_with_inner_and_outer_permutations() -> tensor<1x2x2x2x2xf32> {
  %cst = arith.constant dense <[[[[0.0, 1.0, 2.0, 3.0],   [4.0, 5.0, 6.0, 7.0]],
                                 [[8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0]]]]> : tensor<1x2x2x4xf32>
  %0 = tensor.empty() : tensor<1x2x2x2x2xf32>
  %1 = tensor.pack %cst outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [2]
    into %0 : tensor<1x2x2x4xf32> -> tensor<1x2x2x2x2xf32>
  return %1 : tensor<1x2x2x2x2xf32>
}

// CHECK-LABEL: func.func @no_fold_pack_into_non_empty_with_non_splat
// CHECK: %[[PACK:.+]] = tensor.pack
// CHECK: return %[[PACK]] : tensor<2x4x2x4xf32>
func.func @no_fold_pack_into_non_empty_with_non_splat(%arg0: tensor<2x4x2x4xf32>) -> tensor<2x4x2x4xf32> {
  %cst = arith.constant dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                               [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                               [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
                               [24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0],
                               [32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0],
                               [40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0],
                               [49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0],
                               [57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0]]> : tensor<8x8xf32>
  %pack = tensor.pack %cst inner_dims_pos = [1, 0] inner_tiles = [2, 4]
    into %arg0 : tensor<8x8xf32> -> tensor<2x4x2x4xf32>
  return %pack : tensor<2x4x2x4xf32>
}

// CHECK-LABEL: func.func @no_fold_dynamic_inner_tile_pack_with_non_splat
// CHECK: %[[PACK:.+]] = tensor.pack
// CHECK: return %[[PACK]] : tensor<?x4x2x?xf32>
func.func @no_fold_dynamic_inner_tile_pack_with_non_splat(%outer: index, %tile: index) -> tensor<?x4x2x?xf32> {
  %cst = arith.constant dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                               [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                               [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
                               [24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0],
                               [32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0],
                               [40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0],
                               [49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0],
                               [57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0]]> : tensor<8x8xf32>
  %0 = tensor.empty(%outer, %tile) : tensor<?x4x2x?xf32>
  %pack = tensor.pack %cst inner_dims_pos = [1, 0] inner_tiles = [2, %tile]
    into %0 : tensor<8x8xf32> -> tensor<?x4x2x?xf32>
  return %pack : tensor<?x4x2x?xf32>
}

// CHECK-LABEL: func.func @no_fold_dynamic_outer_dims_pack_with_non_splat
// CHECK: %[[PACK:.+]] = tensor.pack
// CHECK: return %[[PACK]] : tensor<?x?x2x4xf32>
func.func @no_fold_dynamic_outer_dims_pack_with_non_splat(%dim0: index, %dim1: index) -> tensor<?x?x2x4xf32> {
  %cst = arith.constant dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                               [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                               [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
                               [24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0],
                               [32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0],
                               [40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0],
                               [49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0],
                               [57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0]]> : tensor<8x8xf32>
  %0 = tensor.empty(%dim0, %dim1) : tensor<?x?x2x4xf32>
  %pack = tensor.pack %cst inner_dims_pos = [1, 0] inner_tiles = [2, 4]
    into %0 : tensor<8x8xf32> -> tensor<?x?x2x4xf32>
  return %pack : tensor<?x?x2x4xf32>
}

// CHECK-LABEL: func.func @no_fold_padding_pack_with_non_splat
// CHECK: %[[PACK:.+]] = tensor.pack
// CHECK: return %[[PACK]] : tensor<2x4x2x4xf32>
func.func @no_fold_padding_pack_with_non_splat() -> tensor<2x4x2x4xf32> {
  %cst = arith.constant dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                               [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                               [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
                               [24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0],
                               [32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0],
                               [40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0],
                               [49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0],
                               [57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0]]> : tensor<8x8xf32>
  %0 = tensor.empty() : tensor<2x4x2x4xf32>
  %pad = arith.constant 0.0 : f32
  %pack = tensor.pack %cst padding_value(%pad : f32) inner_dims_pos = [1, 0] inner_tiles = [2, 4]
    into %0 : tensor<8x8xf32> -> tensor<2x4x2x4xf32>
  return %pack : tensor<2x4x2x4xf32>
}
