// RUN: mlir-opt %s -linalg-fuse-elementwise-ops -split-input-file | FileCheck %s

// CHECK-LABEL: @transpose_fold_2d_fp32
func.func @transpose_fold_2d_fp32(%init: tensor<3x2xf32>) -> tensor<3x2xf32> {
  %input = arith.constant dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]> : tensor<2x3xf32>
  //               CHECK: %[[CST:.+]] = arith.constant
  // CHECK-SAME{LITERAL}:   dense<[[0.000000e+00, 3.000000e+00], [1.000000e+00, 4.000000e+00], [2.000000e+00, 5.000000e+00]]> : tensor<3x2xf32>
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<2x3xf32>) outs(%init : tensor<3x2xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    linalg.yield %arg1 : f32
  } -> tensor<3x2xf32>
  // CHECK: return %[[CST]]
  return %1 : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: @transpose_fold_2d_fp64
func.func @transpose_fold_2d_fp64(%init: tensor<3x2xf64>) -> tensor<3x2xf64> {
  %input = arith.constant dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]> : tensor<2x3xf64>
  //               CHECK: %[[CST:.+]] = arith.constant
  // CHECK-SAME{LITERAL}:   dense<[[0.000000e+00, 3.000000e+00], [1.000000e+00, 4.000000e+00], [2.000000e+00, 5.000000e+00]]> : tensor<3x2xf64>
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<2x3xf64>) outs(%init : tensor<3x2xf64>) {
  ^bb0(%arg1: f64, %arg2: f64):
    linalg.yield %arg1 : f64
  } -> tensor<3x2xf64>
  // CHECK: return %[[CST]]
  return %1 : tensor<3x2xf64>
}

// -----

// CHECK-LABEL: @transpose_fold_4d_i32
func.func @transpose_fold_4d_i32(%init: tensor<3x1x4x2xi32>) -> tensor<3x1x4x2xi32> {
  %input = arith.constant dense<[[
    [[ 0,  1,  2,  3], [ 4,  5,  6,  7], [ 8,  9, 10, 11]],
    [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]
  ]]> : tensor<1x2x3x4xi32>
  //               CHECK: %[[CST:.+]] = arith.constant dense<[
  // CHECK-SAME{LITERAL}:   [[[0, 12], [1, 13], [2, 14], [3, 15]]],
  // CHECK-SAME{LITERAL}:   [[[4, 16], [5, 17], [6, 18], [7, 19]]],
  // CHECK-SAME{LITERAL}:   [[[8, 20], [9, 21], [10, 22], [11, 23]]]
  // CHECK-SAME{LITERAL}: ]>
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%input : tensor<1x2x3x4xi32>) outs(%init : tensor<3x1x4x2xi32>) {
  ^bb0(%arg1: i32, %arg2: i32):
    linalg.yield %arg1 : i32
  } -> tensor<3x1x4x2xi32>
  // CHECK: return %[[CST]]
  return %1 : tensor<3x1x4x2xi32>
}

// -----

// CHECK-LABEL: @transpose_fold_4d_i16
func.func @transpose_fold_4d_i16(%init: tensor<3x1x4x2xi16>) -> tensor<3x1x4x2xi16> {
  %input = arith.constant dense<[[
    [[ 0,  1,  2,  3], [ 4,  5,  6,  7], [ 8,  9, 10, 11]],
    [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]
  ]]> : tensor<1x2x3x4xi16>
  //               CHECK: %[[CST:.+]] = arith.constant dense<[
  // CHECK-SAME{LITERAL}:   [[[0, 12], [1, 13], [2, 14], [3, 15]]],
  // CHECK-SAME{LITERAL}:   [[[4, 16], [5, 17], [6, 18], [7, 19]]],
  // CHECK-SAME{LITERAL}:   [[[8, 20], [9, 21], [10, 22], [11, 23]]]
  // CHECK-SAME{LITERAL}: ]>
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%input : tensor<1x2x3x4xi16>) outs(%init : tensor<3x1x4x2xi16>) {
  ^bb0(%arg1: i16, %arg2: i16):
    linalg.yield %arg1 : i16
  } -> tensor<3x1x4x2xi16>
  // CHECK: return %[[CST]]
  return %1 : tensor<3x1x4x2xi16>
}

// -----

// CHECK-LABEL: @transpose_nofold_non_cst_input
func.func @transpose_nofold_non_cst_input(%input: tensor<2x3xf32>, %init: tensor<3x2xf32>) -> tensor<3x2xf32> {
  // CHECK: linalg.generic
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<2x3xf32>) outs(%init : tensor<3x2xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    linalg.yield %arg1 : f32
  } -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: @transpose_nofold_yield_const
func.func @transpose_nofold_yield_const(%init: tensor<3x2xf32>) -> tensor<3x2xf32> {
  %input = arith.constant dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]> : tensor<2x3xf32>
  %cst = arith.constant 8.0 : f32
  // CHECK: linalg.generic
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<2x3xf32>) outs(%init : tensor<3x2xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    linalg.yield %cst : f32
  } -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: @transpose_nofold_multi_ops_in_region
func.func @transpose_nofold_multi_ops_in_region(%init: tensor<3x2xf32>) -> tensor<3x2xf32> {
  %input = arith.constant dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]> : tensor<2x3xf32>
  // CHECK: linalg.generic
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<2x3xf32>) outs(%init : tensor<3x2xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    %add = arith.addf %arg1, %arg1 : f32
    linalg.yield %add : f32
  } -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: @named_transpose_fold_2d_fp32
func.func @named_transpose_fold_2d_fp32(%init: tensor<3x2xf32>) -> tensor<3x2xf32> {
  %input = arith.constant dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]> : tensor<2x3xf32>
  //               CHECK: %[[CST:.+]] = arith.constant
  // CHECK-SAME{LITERAL}:   dense<[[0.000000e+00, 3.000000e+00], [1.000000e+00, 4.000000e+00], [2.000000e+00, 5.000000e+00]]> : tensor<3x2xf32>
  %1 = linalg.transpose ins(%input : tensor<2x3xf32>) outs(%init : tensor<3x2xf32>) permutation = [1, 0]
  // CHECK: return %[[CST]]
  return %1 : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: @elementwise_fold_add_f32
func.func @elementwise_fold_add_f32(%init: tensor<4xf32>) -> tensor<4xf32> {
  %lhs = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %rhs = arith.constant dense<[5.0, 6.0, 7.0, 8.0]> : tensor<4xf32>
  //               CHECK: %[[CST:.+]] = arith.constant
  // CHECK-SAME{LITERAL}:   dense<[6.000000e+00, 8.000000e+00, 1.000000e+01, 1.200000e+01]> : tensor<4xf32>
  %1 = linalg.elementwise kind=#linalg.elementwise_kind<add>
    ins(%lhs, %rhs : tensor<4xf32>, tensor<4xf32>)
    outs(%init : tensor<4xf32>) -> tensor<4xf32>
  // CHECK: return %[[CST]]
  return %1 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @elementwise_fold_mul_i32
func.func @elementwise_fold_mul_i32(%init: tensor<3xi32>) -> tensor<3xi32> {
  %lhs = arith.constant dense<[2, 3, 4]> : tensor<3xi32>
  %rhs = arith.constant dense<[5, 6, 7]> : tensor<3xi32>
  //               CHECK: %[[CST:.+]] = arith.constant
  // CHECK-SAME{LITERAL}:   dense<[10, 18, 28]> : tensor<3xi32>
  %1 = linalg.elementwise kind=#linalg.elementwise_kind<mul>
    ins(%lhs, %rhs : tensor<3xi32>, tensor<3xi32>)
    outs(%init : tensor<3xi32>) -> tensor<3xi32>
  // CHECK: return %[[CST]]
  return %1 : tensor<3xi32>
}

// -----

// CHECK-LABEL: @elementwise_fold_sub_f64
func.func @elementwise_fold_sub_f64(%init: tensor<2x2xf64>) -> tensor<2x2xf64> {
  %lhs = arith.constant dense<[[10.0, 20.0], [30.0, 40.0]]> : tensor<2x2xf64>
  %rhs = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>
  //               CHECK: %[[CST:.+]] = arith.constant
  // CHECK-SAME{LITERAL}:   dense<[[9.000000e+00, 1.800000e+01], [2.700000e+01, 3.600000e+01]]> : tensor<2x2xf64>
  %1 = linalg.elementwise kind=#linalg.elementwise_kind<sub>
    ins(%lhs, %rhs : tensor<2x2xf64>, tensor<2x2xf64>)
    outs(%init : tensor<2x2xf64>) -> tensor<2x2xf64>
  // CHECK: return %[[CST]]
  return %1 : tensor<2x2xf64>
}

// -----

// CHECK-LABEL: @elementwise_fold_sin_f32
func.func @elementwise_fold_sin_f32(%init: tensor<4xf32>) -> tensor<4xf32> {
  %input = arith.constant dense<[0.0, 1.0, 2.0, 3.0]> : tensor<4xf32>
  //               CHECK: %[[CST:.+]] = arith.constant
  // CHECK-SAME{LITERAL}:   dense<[0.000000e+00, 0.841470957, 0.909297406, 1.411200e-01]> : tensor<4xf32>
  %1 = linalg.elementwise kind=#linalg.elementwise_kind<sin>
    ins(%input : tensor<4xf32>)
    outs(%init : tensor<4xf32>) -> tensor<4xf32>
  // CHECK: return %[[CST]]
  return %1 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @elementwise_fold_cos_f32
func.func @elementwise_fold_cos_f32(%init: tensor<4xf32>) -> tensor<4xf32> {
  %input = arith.constant dense<[0.0, 1.0, 2.0, 3.0]> : tensor<4xf32>
  //               CHECK: %[[CST:.+]] = arith.constant
  // CHECK-SAME{LITERAL}:   dense<[1.000000e+00, 0.540302277, -0.416146845, -0.989992499]> : tensor<4xf32>
  %1 = linalg.elementwise kind=#linalg.elementwise_kind<cos>
    ins(%input : tensor<4xf32>)
    outs(%init : tensor<4xf32>) -> tensor<4xf32>
  // CHECK: return %[[CST]]
  return %1 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @elementwise_nofold_non_cst_input
func.func @elementwise_nofold_non_cst_input(%input: tensor<4xf32>, %init: tensor<4xf32>) -> tensor<4xf32> {
  %rhs = arith.constant dense<[5.0, 6.0, 7.0, 8.0]> : tensor<4xf32>
  // CHECK: linalg.elementwise
  %1 = linalg.elementwise kind=#linalg.elementwise_kind<add>
    ins(%input, %rhs : tensor<4xf32>, tensor<4xf32>)
    outs(%init : tensor<4xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// Verify that multi-use constants are not folded (controlFn requires single use).
// CHECK-LABEL: @elementwise_nofold_multi_use_cst
func.func @elementwise_nofold_multi_use_cst(%init1: tensor<4xf32>, %init2: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %cst = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  // CHECK: linalg.elementwise kind=#linalg.elementwise_kind<sin>
  %1 = linalg.elementwise kind=#linalg.elementwise_kind<sin>
    ins(%cst : tensor<4xf32>)
    outs(%init1 : tensor<4xf32>) -> tensor<4xf32>
  // CHECK: linalg.elementwise kind=#linalg.elementwise_kind<cos>
  %2 = linalg.elementwise kind=#linalg.elementwise_kind<cos>
    ins(%cst : tensor<4xf32>)
    outs(%init2 : tensor<4xf32>) -> tensor<4xf32>
  return %1, %2 : tensor<4xf32>, tensor<4xf32>
}

