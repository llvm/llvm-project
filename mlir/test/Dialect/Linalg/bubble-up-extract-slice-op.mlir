//RUN: mlir-opt -test-linalg-transform-patterns=test-bubble-up-extract-slice-op-pattern -split-input-file %s | FileCheck %s

func.func @dynamic(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>, %arg2: index, %arg3: index, %arg4: index, %arg5:index) -> tensor<?x?xf32> {
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?xf32>)
    outs(%arg0 : tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %add = arith.addf %b0, %b1 : f32
      linalg.yield %add : f32
  } -> tensor<?x?xf32>
  %1 = tensor.extract_slice %0 [%arg2, %arg3] [%arg4, %arg5] [1, 1]
    : tensor<?x?xf32> to tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

//      CHECK: func @dynamic
//      CHECK: %[[SLICE0:.+]] = tensor.extract_slice %arg0[%arg2, %arg3] [%arg4, %arg5] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK: %[[SLICE1:.+]] = tensor.extract_slice %arg1[%arg3] [%arg5] [1] : tensor<?xf32> to tensor<?xf32>
//      CHECK: %[[SLICE2:.+]] = tensor.extract_slice %arg0[%arg2, %arg3] [%arg4, %arg5] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: ins(%[[SLICE0]], %[[SLICE1]] : tensor<?x?xf32>, tensor<?xf32>) outs(%[[SLICE2]] : tensor<?x?xf32>)
//      CHECK: return %[[GENERIC]] : tensor<?x?xf32>

//-----

func.func @static(%arg0: tensor<16x8xf32>, %arg1: tensor<8xf32>) -> tensor<4x2xf32> {
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<16x8xf32>, tensor<8xf32>)
    outs(%arg0 : tensor<16x8xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %add = arith.addf %b0, %b1 : f32
      linalg.yield %add : f32
  } -> tensor<16x8xf32>
  %1 = tensor.extract_slice %0 [8, 4] [4, 2] [1, 1]
    : tensor<16x8xf32> to tensor<4x2xf32>
  return %1 : tensor<4x2xf32>
}

//      CHECK: func @static
//      CHECK: %[[SLICE0:.+]] = tensor.extract_slice %arg0[8, 4] [4, 2] [1, 1] : tensor<16x8xf32> to tensor<4x2xf32>
//      CHECK: %[[SLICE1:.+]] = tensor.extract_slice %arg1[4] [2] [1] : tensor<8xf32> to tensor<2xf32>
//      CHECK: %[[SLICE2:.+]] = tensor.extract_slice %arg0[8, 4] [4, 2] [1, 1] : tensor<16x8xf32> to tensor<4x2xf32>
//      CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: ins(%[[SLICE0]], %[[SLICE1]] : tensor<4x2xf32>, tensor<2xf32>) outs(%[[SLICE2]] : tensor<4x2xf32>)
//      CHECK: return %[[GENERIC]] : tensor<4x2xf32>

//-----

func.func @mixed(%arg0: tensor<?x8xf32>, %arg1: tensor<8xf32>, %arg2: index, %arg3: index) -> tensor<?x2xf32> {
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<?x8xf32>, tensor<8xf32>)
    outs(%arg0 : tensor<?x8xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %add = arith.addf %b0, %b1 : f32
      linalg.yield %add : f32
  } -> tensor<?x8xf32>
  %1 = tensor.extract_slice %0 [8, %arg2] [%arg3, 2] [1, 1]
    : tensor<?x8xf32> to tensor<?x2xf32>
  return %1 : tensor<?x2xf32>
}

//      CHECK: func @mixed
//      CHECK: %[[SLICE0:.+]] = tensor.extract_slice %arg0[8, %arg2] [%arg3, 2] [1, 1] : tensor<?x8xf32> to tensor<?x2xf32>
//      CHECK: %[[SLICE1:.+]] = tensor.extract_slice %arg1[%arg2] [2] [1] : tensor<8xf32> to tensor<2xf32>
//      CHECK: %[[SLICE2:.+]] = tensor.extract_slice %arg0[8, %arg2] [%arg3, 2] [1, 1] : tensor<?x8xf32> to tensor<?x2xf32>
//      CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: ins(%[[SLICE0]], %[[SLICE1]] : tensor<?x2xf32>, tensor<2xf32>) outs(%[[SLICE2]] : tensor<?x2xf32>)
//      CHECK: return %[[GENERIC]] : tensor<?x2xf32>

//-----

func.func @dynamic_to_static(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>) -> tensor<4x2xf32> {
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?xf32>)
    outs(%arg0 : tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %add = arith.addf %b0, %b1 : f32
      linalg.yield %add : f32
  } -> tensor<?x?xf32>
  %1 = tensor.extract_slice %0 [8, 4] [4, 2] [1, 1]
    : tensor<?x?xf32> to tensor<4x2xf32>
  return %1 : tensor<4x2xf32>
}

//      CHECK: func @dynamic_to_static
//      CHECK: %[[SLICE0:.+]] = tensor.extract_slice %arg0[8, 4] [4, 2] [1, 1] : tensor<?x?xf32> to tensor<4x2xf32>
//      CHECK: %[[SLICE1:.+]] = tensor.extract_slice %arg1[4] [2] [1] : tensor<?xf32> to tensor<2xf32>
//      CHECK: %[[SLICE2:.+]] = tensor.extract_slice %arg0[8, 4] [4, 2] [1, 1] : tensor<?x?xf32> to tensor<4x2xf32>
//      CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: ins(%[[SLICE0]], %[[SLICE1]] : tensor<4x2xf32>, tensor<2xf32>) outs(%[[SLICE2]] : tensor<4x2xf32>)
//      CHECK: return %[[GENERIC]] : tensor<4x2xf32>

//-----

func.func @matmul_slice() -> tensor<2x2xf32> {
    %lhs = arith.constant dense<1.0> : tensor<4x4xf32>
    %rhs = arith.constant dense<1.0> : tensor<4x4xf32>
    %dst = arith.constant dense<[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0]]> : tensor<4x4xf32>
    %0 = linalg.matmul ins(%lhs, %rhs : tensor<4x4xf32>, tensor<4x4xf32>) outs(%dst : tensor<4x4xf32>) -> tensor<4x4xf32>
    %1 = tensor.extract_slice %0[1,1][2,2][1,1] : tensor<4x4xf32> to tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
}

// CHECK: func @matmul_slice
// CHECK: %[[SLICE0:.+]] = arith.constant dense<1.000000e+00> : tensor<2x4xf32>
// CHECK: %[[SLICE1:.+]] = arith.constant dense<1.000000e+00> : tensor<4x2xf32>
// CHECK: %[[SLICE3:.+]] = tensor.extract_slice %[[CST:.+]][1, 1] [2, 2] [1, 1] : tensor<4x4xf32> to tensor<2x2xf32>
// CHECK: %[[MATMUL:.+]] = linalg.matmul ins(%[[SLICE0]], %[[SLICE1]] : tensor<2x4xf32>, tensor<4x2xf32>) outs(%[[SLICE3]] : tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK: return %[[MATMUL]] : tensor<2x2xf32>

//-----

func.func @conv_slice(%input: tensor<1x225x225x3xf32>, %filter: tensor<3x3x3x32xf32>) -> tensor<1x32x32x16xf32> {
  %c112 = arith.constant 112 : index
  %c32 = arith.constant 32 : index
  %c16 = arith.constant 16 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32

  %init = tensor.empty() : tensor<1x112x112x32xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>

  %conv = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
    ins(%input, %filter : tensor<1x225x225x3xf32>, tensor<3x3x3x32xf32>)
    outs(%fill : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>

  %slice = tensor.extract_slice %conv [0, 64, 64, 16] [1, 32, 32, 16] [1, 1, 1, 1] : tensor<1x112x112x32xf32> to tensor<1x32x32x16xf32>

  return %slice : tensor<1x32x32x16xf32>
}

// CHECK: func @conv_slice
// CHECK: %[[INIT:.+]] = tensor.empty() : tensor<1x112x112x32xf32>
// CHECK: %[[SLICE0:.+]] = tensor.extract_slice %arg0[0, 128, 128, 0] [1, 65, 65, 3] [1, 1, 1, 1] : tensor<1x225x225x3xf32> to tensor<1x65x65x3xf32>
// CHECK: %[[SLICE1:.+]] = tensor.extract_slice %arg1[0, 0, 0, 16] [3, 3, 3, 16] [1, 1, 1, 1] : tensor<3x3x3x32xf32> to tensor<3x3x3x16xf32>
// CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[INIT]][0, 64, 64, 16] [1, 32, 32, 16] [1, 1, 1, 1] : tensor<1x112x112x32xf32> to tensor<1x32x32x16xf32>
// CHECK: %[[FILL:.+]] = linalg.fill ins(%[[CST:.+]] : f32) outs(%[[SLICE2]] : tensor<1x32x32x16xf32>) -> tensor<1x32x32x16xf32>
// CHECK: %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%[[SLICE0]], %[[SLICE1]] : tensor<1x65x65x3xf32>, tensor<3x3x3x16xf32>) outs(%[[FILL]] : tensor<1x32x32x16xf32>) -> tensor<1x32x32x16xf32>
// CHECK: return %[[CONV]] : tensor<1x32x32x16xf32>

//-----

// The slice is not supposed to be bubbled up when it is rank-reducing.
func.func @rank_reducing_slice(%width : index) -> tensor<1x1x1x?xf32> {
  %cst = arith.constant 1.000000e+00 : f32
  %init = tensor.empty(%width) : tensor<1x?xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x?xf32>) -> tensor<1x?xf32>
  %slice = tensor.extract_slice %fill[0, 0] [1, %width] [1, 1] : tensor<1x?xf32> to tensor<?xf32>
  %expand = tensor.expand_shape %slice [[0, 1, 2, 3]] : tensor<?xf32> into tensor<1x1x1x?xf32>
  return %expand : tensor<1x1x1x?xf32>
}

// CHECK: func @rank_reducing_slice
// CHECK: %[[INIT:.+]] = tensor.empty
// CHECK: %[[FILL:.+]] = linalg.fill ins
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[FILL]]
// CHECK: %[[EXPAND:.+]] = tensor.expand_shape %[[SLICE]]
// CHECK: return %[[EXPAND]]
