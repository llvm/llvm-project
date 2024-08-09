// RUN: mlir-opt -xegpu-fold-alias-ops -split-input-file %s | FileCheck %s

func.func @fold_subview_with_xegpu_create_nd_tdesc(%arg0 : memref<256x256xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index) ->(!xegpu.tensor_desc<8x16xf32>) {
  %subview = memref.subview %arg0[%arg1, %arg2] [32, 32] [1, 1] :
    memref<256x256xf32> to memref<32x32xf32, strided<[256, 1], offset: ?>>
  %0 = xegpu.create_nd_tdesc %subview[%arg3, %arg4] :
    memref<32x32xf32, strided<[256, 1], offset: ?>> -> !xegpu.tensor_desc<8x16xf32>
  return %0 : !xegpu.tensor_desc<8x16xf32>
}

//   CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
//       CHECK: func @fold_subview_with_xegpu_create_nd_tdesc
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: memref<256x256xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[ARG4:[a-zA-Z0-9]+]]: index
//   CHECK-DAG:   %[[IDX0:.+]] = affine.apply #[[MAP]]()[%[[ARG1]], %[[ARG3]]]
//   CHECK-DAG:   %[[IDX1:.+]] = affine.apply #[[MAP]]()[%[[ARG2]], %[[ARG4]]]
//   CHECK:       xegpu.create_nd_tdesc %[[ARG0]][%[[IDX0]], %[[IDX1]]] : memref<256x256xf32> -> !xegpu.tensor_desc<8x16xf32>

// -----
func.func @fold_subview_with_xegpu_create_nd_tdesc(%arg0 : memref<32x256x256xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index) ->(!xegpu.tensor_desc<8x16xf32>) {
  %subview = memref.subview %arg0[%arg1, %arg2, %arg3] [1, 32, 32] [1, 1, 1] :
    memref<32x256x256xf32> to memref<32x32xf32, strided<[256, 1], offset: ?>>
  %0 = xegpu.create_nd_tdesc %subview[%arg4, %arg5] :
    memref<32x32xf32, strided<[256, 1], offset: ?>> -> !xegpu.tensor_desc<8x16xf32>
  return %0 : !xegpu.tensor_desc<8x16xf32>
}

//   CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
//       CHECK: func @fold_subview_with_xegpu_create_nd_tdesc
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: memref<32x256x256xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[ARG4:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[ARG5:[a-zA-Z0-9]+]]: index
//       CHECK:   %[[C65536:[a-zA-Z0-9]+]] = arith.constant 65536 : index
//   CHECK-DAG:   %[[IDX0:.+]] = affine.apply #[[MAP]]()[%[[ARG2]], %[[ARG4]]]
//   CHECK-DAG:   %[[IDX1:.+]] = affine.apply #[[MAP]]()[%[[ARG3]], %[[ARG5]]]
//       CHECK:   %[[CAST:.+]] = memref.reinterpret_cast %[[ARG0]] to offset: [0], sizes: [8192, 256], strides: [256, 1] : memref<32x256x256xf32> to memref<8192x256xf32>
//       CHECK:   %[[T1:.+]] = arith.muli %[[ARG1]], %[[C65536]] : index
//       CHECK:   %[[T2:.+]] = arith.addi %[[T1]], %[[IDX0]] : index
//       CHECK:   xegpu.create_nd_tdesc %[[CAST]][%[[T2]], %[[IDX1]]] : memref<8192x256xf32> -> !xegpu.tensor_desc<8x16xf32>

// -----
func.func @fold_subview_with_xegpu_create_nd_tdesc(%arg0 : memref<32x32x256x256xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index, %arg6: index) ->(!xegpu.tensor_desc<8x16xf32>) {
  %subview = memref.subview %arg0[%arg1, %arg2, %arg3, %arg4] [1, 1, 32, 32] [1, 1, 1, 1] :
    memref<32x32x256x256xf32> to memref<32x32xf32, strided<[256, 1], offset: ?>>
  %0 = xegpu.create_nd_tdesc %subview[%arg5, %arg6] :
    memref<32x32xf32, strided<[256, 1], offset: ?>> -> !xegpu.tensor_desc<8x16xf32>
  return %0 : !xegpu.tensor_desc<8x16xf32>
}

//   CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
//       CHECK: func @fold_subview_with_xegpu_create_nd_tdesc
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: memref<32x32x256x256xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[ARG4:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[ARG5:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[ARG6:[a-zA-Z0-9]+]]: index
//       CHECK:   %[[C2097152:[a-zA-Z0-9]+]] = arith.constant 2097152 : index
//       CHECK:   %[[C65536:[a-zA-Z0-9]+]] = arith.constant 65536 : index
//   CHECK-DAG:   %[[IDX0:.+]] = affine.apply #[[MAP]]()[%[[ARG3]], %[[ARG5]]]
//   CHECK-DAG:   %[[IDX1:.+]] = affine.apply #[[MAP]]()[%[[ARG4]], %[[ARG6]]]
//       CHECK:   %[[CAST:.+]] = memref.reinterpret_cast %[[ARG0]] to offset: [0], sizes: [262144, 256], strides: [256, 1] : memref<32x32x256x256xf32> to memref<262144x256xf32>
//       CHECK:   %[[T1:.+]] = arith.muli %[[ARG2]], %[[C65536]] : index
//       CHECK:   %[[T2:.+]] = arith.addi %[[T1]], %[[IDX0]] : index
//       CHECK:   %[[T3:.+]] = arith.muli %[[ARG1]], %[[C2097152]] : index
//       CHECK:   %[[T4:.+]] = arith.addi %[[T3]], %[[T2]] : index
//       CHECK:   xegpu.create_nd_tdesc %[[CAST]][%[[T4]], %[[IDX1]]] : memref<262144x256xf32> -> !xegpu.tensor_desc<8x16xf32>
