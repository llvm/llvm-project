// RUN: mlir-opt -xegpu-fold-alias-ops -split-input-file %s | FileCheck %s

func.func @fold_subview_with_xegpu_create_nd_tdesc(%arg0 : memref<256x256xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index) -> vector<8x16xf32> {
  %subview = memref.subview %arg0[%arg1, %arg2] [32, 32] [1, 1] :
    memref<256x256xf32> to memref<32x32xf32, strided<[256, 1], offset: ?>>
  %0 = xegpu.create_nd_tdesc %subview :
    memref<32x32xf32, strided<[256, 1], offset: ?>> -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.load_nd %0[%arg3, %arg4] : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  return %1 : vector<8x16xf32>
}

//       CHECK: func @fold_subview_with_xegpu_create_nd_tdesc
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: memref<256x256xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[ARG4:[a-zA-Z0-9]+]]: index
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[ARG0]][%[[ARG1]], %[[ARG2]]] [32, 32] [1, 1] : memref<256x256xf32> to memref<32x32xf32, strided<[256, 1], offset: ?>>
//       CHECK:   %[[TDESC:.+]] = xegpu.create_nd_tdesc %[[SUBVIEW]] : memref<32x32xf32, strided<[256, 1], offset: ?>> -> !xegpu.tensor_desc<8x16xf32>
//       CHECK:   %[[LOAD:.+]] = xegpu.load_nd %[[TDESC]][%[[ARG3]], %[[ARG4]]] : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
