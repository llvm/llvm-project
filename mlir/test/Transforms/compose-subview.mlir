// RUN: mlir-opt -allow-unregistered-dialect %s -test-compose-subview -split-input-file | FileCheck %s

// CHECK: [[MAP:#.*]] = affine_map<(d0, d1) -> (d0 * 1024 + d1 + 3456)
#map0 = affine_map<(d0, d1) -> (d0 * 1024 + d1 + 2304)>
#map1 = affine_map<(d0, d1) -> (d0 * 1024 + d1 + 3456)>

func @main(%input: memref<4x1024xf32>) -> memref<1x128xf32, #map1> {
  //      CHECK: subview %arg0[3, 384] [1, 128] [1, 1]
  // CHECK-SAME: memref<4x1024xf32> to memref<1x128xf32, [[MAP]]>
  %0 = memref.subview %input[2, 256] [2, 256] [1, 1] : memref<4x1024xf32> to memref<2x256xf32, #map0>
  %1 = memref.subview %0[1, 128] [1, 128] [1, 1] : memref<2x256xf32, #map0> to memref<1x128xf32, #map1>
  return %1 : memref<1x128xf32, #map1>
}

// -----

// CHECK: [[MAP:#.*]] = affine_map<(d0, d1) -> (d0 * 1024 + d1 + 3745)
#map0 = affine_map<(d0, d1) -> (d0 * 1024 + d1 + 1536)>
#map1 = affine_map<(d0, d1) -> (d0 * 1024 + d1 + 2688)>
#map2 = affine_map<(d0, d1) -> (d0 * 1024 + d1 + 3745)>

func @main(%input: memref<4x1024xf32>) -> memref<1x10xf32, #map2> {
  //      CHECK: subview %arg0[3, 673] [1, 10] [1, 1]
  // CHECK-SAME: memref<4x1024xf32> to memref<1x10xf32, [[MAP]]>
  %0 = memref.subview %input[1, 512] [3, 256] [1, 1] : memref<4x1024xf32> to memref<3x256xf32, #map0>
  %1 = memref.subview %0[1, 128] [2, 128] [1, 1] : memref<3x256xf32, #map0> to memref<2x128xf32, #map1>
  %2 = memref.subview %1[1, 33] [1, 10] [1, 1] : memref<2x128xf32, #map1> to memref<1x10xf32, #map2>
  return %2 : memref<1x10xf32, #map2>
}

// -----

// CHECK: [[MAP:#.*]] = affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)
#map = affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>

func @main(%input: memref<4x1024xf32>) -> memref<1x128xf32, #map> {
  // CHECK: [[CST_3:%.*]] = arith.constant 3 : index
  %cst_1 = arith.constant 1 : index
  %cst_2 = arith.constant 2 : index
  //      CHECK: subview %arg0{{\[}}[[CST_3]], 384] [1, 128] [1, 1]
  // CHECK-SAME: memref<4x1024xf32> to memref<1x128xf32, [[MAP]]>
  %0 = memref.subview %input[%cst_2, 256] [2, 256] [1, 1] : memref<4x1024xf32> to memref<2x256xf32, #map>
  %1 = memref.subview %0[%cst_1, 128] [1, 128] [1, 1] : memref<2x256xf32, #map> to memref<1x128xf32, #map>
  return %1 : memref<1x128xf32, #map>
}

// -----

// CHECK: [[MAP:#.*]] = affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)
#map = affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>

func @main(%input: memref<4x1024xf32>) -> memref<1x128xf32, #map> {
  // CHECK: [[CST_3:%.*]] = arith.constant 3 : index
  %cst_2 = arith.constant 2 : index
  // CHECK: [[CST_384:%.*]] = arith.constant 384 : index
  %cst_128 = arith.constant 128 : index
  //      CHECK: subview %arg0{{\[}}[[CST_3]], [[CST_384]]] [1, 128] [1, 1]
  // CHECK-SAME: memref<4x1024xf32> to memref<1x128xf32, [[MAP]]>
  %0 = memref.subview %input[%cst_2, 256] [2, 256] [1, 1] : memref<4x1024xf32> to memref<2x256xf32, #map>
  %1 = memref.subview %0[1, %cst_128] [1, 128] [1, 1] : memref<2x256xf32, #map> to memref<1x128xf32, #map>
  return %1 : memref<1x128xf32, #map>
}
