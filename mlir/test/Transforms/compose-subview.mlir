// RUN: mlir-opt %s -test-compose-subview -split-input-file | FileCheck %s

func.func @main(%input: memref<4x1024xf32>) -> memref<1x128xf32, strided<[1024, 1], offset: 3456>> {
  //      CHECK: subview %arg0[3, 384] [1, 128] [1, 1]
  // CHECK-SAME: memref<4x1024xf32> to memref<1x128xf32, strided<[1024, 1], offset: 3456>>
  %0 = memref.subview %input[2, 256] [2, 256] [1, 1] : memref<4x1024xf32> to memref<2x256xf32, strided<[1024, 1], offset: 2304>>
  %1 = memref.subview %0[1, 128] [1, 128] [1, 1] : memref<2x256xf32, strided<[1024, 1], offset: 2304>> to memref<1x128xf32, strided<[1024, 1], offset: 3456>>
  return %1 : memref<1x128xf32, strided<[1024, 1], offset: 3456>>
}

// -----

func.func @main(%input: memref<4x1024xf32>) -> memref<1x10xf32, strided<[1024, 1], offset: 3745>> {
  //      CHECK: subview %arg0[3, 673] [1, 10] [1, 1]
  // CHECK-SAME: memref<4x1024xf32> to memref<1x10xf32, strided<[1024, 1], offset: 3745>>
  %0 = memref.subview %input[1, 512] [3, 256] [1, 1] : memref<4x1024xf32> to memref<3x256xf32, strided<[1024, 1], offset: 1536>>
  %1 = memref.subview %0[1, 128] [2, 128] [1, 1] : memref<3x256xf32, strided<[1024, 1], offset: 1536>> to memref<2x128xf32, strided<[1024, 1], offset: 2688>>
  %2 = memref.subview %1[1, 33] [1, 10] [1, 1] : memref<2x128xf32, strided<[1024, 1], offset: 2688>> to memref<1x10xf32, strided<[1024, 1], offset: 3745>>
  return %2 : memref<1x10xf32, strided<[1024, 1], offset: 3745>>
}

// -----

func.func @main(%input: memref<4x1024xf32>) -> memref<1x128xf32, strided<[1024, 1], offset: ?>> {
  // CHECK: [[CST_3:%.*]] = arith.constant 3 : index
  %cst_1 = arith.constant 1 : index
  %cst_2 = arith.constant 2 : index
  //      CHECK: subview %arg0{{\[}}[[CST_3]], 384] [1, 128] [1, 1]
  // CHECK-SAME: memref<4x1024xf32> to memref<1x128xf32, strided<[1024, 1], offset: ?>>
  %0 = memref.subview %input[%cst_2, 256] [2, 256] [1, 1] : memref<4x1024xf32> to memref<2x256xf32, strided<[1024, 1], offset: ?>>
  %1 = memref.subview %0[%cst_1, 128] [1, 128] [1, 1] : memref<2x256xf32, strided<[1024, 1], offset: ?>> to memref<1x128xf32, strided<[1024, 1], offset: ?>>
  return %1 : memref<1x128xf32, strided<[1024, 1], offset: ?>>
}

// -----

func.func @main(%input: memref<4x1024xf32>) -> memref<1x128xf32, strided<[1024, 1], offset: ?>> {
  // CHECK: [[CST_3:%.*]] = arith.constant 3 : index
  %cst_2 = arith.constant 2 : index
  // CHECK: [[CST_384:%.*]] = arith.constant 384 : index
  %cst_128 = arith.constant 128 : index
  //      CHECK: subview %arg0{{\[}}[[CST_3]], [[CST_384]]] [1, 128] [1, 1]
  // CHECK-SAME: memref<4x1024xf32> to memref<1x128xf32, strided<[1024, 1], offset: ?>>
  %0 = memref.subview %input[%cst_2, 256] [2, 256] [1, 1] : memref<4x1024xf32> to memref<2x256xf32, strided<[1024, 1], offset: ?>>
  %1 = memref.subview %0[1, %cst_128] [1, 128] [1, 1] : memref<2x256xf32, strided<[1024, 1], offset: ?>> to memref<1x128xf32, strided<[1024, 1], offset: ?>>
  return %1 : memref<1x128xf32, strided<[1024, 1], offset: ?>>
}
