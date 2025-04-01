// RUN: mlir-opt %s -test-compose-subview -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @subview_strided(
// CHECK-SAME:  %[[VAL_0:.*]]: memref<4x1024xf32>) -> memref<1x128xf32, strided<[1024, 1], offset: 3456>> {
func.func @subview_strided(%input: memref<4x1024xf32>) -> memref<1x128xf32, strided<[1024, 1], offset: 3456>> {
  // CHECK: %[[VAL_1:.*]] = memref.subview %[[VAL_0]][3, 384] [1, 128] [1, 1] : memref<4x1024xf32> to memref<1x128xf32, strided<[1024, 1], offset: 3456>>
  %0 = memref.subview %input[2, 256] [2, 256] [1, 1] : memref<4x1024xf32> to memref<2x256xf32, strided<[1024, 1], offset: 2304>>
  %1 = memref.subview %0[1, 128] [1, 128] [1, 1] : memref<2x256xf32, strided<[1024, 1], offset: 2304>> to memref<1x128xf32, strided<[1024, 1], offset: 3456>>
  return %1 : memref<1x128xf32, strided<[1024, 1], offset: 3456>>
}

// -----

// CHECK-LABEL: func.func @subview_strided(
// CHECK-SAME:  %[[VAL_0:.*]]: memref<4x1024xf32>) -> memref<1x10xf32, strided<[1024, 1], offset: 3745>> {
func.func @subview_strided(%input: memref<4x1024xf32>) -> memref<1x10xf32, strided<[1024, 1], offset: 3745>> {
  // CHECK:     %[[VAL_1:.*]] = memref.subview %[[VAL_0]][3, 673] [1, 10] [1, 1] : memref<4x1024xf32> to memref<1x10xf32, strided<[1024, 1], offset: 3745>>
  %0 = memref.subview %input[1, 512] [3, 256] [1, 1] : memref<4x1024xf32> to memref<3x256xf32, strided<[1024, 1], offset: 1536>>
  %1 = memref.subview %0[1, 128] [2, 128] [1, 1] : memref<3x256xf32, strided<[1024, 1], offset: 1536>> to memref<2x128xf32, strided<[1024, 1], offset: 2688>>
  %2 = memref.subview %1[1, 33] [1, 10] [1, 1] : memref<2x128xf32, strided<[1024, 1], offset: 2688>> to memref<1x10xf32, strided<[1024, 1], offset: 3745>>
  return %2 : memref<1x10xf32, strided<[1024, 1], offset: 3745>>
}

// -----

// CHECK-LABEL: func.func @subview_strided(
// CHECK-SAME:  %[[VAL_0:.*]]: memref<4x1024xf32>) -> memref<1x128xf32, strided<[1024, 1], offset: ?>> {
func.func @subview_strided(%input: memref<4x1024xf32>) -> memref<1x128xf32, strided<[1024, 1], offset: ?>> {
  // CHECK: %[[VAL_1:.*]] = arith.constant 3 : index
  %cst_1 = arith.constant 1 : index
  %cst_2 = arith.constant 2 : index
  // CHECK: %[[VAL_2:.*]] = memref.subview %[[VAL_0]]{{\[}}%[[VAL_1]], 384] [1, 128] [1, 1] : memref<4x1024xf32> to memref<1x128xf32, strided<[1024, 1], offset: ?>>
  %0 = memref.subview %input[%cst_2, 256] [2, 256] [1, 1] : memref<4x1024xf32> to memref<2x256xf32, strided<[1024, 1], offset: ?>>
  %1 = memref.subview %0[%cst_1, 128] [1, 128] [1, 1] : memref<2x256xf32, strided<[1024, 1], offset: ?>> to memref<1x128xf32, strided<[1024, 1], offset: ?>>
  return %1 : memref<1x128xf32, strided<[1024, 1], offset: ?>>
}

// -----

// CHECK-LABEL: func.func @subview_strided(
// CHECK-SAME:  %[[VAL_0:.*]]: memref<4x1024xf32>) -> memref<1x128xf32, strided<[1024, 1], offset: ?>> {
func.func @subview_strided(%input: memref<4x1024xf32>) -> memref<1x128xf32, strided<[1024, 1], offset: ?>> {
  // CHECK: %[[VAL_1:.*]] = arith.constant 3 : index
  %cst_2 = arith.constant 2 : index
  // CHECK: %[[VAL_2:.*]] = arith.constant 384 : index
  %cst_128 = arith.constant 128 : index
  // CHECK: %[[VAL_3:.*]] = memref.subview %[[VAL_0]]{{\[}}%[[VAL_1]], %[[VAL_2]]] [1, 128] [1, 1] : memref<4x1024xf32> to memref<1x128xf32, strided<[1024, 1], offset: ?>>
  %0 = memref.subview %input[%cst_2, 256] [2, 256] [1, 1] : memref<4x1024xf32> to memref<2x256xf32, strided<[1024, 1], offset: ?>>
  %1 = memref.subview %0[1, %cst_128] [1, 128] [1, 1] : memref<2x256xf32, strided<[1024, 1], offset: ?>> to memref<1x128xf32, strided<[1024, 1], offset: ?>>
  return %1 : memref<1x128xf32, strided<[1024, 1], offset: ?>>
}

// -----

// CHECK-LABEL: func.func @subview_strided(
// CHECK-SAME:  %[[VAL_0:.*]]: memref<8x1024xf32>) -> memref<1x64xf32, strided<[4096, 4], offset: 4480>> {
func.func @subview_strided(%input: memref<8x1024xf32>) -> memref<1x64xf32, strided<[4096, 4], offset: 4480>> {
  // CHECK:     %[[VAL_1:.*]] = memref.subview %[[VAL_0]][4, 384] [1, 64] [4, 4] : memref<8x1024xf32> to memref<1x64xf32, strided<[4096, 4], offset: 4480>>
  %0 = memref.subview %input[2, 256] [2, 256] [2, 2] : memref<8x1024xf32> to memref<2x256xf32, strided<[2048, 2], offset: 2304>>
  %1 = memref.subview %0[1, 64] [1, 64] [2, 2] : memref<2x256xf32, strided<[2048, 2], offset: 2304>> to memref<1x64xf32, strided<[4096, 4], offset: 4480>>
  return %1 : memref<1x64xf32, strided<[4096, 4], offset: 4480>>
}

// -----

// CHECK-LABEL: func.func @subview_strided(
// CHECK-SAME:  %[[VAL_0:.*]]: memref<30x30xf32>) -> memref<2x2xf32, strided<[240, 8], offset: 217>> {
func.func @subview_strided(%input: memref<30x30xf32>) -> memref<2x2xf32, strided<[240, 8], offset: 217>> {
  // CHECK:     %[[VAL_1:.*]] = memref.subview %[[VAL_0]][7, 7] [2, 2] [8, 8] : memref<30x30xf32> to memref<2x2xf32, strided<[240, 8], offset: 217>>
  %0 = memref.subview %input[1, 1] [12, 12] [2, 2] : memref<30x30xf32> to memref<12x12xf32, strided<[60, 2], offset: 31>>
  %1 = memref.subview %0[1, 1] [5, 5] [2, 2] : memref<12x12xf32, strided<[60, 2], offset: 31>> to memref<5x5xf32, strided<[120, 4], offset: 93>>
  %2 = memref.subview %1[1, 1] [2, 2] [2, 2] : memref<5x5xf32, strided<[120, 4], offset: 93>> to memref<2x2xf32, strided<[240, 8], offset: 217>>
  return %2 : memref<2x2xf32, strided<[240, 8], offset: 217>> 
}

// -----

// CHECK-LABEL: func.func @subview_strided(
// CHECK-SAME:  %[[VAL_0:.*]]: memref<4x1024xf32>) -> memref<1x64xf32, strided<[4096, 4], offset: ?>> {
func.func @subview_strided(%input: memref<4x1024xf32>) -> memref<1x64xf32, strided<[4096, 4], offset: ?>> {
  // CHECK:     %[[VAL_1:.*]] = arith.constant 4 : index
  %cst_2 = arith.constant 2 : index
  // CHECK:     %[[VAL_2:.*]] = arith.constant 384 : index
  %cst_64 = arith.constant 64 : index
  // CHECK:     %[[VAL_3:.*]] = memref.subview %[[VAL_0]]{{\[}}%[[VAL_1]], %[[VAL_2]]] [1, 64] [4, 4] : memref<4x1024xf32> to memref<1x64xf32, strided<[4096, 4], offset: ?>>
  %0 = memref.subview %input[%cst_2, 256] [2, 256] [2, 2] : memref<4x1024xf32> to memref<2x256xf32, strided<[2048, 2], offset: ?>>
  %1 = memref.subview %0[1, %cst_64] [1, 64] [2, 2] : memref<2x256xf32, strided<[2048, 2], offset: ?>> to memref<1x64xf32, strided<[4096, 4], offset: ?>>
  return %1 : memref<1x64xf32, strided<[4096, 4], offset: ?>>
}

// -----

// CHECK-LABEL: func.func @subview_strided(
// CHECK-SAME:  %[[VAL_0:.*]]: memref<4x1024xf32>) -> memref<1x64xf32, strided<[4096, 4], offset: ?>> {
func.func @subview_strided(%input: memref<4x1024xf32>) -> memref<1x64xf32, strided<[4096, 4], offset: ?>> {
  // CHECK:     %[[VAL_1:.*]] = arith.constant 4 : index
  %cst_1 = arith.constant 1 : index
  %cst_2 = arith.constant 2 : index
  // CHECK:     %[[VAL_2:.*]] = memref.subview %[[VAL_0]]{{\[}}%[[VAL_1]], 384] [1, 64] [4, 4] : memref<4x1024xf32> to memref<1x64xf32, strided<[4096, 4], offset: ?>>
  %0 = memref.subview %input[%cst_2, 256] [2, 256] [2, 2] : memref<4x1024xf32> to memref<2x256xf32, strided<[2048, 2], offset: ?>>
  %1 = memref.subview %0[%cst_1, 64] [1, 64] [2, 2] : memref<2x256xf32, strided<[2048, 2], offset: ?>> to memref<1x64xf32, strided<[4096, 4], offset: ?>>
  return %1 : memref<1x64xf32, strided<[4096, 4], offset: ?>>
}
