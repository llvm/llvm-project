// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

// CHECK-DAG: #[[$BASE_MAP1:map[0-9]*]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-DAG: #[[$SUBVIEW_MAP1:map[0-9]*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
// CHECK-DAG: #[[$SUBVIEW_MAP11:map[0-9]*]] = affine_map<() -> (4)>
// CHECK-DAG: #[[$SUBVIEW_MAP12:map[0-9]*]] = affine_map<()[s0] -> (s0)>

// CHECK-LABEL: func @memref_subview(%arg0
func.func @memref_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %0 = memref.alloc() : memref<8x16x4xf32, strided<[64, 4, 1], offset: 0>>
  // CHECK: subview %{{.*}}[%[[c0]], %[[c0]], %[[c0]]] [%{{.*}}, %{{.*}}, %{{.*}}] [%[[c1]], %[[c1]], %[[c1]]] :
  // CHECK-SAME: memref<8x16x4xf32, strided<[64, 4, 1]>>
  // CHECK-SAME: to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
  %1 = memref.subview %0[%c0, %c0, %c0][%arg0, %arg1, %arg2][%c1, %c1, %c1]
    : memref<8x16x4xf32, strided<[64, 4, 1], offset: 0>> to
      memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>

  %2 = memref.alloc()[%arg2] : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>>
  // CHECK: memref.subview %{{.*}}[%[[c1]]] [%{{.*}}] [%[[c1]]] :
  // CHECK-SAME: memref<64xf32, #[[$BASE_MAP1]]>
  // CHECK-SAME: to memref<?xf32, #[[$SUBVIEW_MAP1]]>
  %3 = memref.subview %2[%c1][%arg0][%c1]
    : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>> to
      memref<?xf32, affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>>

  %4 = memref.alloc() : memref<64x22xf32, strided<[22, 1]>>
  // CHECK: memref.subview %{{.*}}[%[[c0]], %[[c1]]] [%{{.*}}, %{{.*}}] [%[[c1]], %[[c0]]] :
  // CHECK-SAME: memref<64x22xf32, strided<[22, 1]>>
  // CHECK-SAME: to memref<?x?xf32, strided<[?, ?], offset: ?>>
  %5 = memref.subview %4[%c0, %c1][%arg0, %arg1][%c1, %c0]
    : memref<64x22xf32, strided<[22, 1], offset: 0>> to
      memref<?x?xf32, strided<[?, ?], offset: ?>>

  // CHECK: memref.subview %{{.*}}[0, 2, 0] [4, 4, 4] [1, 1, 1] :
  // CHECK-SAME: memref<8x16x4xf32, strided<[64, 4, 1]>>
  // CHECK-SAME: to memref<4x4x4xf32, strided<[64, 4, 1], offset: 8>>
  %6 = memref.subview %0[0, 2, 0][4, 4, 4][1, 1, 1]
    : memref<8x16x4xf32, strided<[64, 4, 1], offset: 0>> to
      memref<4x4x4xf32, strided<[64, 4, 1], offset: 8>>

  %7 = memref.alloc(%arg1, %arg2) : memref<?x?xf32>
  // CHECK: memref.subview {{%.*}}[0, 0] [4, 4] [1, 1] :
  // CHECK-SAME: memref<?x?xf32>
  // CHECK-SAME: to memref<4x4xf32, strided<[?, 1]>>
  %8 = memref.subview %7[0, 0][4, 4][1, 1]
    : memref<?x?xf32> to memref<4x4xf32, strided<[?, 1]>>

  %9 = memref.alloc() : memref<16x4xf32>
  // CHECK: memref.subview {{%.*}}[{{%.*}}, {{%.*}}] [4, 4] [{{%.*}}, {{%.*}}] :
  // CHECK-SAME: memref<16x4xf32>
  // CHECK-SAME: to memref<4x4xf32, strided<[?, ?], offset: ?>>
  %10 = memref.subview %9[%arg1, %arg1][4, 4][%arg2, %arg2]
    : memref<16x4xf32> to memref<4x4xf32, strided<[?, ?], offset: ?>>

  // CHECK: memref.subview {{%.*}}[{{%.*}}, {{%.*}}] [4, 4] [2, 2] :
  // CHECK-SAME: memref<16x4xf32>
  // CHECK-SAME: to memref<4x4xf32, strided<[8, 2], offset: ?>>
  %11 = memref.subview %9[%arg1, %arg2][4, 4][2, 2]
    : memref<16x4xf32> to memref<4x4xf32, strided<[8, 2], offset: ?>>

  %12 = memref.alloc() : memref<1x9x1x4x1xf32, strided<[36, 36, 4, 4, 1]>>
  // CHECK: memref.subview
  // CHECK-SAME: [1, 9, 1, 4, 1]
  // CHECK-SAME: memref<1x9x1x4x1xf32, strided<[36, 36, 4, 4, 1]>> to memref<9x4xf32, strided<[?, ?], offset: ?>>
  %13 = memref.subview %12[%arg1, %arg1, %arg1, %arg1, %arg1][1, 9, 1, 4, 1][%arg2, %arg2, %arg2, %arg2, %arg2] : memref<1x9x1x4x1xf32, strided<[36, 36, 4, 4, 1], offset: 0>> to memref<9x4xf32, strided<[?, ?], offset: ?>>
  // CHECK: memref.subview
  // CHECK-SAME: [1, 9, 1, 4, 1]
  // CHECK-SAME: memref<1x9x1x4x1xf32, strided<[36, 36, 4, 4, 1]>> to memref<1x9x4xf32, strided<[?, ?, ?], offset: ?>>
  %14 = memref.subview %12[%arg1, %arg1, %arg1, %arg1, %arg1][1, 9, 1, 4, 1][%arg2, %arg2, %arg2, %arg2, %arg2] : memref<1x9x1x4x1xf32, strided<[36, 36, 4, 4, 1], offset: 0>> to memref<1x9x4xf32, strided<[?, ?, ?], offset: ?>>

  %15 = memref.alloc(%arg1, %arg2)[%c0, %c1, %arg1, %arg0, %arg0, %arg2, %arg2] : memref<1x?x5x1x?x1xf32, strided<[?, ?, ?, ?, ?, ?], offset: ?>>
  // CHECK: memref.subview %{{.*}}[0, 0, 0, 0, 0, 0] [1, %{{.*}}, 5, 1, %{{.*}}, 1] [1, 1, 1, 1, 1, 1]  :
  // CHECK-SAME: memref<1x?x5x1x?x1xf32, strided<[?, ?, ?, ?, ?, ?], offset: ?>> to memref<?x5x?xf32, strided<[?, ?, ?], offset: ?>>
  %16 = memref.subview %15[0, 0, 0, 0, 0, 0][1, %arg1, 5, 1, %arg2, 1][1, 1, 1, 1, 1, 1] : memref<1x?x5x1x?x1xf32, strided<[?, ?, ?, ?, ?, ?], offset: ?>> to memref<?x5x?xf32, strided<[?, ?, ?], offset: ?>>
  // CHECK: memref.subview %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] [1, %{{.*}}, 5, 1, %{{.*}}, 1] [1, 1, 1, 1, 1, 1]  :
  // CHECK-SAME: memref<1x?x5x1x?x1xf32, strided<[?, ?, ?, ?, ?, ?], offset: ?>> to memref<?x5x?x1xf32, strided<[?, ?, ?, ?], offset: ?>>
  %17 = memref.subview %15[%arg1, %arg1, %arg1, %arg1, %arg1, %arg1][1, %arg1, 5, 1, %arg2, 1][1, 1, 1, 1, 1, 1] :  memref<1x?x5x1x?x1xf32, strided<[?, ?, ?, ?, ?, ?], offset: ?>> to memref<?x5x?x1xf32, strided<[?, ?, ?, ?], offset: ?>>

  %18 = memref.alloc() : memref<1x8xf32>
  // CHECK: memref.subview %{{.*}}[0, 0] [1, 8] [1, 1]  : memref<1x8xf32> to memref<8xf32>
  %19 = memref.subview %18[0, 0][1, 8][1, 1] : memref<1x8xf32> to memref<8xf32>

  %20 = memref.alloc() : memref<8x16x4xf32>
  // CHECK: memref.subview %{{.*}}[0, 0, 0] [1, 16, 4] [1, 1, 1]  : memref<8x16x4xf32> to memref<16x4xf32>
  %21 = memref.subview %20[0, 0, 0][1, 16, 4][1, 1, 1] : memref<8x16x4xf32> to memref<16x4xf32>

  %22 = memref.subview %20[3, 4, 2][1, 6, 3][1, 1, 1] : memref<8x16x4xf32> to memref<6x3xf32, strided<[4, 1], offset: 210>>

  %23 = memref.alloc() : memref<f32>
  %78 = memref.subview %23[] [] []  : memref<f32> to memref<f32>

  /// Subview with only leading operands.
  %24 = memref.alloc() : memref<5x3xf32>
  // CHECK: memref.subview %{{.*}}[2, 0] [3, 3] [1, 1] : memref<5x3xf32> to memref<3x3xf32, strided<[3, 1], offset: 6>>
  %25 = memref.subview %24[2, 0][3, 3][1, 1]: memref<5x3xf32> to memref<3x3xf32, strided<[3, 1], offset: 6>>

  /// Rank-reducing subview with only leading operands.
  // CHECK: memref.subview %{{.*}}[1, 0] [1, 3] [1, 1] : memref<5x3xf32> to memref<3xf32, strided<[1], offset: 3>>
  %26 = memref.subview %24[1, 0][1, 3][1, 1]: memref<5x3xf32> to memref<3xf32, strided<[1], offset: 3>>

  // Corner-case of 0-D rank-reducing subview with an offset.
  // CHECK: memref.subview %{{.*}}[1, 1] [1, 1] [1, 1] : memref<5x3xf32> to memref<f32, #[[$SUBVIEW_MAP11]]>
  %27 = memref.subview %24[1, 1] [1, 1] [1, 1] : memref<5x3xf32> to memref<f32, affine_map<() -> (4)>>

  // CHECK: memref.subview %{{.*}}[%{{.*}}, 1] [1, 1] [1, 1] : memref<5x3xf32> to memref<f32, #[[$SUBVIEW_MAP12]]>
  %28 = memref.subview %24[%arg0, 1] [1, 1] [1, 1] : memref<5x3xf32> to memref<f32, affine_map<()[s0] -> (s0)>>

  // CHECK: memref.subview %{{.*}}[0, %{{.*}}] [%{{.*}}, 1] [1, 1] : memref<?x?xf32> to memref<?xf32, #[[$SUBVIEW_MAP1]]>
  %a30 = memref.alloc(%arg0, %arg0) : memref<?x?xf32>
  %30 = memref.subview %a30[0, %arg1][%arg2, 1][1, 1] : memref<?x?xf32> to memref<?xf32, affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>>

  %c8 = arith.constant 8 : index
  %a40 = memref.alloc() : memref<16x16xf32>
  // CHECK: memref.subview
  %40 = memref.subview %a40[%c8, 8][8, 8][1, 1]  :
    memref<16x16xf32> to memref<8x8xf32, affine_map<(d0, d1)[s0] -> (d0 * 16 + d1 + s0)>>

  return
}

