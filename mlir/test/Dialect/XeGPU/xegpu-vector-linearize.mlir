// RUN: mlir-opt %s -split-input-file -xegpu-vector-linearize | FileCheck %s

// CHECK-LABEL: @test_linearize
//  CHECK-SAME: (%[[ARG0:.*]]: vector<2x2xf32>) -> vector<2x2xf32> {
//       CHECK: %[[T0:.*]] = vector.shape_cast %[[ARG0]] : vector<2x2xf32> to vector<4xf32>
//       CHECK: %[[CST:.*]] = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : vector<4xf32>
//       CHECK: %[[T1:.*]] = math.sin %[[T0]] : vector<4xf32>
//       CHECK: %[[T2:.*]] = arith.addf %[[T0]], %[[CST]] : vector<4xf32>
//       CHECK: %[[T3:.*]] = arith.addf %[[T2]], %[[T1]] : vector<4xf32>
//       CHECK: %[[T4:.*]] = vector.shape_cast %[[T3]] : vector<4xf32> to vector<2x2xf32>
//       CHECK: return %[[T4]] : vector<2x2xf32>
func.func @test_linearize(%arg0: vector<2x2xf32>) -> vector<2x2xf32> {
  %0 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : vector<2x2xf32>
// Arith and math ops are handled in generic way, check some of them
  %1 = math.sin %arg0 : vector<2x2xf32>
  %2 = arith.addf %arg0, %0 :  vector<2x2xf32>
  %3 = arith.addf %2, %1 :  vector<2x2xf32>
  return %3 : vector<2x2xf32>
}

// -----

// CHECK-LABEL: test_const_novector
//       CHECK:  %[[R:.*]] = arith.constant 42 : i32
//       CHECK:  return %[[R]] : i32
func.func @test_const_novector() -> i32 {
  %0 = arith.constant 42 : i32
  return %0 : i32
}

// -----
// CHECK-LABEL: test_create_mask
//       CHECK: vector.create_mask {{.*}} : vector<16xi1>
func.func @test_create_mask() -> vector<1x16xi1> {
  %c0 = arith.constant 0 : index
  %c20 = arith.constant 20 : index
  %0 = vector.create_mask %c0, %c20 : vector<1x16xi1>
  return %0 : vector<1x16xi1>
}

// -----
// CHECK-LABEL: test_extract_strided_slice
//  CHECK-SAME: (%[[ORIG_ARG:.*]]: vector<8x16xf32>) -> vector<8x8xf32>
//       CHECK: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<8x16xf32> to vector<128xf32>
//       CHECK: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG]], %[[ARG]]
//       CHECK: [8, 9, 10, 11, 12, 13, 14, 15,
//       CHECK: 24, 25, 26, 27, 28, 29, 30, 31,
//       CHECK: 40, 41, 42, 43, 44, 45, 46, 47,
//       CHECK: 56, 57, 58, 59, 60, 61, 62, 63,
//       CHECK: 72, 73, 74, 75, 76, 77, 78, 79,
//       CHECK: 88, 89, 90, 91, 92, 93, 94, 95,
//       CHECK: 104, 105, 106, 107, 108, 109, 110, 111,
//       CHECK: 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf32>, vector<128xf32>
//       CHECK: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<64xf32> to vector<8x8xf32>
//       CHECK: return %[[RES]] : vector<8x8xf32>
func.func @test_extract_strided_slice_1(%arg0 : vector<8x16xf32>) -> vector<8x8xf32> {
  %0 = vector.extract_strided_slice %arg0 { sizes = [8, 8], strides = [1, 1], offsets = [0, 8]}
     : vector<8x16xf32> to vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// -----
// CHECK-LABEL: test_extract_strided_slice_2
//  CHECK-SAME: (%[[ORIG_ARG:.*]]: vector<2x32x8xf32>) -> vector<1x8x8xf32>
//       CHECK: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<2x32x8xf32> to vector<512xf32>
//       CHECK: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG]], %[[ARG]]
//       CHECK: [448, 449, 450, 451, 452, 453, 454, 455,
//       CHECK: 456, 457, 458, 459, 460, 461, 462, 463,
//       CHECK: 464, 465, 466, 467, 468, 469, 470, 471,
//       CHECK: 472, 473, 474, 475, 476, 477, 478, 479,
//       CHECK: 480, 481, 482, 483, 484, 485, 486, 487,
//       CHECK: 488, 489, 490, 491, 492, 493, 494, 495,
//       CHECK: 496, 497, 498, 499, 500, 501, 502, 503,
//       CHECK: 504, 505, 506, 507, 508, 509, 510, 511] : vector<512xf32>, vector<512xf32>
//       CHECK: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<64xf32> to vector<1x8x8xf32>
//       CHECK: return %[[RES]] : vector<1x8x8xf32>
func.func @test_extract_strided_slice_2(%arg0 : vector<2x32x8xf32>) -> vector<1x8x8xf32> {
  %0 = vector.extract_strided_slice %arg0 { offsets = [1, 24], strides = [1, 1], sizes = [1, 8] }
    : vector<2x32x8xf32> to vector<1x8x8xf32>
  return %0 : vector<1x8x8xf32>
}

// -----
// CHECK-LABEL: test_vector_shuffle
//  CHECK-SAME: (%[[ORIG_ARG1:.*]]: vector<4x4xf32>, %[[ORIG_ARG2:.*]]: vector<4x4xf32>) -> vector<8x4xf32> {
//       CHECK: %[[ARG2:.*]] = vector.shape_cast %[[ORIG_ARG2]] : vector<4x4xf32> to vector<16xf32>
//       CHECK: %[[ARG1:.*]] = vector.shape_cast %[[ORIG_ARG1]] : vector<4x4xf32> to vector<16xf32>
//       CHECK: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG1]], %[[ARG2]]
//       CHECK: [0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23,
//       CHECK: 8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
//       CHECK: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<32xf32> to vector<8x4xf32>
//       CHECK: return %[[RES]] : vector<8x4xf32>
func.func @test_vector_shuffle(%arg0: vector<4x4xf32>, %arg1: vector<4x4xf32>) -> vector<8x4xf32> {
  %0 = vector.shuffle %arg0, %arg1 [0, 4, 1, 5, 2, 6, 3, 7] : vector<4x4xf32>, vector<4x4xf32>
  return %0 : vector<8x4xf32>
}

// -----
// CHECK-LABEL: test_vector_extract
// CHECK-SAME: (%[[ORIG_ARG:.*]]: vector<2x8x4xf32>) -> vector<8x4xf32>
// CHECK: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<2x8x4xf32> to vector<64xf32>
// CHECK: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG]], %[[ARG]]
// CHECK: [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
// CHECK: 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<64xf32>, vector<64xf32>
// CHECK: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<32xf32> to vector<8x4xf32>
// CHECK: return %[[RES]] : vector<8x4xf32>
func.func @test_vector_extract(%arg0: vector<2x8x4xf32>) -> vector<8x4xf32> {
  %0 = vector.extract %arg0[1]: vector<8x4xf32> from vector<2x8x4xf32>
  return %0 : vector<8x4xf32>
}

// -----
// CHECK-LABEL: test_vector_insert
// CHECK-SAME: (%[[DEST:.*]]: vector<2x8x4xf32>, %[[SRC:.*]]: vector<8x4xf32>) -> vector<2x8x4xf32>
// CHECK: %[[ARG_DEST:.*]] = vector.shape_cast %[[DEST]] : vector<2x8x4xf32> to vector<64xf32>
// CHECK: %[[ARG_SRC:.*]] = vector.shape_cast %[[SRC]] : vector<8x4xf32> to vector<32xf32>
// CHECK: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG_DEST]], %[[ARG_SRC]]
// CHECK: [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
// CHECK-SAME: 88, 89, 90, 91, 92, 93, 94, 95, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
// CHECK-SAME: 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<64xf32>, vector<32xf32>
// CHECK: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<64xf32> to vector<2x8x4xf32>
// CHECK: return %[[RES]] : vector<2x8x4xf32>
func.func @test_vector_insert(%arg0: vector<2x8x4xf32>, %arg1: vector<8x4xf32>) -> vector<2x8x4xf32> {
  %0 = vector.insert %arg1, %arg0[0]: vector<8x4xf32> into vector<2x8x4xf32>
  return %0 : vector<2x8x4xf32>
}

// -----
// CHECK-LABEL: test_vector_insert_2d_idx
// CHECK-SAME: (%[[DEST:.*]]: vector<2x8x4xf32>, %[[SRC:.*]]: vector<4xf32>) -> vector<2x8x4xf32>
// CHECK: %[[ARG_DEST:.*]] = vector.shape_cast %[[DEST]] : vector<2x8x4xf32> to vector<64xf32>
// CHECK: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG_DEST]], %[[SRC]]
// CHECK: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 64, 65, 66, 67, 16, 17, 18, 19, 20, 21,
// CHECK-SAME: 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
// CHECK-SAME: 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<64xf32>, vector<4xf32>
// CHECK: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<64xf32> to vector<2x8x4xf32>
// CHECK: return %[[RES]] : vector<2x8x4xf32>
func.func @test_vector_insert_2d_idx(%arg0: vector<2x8x4xf32>, %arg1: vector<4xf32>) -> vector<2x8x4xf32> {
  %0 = vector.insert %arg1, %arg0[0, 3]: vector<4xf32> into vector<2x8x4xf32>
  return %0 : vector<2x8x4xf32>
}

// -----
// CHECK-LABEL: test_vector_transpose
// CHECK-SAME: (%[[ORIG_ARG:.*]]: vector<2x8xf32>) -> vector<8x2xf32>
// CHECK: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<2x8xf32> to vector<16xf32>
// CHECK: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG]], %[[ARG]]
// CHECK: [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<16xf32>, vector<16xf32>
// CHECK: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<16xf32> to vector<8x2xf32>
// CHECK: return %[[RES]] : vector<8x2xf32>
func.func @test_vector_transpose(%arg: vector<2x8xf32>) -> vector<8x2xf32> {
  %0 = vector.transpose %arg, [1, 0] : vector<2x8xf32> to vector<8x2xf32>
  return %0 : vector<8x2xf32>
}

// -----
// CHECK-LABEL: test_vector_transpose_16x16
// CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
func.func @test_vector_transpose_16x16(%arg: vector<16x16xf32>) -> vector<16x16xf32> {
  %0 = vector.transpose %arg, [1, 0] : vector<16x16xf32> to vector<16x16xf32>
  return %0 : vector<16x16xf32>
}

// -----
// CHECK-LABEL: func.func @test_vector_store_load_4x4
// CHECK-SAME: (%[[MEMREF:.*]]: memref<4x4xf32>)
// CHECK: %[[C3:.*]] = arith.constant 3 : index
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[V0:.*]] = vector.load %[[MEMREF]][%[[C0]], %[[C0]]] : memref<4x4xf32>, vector<4xf32>
// CHECK: %[[V1:.*]] = vector.load %[[MEMREF]][%[[C1]], %[[C0]]] : memref<4x4xf32>, vector<4xf32>
// CHECK: %[[V2:.*]] = vector.load %[[MEMREF]][%[[C2]], %[[C0]]] : memref<4x4xf32>, vector<4xf32>
// CHECK: %[[V3:.*]] = vector.load %[[MEMREF]][%[[C3]], %[[C0]]] : memref<4x4xf32>, vector<4xf32>
// CHECK: vector.store %[[V0]], %[[MEMREF]][%[[C0]], %[[C0]]] : memref<4x4xf32>, vector<4xf32>
// CHECK: vector.store %[[V1]], %[[MEMREF]][%[[C1]], %[[C0]]] : memref<4x4xf32>, vector<4xf32>
// CHECK: vector.store %[[V2]], %[[MEMREF]][%[[C2]], %[[C0]]] : memref<4x4xf32>, vector<4xf32>
// CHECK: vector.store %[[V3]], %[[MEMREF]][%[[C3]], %[[C0]]] : memref<4x4xf32>, vector<4xf32>
func.func @test_vector_store_load_4x4(%buffer: memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %0 = vector.load %buffer[%c0, %c0] : memref<4x4xf32>, vector<4x4xf32>
  vector.store %0, %buffer[%c0, %c0] : memref<4x4xf32>, vector<4x4xf32>
  return
}

// -----

func.func @test_vector_store_load_4x4_f16(%buffer: memref<4x4xf16>) {
  %c0 = arith.constant 0 : index
  %0 = vector.load %buffer[%c0, %c0] : memref<4x4xf16>, vector<4x4xf16>
  vector.store %0, %buffer[%c0, %c0] : memref<4x4xf16>, vector<4x4xf16>
  return
}
// CHECK-LABEL: func.func @test_vector_store_load_4x4_f16
// CHECK-SAME: (%[[MEMREF:.*]]: memref<4x4xf16>)
// CHECK: %[[C3:.*]] = arith.constant 3 : index
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[V0:.*]] = vector.load %[[MEMREF]][%[[C0]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
// CHECK: %[[V1:.*]] = vector.load %[[MEMREF]][%[[C1]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
// CHECK: %[[V2:.*]] = vector.load %[[MEMREF]][%[[C2]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
// CHECK: %[[V3:.*]] = vector.load %[[MEMREF]][%[[C3]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
// CHECK: vector.store %[[V0]], %[[MEMREF]][%[[C0]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
// CHECK: vector.store %[[V1]], %[[MEMREF]][%[[C1]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
// CHECK: vector.store %[[V2]], %[[MEMREF]][%[[C2]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
// CHECK: vector.store %[[V3]], %[[MEMREF]][%[[C3]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>

// -----
// CHECK-LABEL: @test_linearize_index
//  CHECK-SAME: (%[[ARG0:.*]]: vector<2x2xindex>, %[[ARG1:.*]]: vector<2x2xi32>) -> vector<2x2xindex> {
//       CHECK: %[[T0:.*]] = vector.shape_cast %[[ARG1]] : vector<2x2xi32> to vector<4xi32>
//       CHECK: %[[T1:.*]] = vector.shape_cast %[[ARG0]] : vector<2x2xindex> to vector<4xindex>
//       CHECK: %[[CST:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
//       CHECK: %[[T2:.*]] = arith.addi %[[T1]], %[[CST]] : vector<4xindex>
//       CHECK: %[[T3:.*]] = arith.index_cast %[[T2]] : vector<4xindex> to vector<4xi32>
//       CHECK: %[[T4:.*]] = arith.muli %[[T3]], %[[T0]] : vector<4xi32>
//       CHECK: %[[T5:.*]] = arith.index_cast %[[T4]] : vector<4xi32> to vector<4xindex>
//       CHECK: %[[T6:.*]] = vector.shape_cast %[[T5]] : vector<4xindex> to vector<2x2xindex>
//       CHECK: return %[[T6]] : vector<2x2xindex>
func.func @test_linearize_index(%arg0: vector<2x2xindex>, %arg1: vector<2x2xi32>) -> vector<2x2xindex> {
  %0 = arith.constant dense<[[0, 1], [2, 3]]> : vector<2x2xindex>
// Arith and math ops are handled in generic way, check some of them
  %1 = arith.addi %arg0, %0 :  vector<2x2xindex>
  %2 = arith.index_cast %1 : vector<2x2xindex> to vector<2x2xi32>
  %3 = arith.muli %2, %arg1 : vector<2x2xi32>
  %4 = arith.index_cast %3 : vector<2x2xi32> to vector<2x2xindex>
  return %4 : vector<2x2xindex>
}

// -----
// CHECK-LABEL: @add_kernel_f32
//       CHECK: %[[CST0:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xindex>
//       CHECK: %[[CST1:.*]] = arith.constant dense<[16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]> : vector<16xindex>
//       CHECK: %[[T0:.*]] = vector.splat %{{.*}} : vector<16xindex>
//       CHECK: %[[T1:.*]] = arith.addi %[[T0]], %[[CST0]] : vector<16xindex>
//       CHECK: %[[T2:.*]] = arith.addi %[[T0]], %[[CST1]] : vector<16xindex>
//       CHECK: %[[T3:.*]] = arith.index_cast %[[T1]] : vector<16xindex> to vector<16xi32>
//       CHECK: %[[T4:.*]] = arith.index_cast %[[T2]] : vector<16xindex> to vector<16xi32>
//       CHECK: %[[T5:.*]] = vector.splat %{{.*}} : vector<16xi32>
//       CHECK: %[[T6:.*]] = arith.addi %[[T5]], %[[T3]] : vector<16xi32>
//       CHECK: %[[T7:.*]] = arith.addi %[[T5]], %[[T4]] : vector<16xi32>
//       CHECK: %[[T8:.*]] = arith.index_cast %[[T6]] : vector<16xi32> to vector<16xindex>
//       CHECK: %[[T9:.*]] = arith.index_cast %[[T7]] : vector<16xi32> to vector<16xindex>
gpu.module @add_kernel_f32 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Bfloat16ConversionINTEL, BFloat16TypeKHR, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, VectorAnyINTEL, VectorComputeINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_bfloat16, SPV_KHR_expect_assume, SPV_INTEL_bfloat16_conversion, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
  gpu.func @add_kernel_f32(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>) kernel attributes {VectorComputeFunctionINTEL, known_block_size = array<i32: 1, 32, 1>, known_grid_size = array<i32: 1, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %cst = arith.constant dense<true> : vector<16xi1>
    %c32 = arith.constant 32 : index
    %c1024_i32 = arith.constant 1024 : i32
    %cst_0 = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]> : vector<1x16xindex>
    %cst_1 = arith.constant dense<[[16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]> : vector<1x16xindex>
    %thread_id_x = gpu.thread_id  x
    %thread_id_y = gpu.thread_id  y
    %block_dim_y = gpu.block_dim  y
    %0 = arith.muli %thread_id_x, %block_dim_y : index
    %1 = arith.addi %0, %thread_id_y : index
    %cast = memref.cast %arg0 : memref<*xf32> to memref<?xf32>
    %cast_2 = memref.cast %arg1 : memref<*xf32> to memref<?xf32>
    %cast_3 = memref.cast %arg2 : memref<*xf32> to memref<?xf32>
    %2 = arith.remsi %1, %c32 : index
    %3 = arith.muli %2, %c32 : index
    %4 = vector.splat %3 : vector<1x16xindex>
    %5 = arith.addi %4, %cst_0 : vector<1x16xindex>
    %6 = arith.addi %4, %cst_1 : vector<1x16xindex>
    %7 = arith.index_cast %5 : vector<1x16xindex> to vector<1x16xi32>
    %8 = arith.index_cast %6 : vector<1x16xindex> to vector<1x16xi32>
    %block_id_x = gpu.block_id  x
    %9 = arith.index_cast %block_id_x : index to i32
    %10 = arith.muli %9, %c1024_i32 : i32
    %11 = vector.splat %10 : vector<1x16xi32>
    %12 = arith.addi %11, %7 : vector<1x16xi32>
    %13 = arith.addi %11, %8 : vector<1x16xi32>
    %14 = arith.index_cast %12 : vector<1x16xi32> to vector<1x16xindex>
    %15 = arith.index_cast %13 : vector<1x16xi32> to vector<1x16xindex>
    %16 = vector.shape_cast %14 : vector<1x16xindex> to vector<16xindex>
    %17 = xegpu.create_tdesc %cast, %16 : memref<?xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>
    %18 = vector.shape_cast %15 : vector<1x16xindex> to vector<16xindex>
    %19 = xegpu.create_tdesc %cast, %18 : memref<?xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>
    %20 = xegpu.load %17, %cst <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<16xi1> -> vector<16xf32>
    %21 = vector.shape_cast %20 : vector<16xf32> to vector<1x16xf32>
    %22 = xegpu.load %19, %cst <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<16xi1> -> vector<16xf32>
    %23 = vector.shape_cast %22 : vector<16xf32> to vector<1x16xf32>
    %24 = xegpu.create_tdesc %cast_2, %16 : memref<?xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>
    %25 = xegpu.create_tdesc %cast_2, %18 : memref<?xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>
    %26 = xegpu.load %24, %cst <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<16xi1> -> vector<16xf32>
    %27 = vector.shape_cast %26 : vector<16xf32> to vector<1x16xf32>
    %28 = xegpu.load %25, %cst <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<16xi1> -> vector<16xf32>
    %29 = vector.shape_cast %28 : vector<16xf32> to vector<1x16xf32>
    %30 = arith.addf %21, %27 : vector<1x16xf32>
    %31 = arith.addf %23, %29 : vector<1x16xf32>
    %32 = xegpu.create_tdesc %cast_3, %16 : memref<?xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>
    %33 = xegpu.create_tdesc %cast_3, %18 : memref<?xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>
    %34 = vector.shape_cast %30 : vector<1x16xf32> to vector<16xf32>
    xegpu.store %34, %32, %cst <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<16xi1>
    %35 = vector.shape_cast %31 : vector<1x16xf32> to vector<16xf32>
    xegpu.store %35, %33, %cst <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<16xi1>
    gpu.return
  }
}
