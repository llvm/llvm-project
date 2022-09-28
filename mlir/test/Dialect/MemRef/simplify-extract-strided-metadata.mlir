// RUN: mlir-opt --simplify-extract-strided-metadata -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: func @extract_strided_metadata_constants
//  CHECK-SAME: (%[[ARG:.*]]: memref<5x4xf32, strided<[4, 1], offset: 2>>)
func.func @extract_strided_metadata_constants(%base: memref<5x4xf32, strided<[4, 1], offset: 2>>)
    -> (memref<f32>, index, index, index, index, index) {
  //   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  //   CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  //   CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
  //   CHECK-DAG: %[[C5:.*]] = arith.constant 5 : index
  
  //       CHECK: %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:2, %[[STRIDES:.*]]:2 = memref.extract_strided_metadata %[[ARG]]
  %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %base :
    memref<5x4xf32, strided<[4,1], offset:2>>
    -> memref<f32>, index, index, index, index, index

  // CHECK: %[[BASE]], %[[C2]], %[[C5]], %[[C4]], %[[C4]], %[[C1]]
  return %base_buffer, %offset, %sizes#0, %sizes#1, %strides#0, %strides#1 :
    memref<f32>, index, index, index, index, index
}

// -----

// Check that we simplify extract_strided_metadata of subview to
// base_buf, base_offset, base_sizes, base_strides = extract_strided_metadata
// strides = base_stride_i * subview_stride_i
// offset = base_offset + sum(subview_offsets_i * strides_i).
//
// This test also checks that we don't create useless arith operations
// when subview_offsets_i is 0.
//
// CHECK-LABEL: func @extract_strided_metadata_of_subview
//  CHECK-SAME: (%[[ARG:.*]]: memref<5x4xf32>)
//
// Materialize the offset for dimension 1.
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
//
// Plain extract_strided_metadata.
//   CHECK-DAG: %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:2, %[[STRIDES:.*]]:2 = memref.extract_strided_metadata %[[ARG]]
//
// Final offset is:
//   origOffset + (== 0)
//   base_stride0 * subview_stride0 * subview_offset0 + (== 4 * 1 * 0 == 0)
//   base_stride1 * subview_stride1 * subview_offset1 (== 1 * 1 * 2)
//  == 2
//
// Return the new tuple.
//       CHECK: return %[[BASE]], %[[C2]], %[[C2]], %[[C2]], %[[C4]], %[[C1]]
func.func @extract_strided_metadata_of_subview(%base: memref<5x4xf32>)
    -> (memref<f32>, index, index, index, index, index) {

  %subview = memref.subview %base[0, 2][2, 2][1, 1] :
    memref<5x4xf32> to memref<2x2xf32, strided<[4, 1], offset: 2>>

  %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %subview :
    memref<2x2xf32, strided<[4,1], offset:2>>
    -> memref<f32>, index, index, index, index, index

  return %base_buffer, %offset, %sizes#0, %sizes#1, %strides#0, %strides#1 :
    memref<f32>, index, index, index, index, index
}

// -----

// Check that we simplify extract_strided_metadata of subview properly
// when dynamic sizes are involved.
// See extract_strided_metadata_of_subview for an explanation of the actual
// expansion.
// Orig strides: [64, 4, 1]
// Sub strides: [1, 1, 1]
// => New strides: [64, 4, 1]
//
// Orig offset: 0
// Sub offsets: [3, 4, 2]
// => Final offset: 3 * 64 + 4 * 4 + 2 * 1 + 0 == 210
//
// Final sizes == subview sizes == [%size, 6, 3]
//
// CHECK-LABEL: func @extract_strided_metadata_of_subview_with_dynamic_size
//  CHECK-SAME: (%[[ARG:.*]]: memref<8x16x4xf32>,
//  CHECK-SAME: %[[DYN_SIZE:.*]]: index)
//
//   CHECK-DAG: %[[C210:.*]] = arith.constant 210 : index
//   CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
//   CHECK-DAG: %[[C6:.*]] = arith.constant 6 : index
//   CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
//   CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//
//   CHECK-DAG: %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:3, %[[STRIDES:.*]]:3 = memref.extract_strided_metadata %[[ARG]]
//
//       CHECK: return %[[BASE]], %[[C210]], %[[DYN_SIZE]], %[[C6]], %[[C3]], %[[C64]], %[[C4]], %[[C1]]
func.func @extract_strided_metadata_of_subview_with_dynamic_size(
    %base: memref<8x16x4xf32>, %size: index)
    -> (memref<f32>, index, index, index, index, index, index, index) {

  %subview = memref.subview %base[3, 4, 2][%size, 6, 3][1, 1, 1] :
    memref<8x16x4xf32> to memref<?x6x3xf32, strided<[64, 4, 1], offset: 210>>

  %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %subview :
    memref<?x6x3xf32, strided<[64,4,1], offset: 210>>
    -> memref<f32>, index, index, index, index, index, index, index

  return %base_buffer, %offset, %sizes#0, %sizes#1, %sizes#2, %strides#0, %strides#1, %strides#2 :
    memref<f32>, index, index, index, index, index, index, index
}

// -----

// Check that we simplify extract_strided_metadata of subview properly
// when the subview reduces the ranks.
// In particular the returned strides must come from #1 and #2 of the %strides
// value of the new extract_strided_metadata_of_subview, not #0 and #1.
// See extract_strided_metadata_of_subview for an explanation of the actual
// expansion.
//
// Orig strides: [64, 4, 1]
// Sub strides: [1, 1, 1]
// => New strides: [64, 4, 1]
// Final strides == filterOutReducedDim(new strides, 0) == [4 , 1]
//
// Orig offset: 0
// Sub offsets: [3, 4, 2]
// => Final offset: 3 * 64 + 4 * 4 + 2 * 1 + 0 == 210
//
// Final sizes == filterOutReducedDim(subview sizes, 0) == [6, 3]
//
// CHECK-LABEL: func @extract_strided_metadata_of_rank_reduced_subview
//  CHECK-SAME: (%[[ARG:.*]]: memref<8x16x4xf32>)
//
//   CHECK-DAG: %[[C210:.*]] = arith.constant 210 : index
//   CHECK-DAG: %[[C6:.*]] = arith.constant 6 : index
//   CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
//   CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//
//   CHECK-DAG: %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:3, %[[STRIDES:.*]]:3 = memref.extract_strided_metadata %[[ARG]]
//
//       CHECK: return %[[BASE]], %[[C210]], %[[C6]], %[[C3]], %[[C4]], %[[C1]]
func.func @extract_strided_metadata_of_rank_reduced_subview(%base: memref<8x16x4xf32>)
    -> (memref<f32>, index, index, index, index, index) {

  %subview = memref.subview %base[3, 4, 2][1, 6, 3][1, 1, 1] :
    memref<8x16x4xf32> to memref<6x3xf32, strided<[4, 1], offset: 210>>

  %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %subview :
    memref<6x3xf32, strided<[4,1], offset: 210>>
    -> memref<f32>, index, index, index, index, index

  return %base_buffer, %offset, %sizes#0, %sizes#1, %strides#0, %strides#1 :
    memref<f32>, index, index, index, index, index
}

// -----

// Check that we simplify extract_strided_metadata of subview properly
// when the subview reduces the rank and some of the strides are variable.
// In particular, we check that:
// A. The dynamic stride is multiplied with the base stride to create the new
//    stride for dimension 1.
// B. The first returned stride is the value computed in #A.
// See extract_strided_metadata_of_subview for an explanation of the actual
// expansion.
//
// Orig strides: [64, 4, 1]
// Sub strides: [1, %stride, 1]
// => New strides: [64, 4 * %stride, 1]
// Final strides == filterOutReducedDim(new strides, 0) == [4 * %stride , 1]
//
// Orig offset: 0
// Sub offsets: [3, 4, 2]
// => Final offset: 3 * 64 + 4 * 4 * %stride + 2 * 1 + 0 == 16 * %stride + 194
//
//   CHECK-DAG: #[[$STRIDE1_MAP:.*]] = affine_map<()[s0] -> (s0 * 4)>
//   CHECK-DAG: #[[$OFFSET_MAP:.*]] = affine_map<()[s0] -> (s0 * 16 + 194)>
// CHECK-LABEL: func @extract_strided_metadata_of_rank_reduced_subview_w_variable_strides
//  CHECK-SAME: (%[[ARG:.*]]: memref<8x16x4xf32>,
//  CHECK-SAME: %[[DYN_STRIDE:.*]]: index)
//
//   CHECK-DAG: %[[C6:.*]] = arith.constant 6 : index
//   CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//
//   CHECK-DAG: %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:3, %[[STRIDES:.*]]:3 = memref.extract_strided_metadata %[[ARG]]
//
//   CHECK-DAG: %[[DIM1_STRIDE:.*]] = affine.apply #[[$STRIDE1_MAP]]()[%[[DYN_STRIDE]]]
//   CHECK-DAG: %[[FINAL_OFFSET:.*]] = affine.apply #[[$OFFSET_MAP]]()[%[[DYN_STRIDE]]]
//
//       CHECK: return %[[BASE]], %[[FINAL_OFFSET]], %[[C6]], %[[C3]], %[[DIM1_STRIDE]], %[[C1]]
func.func @extract_strided_metadata_of_rank_reduced_subview_w_variable_strides(
    %base: memref<8x16x4xf32>, %stride: index)
    -> (memref<f32>, index, index, index, index, index) {

  %subview = memref.subview %base[3, 4, 2][1, 6, 3][1, %stride, 1] :
    memref<8x16x4xf32> to memref<6x3xf32, strided<[4, 1], offset: 210>>

  %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %subview :
    memref<6x3xf32, strided<[4, 1], offset: 210>>
    -> memref<f32>, index, index, index, index, index

  return %base_buffer, %offset, %sizes#0, %sizes#1, %strides#0, %strides#1 :
    memref<f32>, index, index, index, index, index
}

// -----

// Check that we simplify extract_strided_metadata of subview properly
// when the subview uses variable offsets.
// See extract_strided_metadata_of_subview for an explanation of the actual
// expansion.
//
// Orig strides: [128, 1]
// Sub strides: [1, 1]
// => New strides: [128, 1]
//
// Orig offset: 0
// Sub offsets: [%arg1, %arg2]
// => Final offset: 128 * arg1 + 1 * %arg2 + 0
//
//   CHECK-DAG: #[[$OFFSETS_MAP:.*]] = affine_map<()[s0, s1] -> (s0 * 128 + s1)>
// CHECK-LABEL: func @extract_strided_metadata_of_subview_w_variable_offset
//  CHECK-SAME: (%[[ARG:.*]]: memref<384x128xf32>,
//  CHECK-SAME: %[[DYN_OFFSET0:.*]]: index,
//  CHECK-SAME: %[[DYN_OFFSET1:.*]]: index)
//
//   CHECK-DAG: %[[C128:.*]] = arith.constant 128 : index
//   CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:2, %[[STRIDES:.*]]:2 = memref.extract_strided_metadata %[[ARG]]
//
//   CHECK-DAG: %[[FINAL_OFFSET:.*]] = affine.apply #[[$OFFSETS_MAP]]()[%[[DYN_OFFSET0]], %[[DYN_OFFSET1]]]
//
//       CHECK: return %[[BASE]], %[[FINAL_OFFSET]], %[[C64]], %[[C64]], %[[C128]], %[[C1]]
func.func @extract_strided_metadata_of_subview_w_variable_offset(
    %arg0: memref<384x128xf32>, %arg1 : index, %arg2 : index)
    -> (memref<f32>, index, index, index, index, index) {

  %subview = memref.subview %arg0[%arg1, %arg2] [64, 64] [1, 1] :
    memref<384x128xf32> to memref<64x64xf32, strided<[128, 1], offset: ?>>

  %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %subview :
  memref<64x64xf32, strided<[128, 1], offset: ?>> -> memref<f32>, index, index, index, index, index

  return %base_buffer, %offset, %sizes#0, %sizes#1, %strides#0, %strides#1 :
    memref<f32>, index, index, index, index, index
}

// -----

// Check that all the math is correct for all types of computations.
// We achieve that by using dynamic values for all the different types:
// - Offsets
// - Sizes
// - Strides
//
// Orig strides: [s0, s1, s2]
// Sub strides: [subS0, subS1, subS2]
// => New strides: [s0 * subS0, s1 * subS1, s2 * subS2]
// ==> 1 affine map (used for each stride) with two values.
//
// Orig offset: origOff
// Sub offsets: [subO0, subO1, subO2]
// => Final offset: s0 * subS0 * subO0 + ... + s2 * subS2 * subO2 + origOff
// ==> 1 affine map with (rank * 3 + 1) symbols
//
// CHECK-DAG: #[[$STRIDE_MAP:.*]] = affine_map<()[s0, s1] -> (s0 * s1)>
// CHECK-DAG: #[[$OFFSET_MAP:.*]] = affine_map<()[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (s0 + (s1 * s2) * s3 + (s4 * s5) * s6 + (s7 * s8) * s9)>
// CHECK-LABEL: func @extract_strided_metadata_of_subview_all_dynamic
//  CHECK-SAME: (%[[ARG:.*]]: memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>, %[[DYN_OFFSET0:.*]]: index, %[[DYN_OFFSET1:.*]]: index, %[[DYN_OFFSET2:.*]]: index, %[[DYN_SIZE0:.*]]: index, %[[DYN_SIZE1:.*]]: index, %[[DYN_SIZE2:.*]]: index, %[[DYN_STRIDE0:.*]]: index, %[[DYN_STRIDE1:.*]]: index, %[[DYN_STRIDE2:.*]]: index)
//
//   CHECK-DAG: %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:3, %[[STRIDES:.*]]:3 = memref.extract_strided_metadata %[[ARG]]
//
//  CHECK-DAG: %[[FINAL_STRIDE0:.*]] = affine.apply #[[$STRIDE_MAP]]()[%[[DYN_STRIDE0]], %[[STRIDES]]#0]
//  CHECK-DAG: %[[FINAL_STRIDE1:.*]] = affine.apply #[[$STRIDE_MAP]]()[%[[DYN_STRIDE1]], %[[STRIDES]]#1]
//  CHECK-DAG: %[[FINAL_STRIDE2:.*]] = affine.apply #[[$STRIDE_MAP]]()[%[[DYN_STRIDE2]], %[[STRIDES]]#2]
//
//  CHECK-DAG: %[[FINAL_OFFSET:.*]] = affine.apply #[[$OFFSET_MAP]]()[%[[OFFSET]], %[[DYN_OFFSET0]], %[[DYN_STRIDE0]], %[[STRIDES]]#0, %[[DYN_OFFSET1]], %[[DYN_STRIDE1]], %[[STRIDES]]#1, %[[DYN_OFFSET2]], %[[DYN_STRIDE2]], %[[STRIDES]]#2]
//
//       CHECK: return %[[BASE]], %[[FINAL_OFFSET]], %[[DYN_SIZE0]], %[[DYN_SIZE1]], %[[DYN_SIZE2]], %[[FINAL_STRIDE0]], %[[FINAL_STRIDE1]], %[[FINAL_STRIDE2]]
func.func @extract_strided_metadata_of_subview_all_dynamic(
    %base: memref<?x?x?xf32, strided<[?,?,?], offset:?>>,
    %offset0: index, %offset1: index, %offset2: index,
    %size0: index, %size1: index, %size2: index,
    %stride0: index, %stride1: index, %stride2: index)
    -> (memref<f32>, index, index, index, index, index, index, index) {

  %subview = memref.subview %base[%offset0, %offset1, %offset2]
                                 [%size0, %size1, %size2]
                                 [%stride0, %stride1, %stride2] :
    memref<?x?x?xf32, strided<[?,?,?], offset: ?>> to
      memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>

  %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %subview :
    memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
    -> memref<f32>, index, index, index, index, index, index, index

  return %base_buffer, %offset, %sizes#0, %sizes#1, %sizes#2, %strides#0, %strides#1, %strides#2 :
    memref<f32>, index, index, index, index, index, index, index
}

// -----

// Check that we properly simplify extract_strided_metadata of expand_shape
// into:
// baseBuffer, baseOffset, baseSizes, baseStrides =
//     extract_strided_metadata(memref)
// sizes#reassIdx =
//     baseSizes#reassDim / product(expandShapeSizes#j,
//                                  for j in group excluding reassIdx)
// strides#reassIdx =
//     baseStrides#reassDim * product(expandShapeSizes#j, for j in
//                                    reassIdx+1..reassIdx+group.size)
//
// Here we have:
// For the group applying to dim0:
// size 0 = 3
// size 1 = 5
// size 2 = 2
// stride 0 = baseStrides#0 * 5 * 2
//          = 4 * 5 * 2
//          = 40
// stride 1 = baseStrides#0 * 2
//          = 4 * 2
//          = 8
// stride 2 = baseStrides#0
//          = 4
//
// For the group applying to dim1:
// size 3 = 2
// size 4 = 2
// stride 3 = baseStrides#1 * 2
//          = 1 * 2
//          = 2
// stride 4 = baseStrides#1
//          = 1
//
// Base and offset are unchanged.
//
// CHECK-LABEL: func @extract_strided_metadata_of_expand_shape_all_static
//  CHECK-SAME: (%[[ARG:.*]]: memref<30x4xi16>)
//
//   CHECK-DAG: %[[C40:.*]] = arith.constant 40 : index
//   CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
//   CHECK-DAG: %[[C5:.*]] = arith.constant 5 : index
//   CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
//   CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//
//   CHECK-DAG: %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:2, %[[STRIDES:.*]]:2 = memref.extract_strided_metadata %[[ARG]] : memref<30x4xi16> -> memref<i16>, index, index, index, index, index
//
//   CHECK: return %[[BASE]], %[[C0]], %[[C3]], %[[C5]], %[[C2]], %[[C2]], %[[C2]], %[[C40]], %[[C8]], %[[C4]], %[[C2]], %[[C1]] : memref<i16>, index, index, index, index, index, index, index, index, index, index, index
func.func @extract_strided_metadata_of_expand_shape_all_static(
    %arg : memref<30x4xi16>)
    -> (memref<i16>, index,
       index, index, index, index, index,
       index, index, index, index, index) {

  %expand_shape = memref.expand_shape %arg[[0, 1, 2], [3, 4]] :
    memref<30x4xi16> into memref<3x5x2x2x2xi16>

  %base, %offset, %sizes:5, %strides:5 = memref.extract_strided_metadata %expand_shape :
    memref<3x5x2x2x2xi16>
    -> memref<i16>, index,
       index, index, index, index, index,
       index, index, index, index, index

  return %base, %offset,
    %sizes#0, %sizes#1, %sizes#2, %sizes#3, %sizes#4,
    %strides#0, %strides#1, %strides#2, %strides#3, %strides#4 :
      memref<i16>, index,
      index, index, index, index, index,
      index, index, index, index, index
}

// -----

// Check that we properly simplify extract_strided_metadata of expand_shape
// when dynamic sizes, strides, and offsets are involved.
// See extract_strided_metadata_of_expand_shape_all_static for an explanation
// of the expansion.
//
// One of the important characteristic of this test is that the dynamic
// dimensions produced by the expand_shape appear both in the first dimension
// (for group 1) and the non-first dimension (second dimension for group 2.)
// The idea is to make sure that:
// 1. We properly account for dynamic shapes even when the strides are not
//    affected by them. (When the dynamic dimension is the first one.)
// 2. We properly compute the strides affected by dynamic shapes. (When the
//    dynamic dimension is not the first one.)
//
// Here we have:
// For the group applying to dim0:
// size 0 = baseSizes#0 / (all static sizes in that group)
//        = baseSizes#0 / (7 * 8 * 9)
//        = baseSizes#0 / 504
// size 1 = 7
// size 2 = 8
// size 3 = 9
// stride 0 = baseStrides#0 * 7 * 8 * 9
//          = baseStrides#0 * 504
// stride 1 = baseStrides#0 * 8 * 9
//          = baseStrides#0 * 72
// stride 2 = baseStrides#0 * 9
// stride 3 = baseStrides#0
//
// For the group applying to dim1:
// size 4 = 10
// size 5 = 2
// size 6 = baseSizes#1 / (all static sizes in that group)
//        = baseSizes#1 / (10 * 2 * 3)
//        = baseSizes#1 / 60
// size 7 = 3
// stride 4 = baseStrides#1 * size 5 * size 6 * size 7
//          = baseStrides#1 * 2 * (baseSizes#1 / 60) * 3
//          = baseStrides#1 * (baseSizes#1 / 60) * 6
//          and since we know that baseSizes#1 is a multiple of 60:
//          = baseStrides#1 * (baseSizes#1 / 10)
// stride 5 = baseStrides#1 * size 6 * size 7
//          = baseStrides#1 * (baseSizes#1 / 60) * 3
//          = baseStrides#1 * (baseSizes#1 / 20)
// stride 6 = baseStrides#1 * size 7
//          = baseStrides#1 * 3
// stride 7 = baseStrides#1
//
// Base and offset are unchanged.
//
//   CHECK-DAG: #[[$DIM0_SIZE_MAP:.*]] = affine_map<()[s0] -> (s0 floordiv 504)>
//   CHECK-DAG: #[[$DIM6_SIZE_MAP:.*]] = affine_map<()[s0] -> (s0 floordiv 60)>
//
//   CHECK-DAG: #[[$DIM0_STRIDE_MAP:.*]] = affine_map<()[s0] -> (s0 * 504)>
//   CHECK-DAG: #[[$DIM1_STRIDE_MAP:.*]] = affine_map<()[s0] -> (s0 * 72)>
//   CHECK-DAG: #[[$DIM2_STRIDE_MAP:.*]] = affine_map<()[s0] -> (s0 * 9)>
//   CHECK-DAG: #[[$DIM4_STRIDE_MAP:.*]] = affine_map<()[s0, s1] -> ((s0 floordiv 10) * s1)>
//   CHECK-DAG: #[[$DIM5_STRIDE_MAP:.*]] = affine_map<()[s0, s1] -> ((s0 floordiv 20) * s1)>
//   CHECK-DAG: #[[$DIM6_STRIDE_MAP:.*]] = affine_map<()[s0] -> (s0 * 3)>
// CHECK-LABEL: func @extract_strided_metadata_of_expand_shape_all_dynamic
//  CHECK-SAME: (%[[ARG:.*]]: memref<?x?xf32,
//
//   CHECK-DAG: %[[C10:.*]] = arith.constant 10 : index
//   CHECK-DAG: %[[C9:.*]] = arith.constant 9 : index
//   CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
//   CHECK-DAG: %[[C7:.*]] = arith.constant 7 : index
//   CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
//
//   CHECK-DAG: %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:2, %[[STRIDES:.*]]:2 = memref.extract_strided_metadata %[[ARG]] : memref<?x?xf32, strided<[?, ?], offset: ?>> -> memref<f32>, index, index, index, index, index
//
//   CHECK-DAG: %[[DYN_SIZE0:.*]] = affine.apply #[[$DIM0_SIZE_MAP]]()[%[[SIZES]]#0]
//   CHECK-DAG: %[[DYN_SIZE6:.*]] = affine.apply #[[$DIM6_SIZE_MAP]]()[%[[SIZES]]#1]
//   CHECK-DAG: %[[DYN_STRIDE0:.*]] = affine.apply #[[$DIM0_STRIDE_MAP]]()[%[[STRIDES]]#0]
//   CHECK-DAG: %[[DYN_STRIDE1:.*]] = affine.apply #[[$DIM1_STRIDE_MAP]]()[%[[STRIDES]]#0]
//   CHECK-DAG: %[[DYN_STRIDE2:.*]] = affine.apply #[[$DIM2_STRIDE_MAP]]()[%[[STRIDES]]#0]
//   CHECK-DAG: %[[DYN_STRIDE4:.*]] = affine.apply #[[$DIM4_STRIDE_MAP]]()[%[[SIZES]]#1, %[[STRIDES]]#1]
//   CHECK-DAG: %[[DYN_STRIDE5:.*]] = affine.apply #[[$DIM5_STRIDE_MAP]]()[%[[SIZES]]#1, %[[STRIDES]]#1]
//   CHECK-DAG: %[[DYN_STRIDE6:.*]] = affine.apply #[[$DIM6_STRIDE_MAP]]()[%[[STRIDES]]#1]

//   CHECK: return %[[BASE]], %[[OFFSET]], %[[DYN_SIZE0]], %[[C7]], %[[C8]], %[[C9]], %[[C10]], %[[C2]], %[[DYN_SIZE6]], %[[C3]], %[[DYN_STRIDE0]], %[[DYN_STRIDE1]], %[[DYN_STRIDE2]], %[[STRIDES]]#0, %[[DYN_STRIDE4]], %[[DYN_STRIDE5]], %[[DYN_STRIDE6]], %[[STRIDES]]#1 : memref<f32>, index, index, index, index, index, index, index, index, index, index, index, index, index
func.func @extract_strided_metadata_of_expand_shape_all_dynamic(
    %base: memref<?x?xf32, strided<[?,?], offset:?>>,
    %offset0: index, %offset1: index, %offset2: index,
    %size0: index, %size1: index, %size2: index,
    %stride0: index, %stride1: index, %stride2: index)
    -> (memref<f32>, index,
       index, index, index, index, index, index, index, index,
       index, index, index, index, index, index, index, index) {

  %subview = memref.expand_shape %base[[0, 1, 2, 3],[4, 5, 6, 7]] :
    memref<?x?xf32, strided<[?,?], offset: ?>> into
      memref<?x7x8x9x10x2x?x3xf32, strided<[?, ?, ?, ?, ?, ?, ?, ?], offset: ?>>

  %base_buffer, %offset, %sizes:8, %strides:8 = memref.extract_strided_metadata %subview :
    memref<?x7x8x9x10x2x?x3xf32, strided<[?, ?, ?, ?, ?, ?, ?, ?], offset: ?>>
    -> memref<f32>, index,
       index, index, index, index, index, index, index, index,
       index, index, index, index, index, index, index, index

  return %base_buffer, %offset,
    %sizes#0, %sizes#1, %sizes#2, %sizes#3, %sizes#4, %sizes#5, %sizes#6, %sizes#7,
    %strides#0, %strides#1, %strides#2, %strides#3, %strides#4, %strides#5, %strides#6, %strides#7 :
      memref<f32>, index,
      index, index, index, index, index, index, index, index,
      index, index, index, index, index, index, index, index
}


// -----

// Check that we properly handle extract_strided_metadata of expand_shape for
// 0-D input.
// The 0-D case is pretty boring:
// All expanded sizes are 1, likewise for the strides, and we keep the
// original base and offset.
// We have still a test for it, because since the input reassociation map
// of the expand_shape is empty, the handling of such shape hits a corner
// case.
// CHECK-LABEL: func @extract_strided_metadata_of_expand_shape_all_static_0_rank
//  CHECK-SAME: (%[[ARG:.*]]: memref<i16, strided<[], offset: ?>>)
//
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//
//   CHECK-DAG: %[[BASE:.*]], %[[OFFSET:.*]] = memref.extract_strided_metadata %[[ARG]] : memref<i16, strided<[], offset: ?>> -> memref<i16>, index
//
//   CHECK: return %[[BASE]], %[[OFFSET]], %[[C1]], %[[C1]], %[[C1]], %[[C1]], %[[C1]], %[[C1]], %[[C1]], %[[C1]], %[[C1]], %[[C1]] : memref<i16>, index, index, index, index, index, index, index, index, index, index, index
func.func @extract_strided_metadata_of_expand_shape_all_static_0_rank(
    %arg : memref<i16, strided<[], offset: ?>>)
    -> (memref<i16>, index,
       index, index, index, index, index,
       index, index, index, index, index) {

  %expand_shape = memref.expand_shape %arg[] :
    memref<i16, strided<[], offset: ?>> into memref<1x1x1x1x1xi16, strided<[1,1,1,1,1], offset: ?>>

  %base, %offset, %sizes:5, %strides:5 = memref.extract_strided_metadata %expand_shape :
    memref<1x1x1x1x1xi16, strided<[1,1,1,1,1], offset: ?>>
    -> memref<i16>, index,
       index, index, index, index, index,
       index, index, index, index, index

  return %base, %offset,
    %sizes#0, %sizes#1, %sizes#2, %sizes#3, %sizes#4,
    %strides#0, %strides#1, %strides#2, %strides#3, %strides#4 :
      memref<i16>, index,
      index, index, index, index, index,
      index, index, index, index, index
}

// -----

// Check that we simplify extract_strided_metadata(alloc)
// into simply the alloc with the information extracted from
// the memref type and arguments of the alloc.
//
// baseBuffer = reinterpret_cast alloc
// offset = 0
// sizes = shape(memref)
// strides = strides(memref)
//
// For dynamic shapes, we simply use the values that feed the alloc.
//
// Simple rank 0 test: we don't need a reinterpret_cast here.
// CHECK-LABEL: func @extract_strided_metadata_of_alloc_all_static_0_rank
//
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[ALLOC:.*]] = memref.alloc()
//       CHECK: return %[[ALLOC]], %[[C0]] : memref<i16>, index
func.func @extract_strided_metadata_of_alloc_all_static_0_rank()
    -> (memref<i16>, index) {

  %A = memref.alloc() : memref<i16>
  %base, %offset = memref.extract_strided_metadata %A :
    memref<i16>
    -> memref<i16>, index

  return %base, %offset :
      memref<i16>, index
}

// -----

// Simplification of extract_strided_metadata(alloc).
// Check that we properly use the dynamic sizes to
// create the new sizes and strides.
// size 0 = dyn_size0
// size 1 = 4
// size 2 = dyn_size2
// size 3 = dyn_size3
//
// stride 0 = size 1 * size 2 * size 3
//          = 4 * dyn_size2 * dyn_size3
// stride 1 = size 2 * size 3
//          = dyn_size2 * dyn_size3
// stride 2 = size 3
//          = dyn_size3
// stride 3 = 1
//
//   CHECK-DAG: #[[$STRIDE0_MAP:.*]] = affine_map<()[s0, s1] -> ((s0 * s1) * 4)>
//   CHECK-DAG: #[[$STRIDE1_MAP:.*]] = affine_map<()[s0, s1] -> (s0 * s1)>
// CHECK-LABEL: extract_strided_metadata_of_alloc_dyn_size
//  CHECK-SAME: (%[[DYN_SIZE0:.*]]: index, %[[DYN_SIZE2:.*]]: index, %[[DYN_SIZE3:.*]]: index)
//
//   CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[ALLOC:.*]] = memref.alloc(%[[DYN_SIZE0]], %[[DYN_SIZE2]], %[[DYN_SIZE3]])
//
//   CHECK-DAG: %[[STRIDE0:.*]] = affine.apply #[[$STRIDE0_MAP]]()[%[[DYN_SIZE2]], %[[DYN_SIZE3]]]
//   CHECK-DAG: %[[STRIDE1:.*]] = affine.apply #[[$STRIDE1_MAP]]()[%[[DYN_SIZE2]], %[[DYN_SIZE3]]]
//
//   CHECK-DAG:  %[[CASTED_ALLOC:.*]] = memref.reinterpret_cast %[[ALLOC]] to offset: [0], sizes: [], strides: [] : memref<?x4x?x?xi16> to memref<i16>
//
//       CHECK: return %[[CASTED_ALLOC]], %[[C0]], %[[DYN_SIZE0]], %[[C4]], %[[DYN_SIZE2]], %[[DYN_SIZE3]], %[[STRIDE0]], %[[STRIDE1]], %[[DYN_SIZE3]], %[[C1]]
func.func @extract_strided_metadata_of_alloc_dyn_size(
  %dyn_size0 : index, %dyn_size2 : index, %dyn_size3 : index)
    -> (memref<i16>, index,
        index, index, index, index,
        index, index, index, index) {

  %A = memref.alloc(%dyn_size0, %dyn_size2, %dyn_size3) : memref<?x4x?x?xi16>

  %base, %offset, %sizes:4, %strides:4 = memref.extract_strided_metadata %A :
    memref<?x4x?x?xi16>
    -> memref<i16>, index,
       index, index, index, index,
       index, index, index, index

  return %base, %offset,
    %sizes#0, %sizes#1, %sizes#2, %sizes#3,
    %strides#0, %strides#1, %strides#2, %strides#3 :
      memref<i16>, index,
      index, index, index, index,
      index, index, index, index
}

// -----

// Same check as extract_strided_metadata_of_alloc_dyn_size but alloca
// instead of alloc. Just to make sure we handle allocas the same way
// we do with alloc.
// While at it, test a slightly different shape than
// extract_strided_metadata_of_alloc_dyn_size.
//
// size 0 = dyn_size0
// size 1 = dyn_size1
// size 2 = 4
// size 3 = dyn_size3
//
// stride 0 = size 1 * size 2 * size 3
//          = dyn_size1 * 4 * dyn_size3
// stride 1 = size 2 * size 3
//          = 4 * dyn_size3
// stride 2 = size 3
//          = dyn_size3
// stride 3 = 1
//
//   CHECK-DAG: #[[$STRIDE0_MAP:.*]] = affine_map<()[s0, s1] -> ((s0 * s1) * 4)>
//   CHECK-DAG: #[[$STRIDE1_MAP:.*]] = affine_map<()[s0] -> (s0 * 4)>
// CHECK-LABEL: extract_strided_metadata_of_alloca_dyn_size
//  CHECK-SAME: (%[[DYN_SIZE0:.*]]: index, %[[DYN_SIZE1:.*]]: index, %[[DYN_SIZE3:.*]]: index)
//
//   CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[ALLOCA:.*]] = memref.alloca(%[[DYN_SIZE0]], %[[DYN_SIZE1]], %[[DYN_SIZE3]])
//
//   CHECK-DAG: %[[STRIDE0:.*]] = affine.apply #[[$STRIDE0_MAP]]()[%[[DYN_SIZE1]], %[[DYN_SIZE3]]]
//   CHECK-DAG: %[[STRIDE1:.*]] = affine.apply #[[$STRIDE1_MAP]]()[%[[DYN_SIZE3]]]
//
//   CHECK-DAG:  %[[CASTED_ALLOCA:.*]] = memref.reinterpret_cast %[[ALLOCA]] to offset: [0], sizes: [], strides: [] : memref<?x?x4x?xi16> to memref<i16>
//
//       CHECK: return %[[CASTED_ALLOCA]], %[[C0]], %[[DYN_SIZE0]], %[[DYN_SIZE1]], %[[C4]], %[[DYN_SIZE3]], %[[STRIDE0]], %[[STRIDE1]], %[[DYN_SIZE3]], %[[C1]]
func.func @extract_strided_metadata_of_alloca_dyn_size(
  %dyn_size0 : index, %dyn_size1 : index, %dyn_size3 : index)
    -> (memref<i16>, index,
        index, index, index, index,
        index, index, index, index) {

  %A = memref.alloca(%dyn_size0, %dyn_size1, %dyn_size3) : memref<?x?x4x?xi16>

  %base, %offset, %sizes:4, %strides:4 = memref.extract_strided_metadata %A :
    memref<?x?x4x?xi16>
    -> memref<i16>, index,
       index, index, index, index,
       index, index, index, index

  return %base, %offset,
    %sizes#0, %sizes#1, %sizes#2, %sizes#3,
    %strides#0, %strides#1, %strides#2, %strides#3 :
      memref<i16>, index,
      index, index, index, index,
      index, index, index, index
}

// -----

// The following few alloc tests are negative tests (the simplification
// doesn't happen) to make sure non trivial memref types are treated
// as "not been normalized".
// CHECK-LABEL: extract_strided_metadata_of_alloc_with_variable_offset
//       CHECK: %[[ALLOC:.*]] = memref.alloc
//       CHECK: %[[BASE:[^,]*]], {{.*}} = memref.extract_strided_metadata %[[ALLOC]]
//       CHECK: return %[[BASE]]
#map0 = affine_map<(d0)[s0] -> (d0 + s0)>
func.func @extract_strided_metadata_of_alloc_with_variable_offset(%arg : index)
    -> (memref<i16>, index, index, index) {

  %A = memref.alloc()[%arg] : memref<4xi16, #map0>
  %base, %offset, %size, %stride = memref.extract_strided_metadata %A :
    memref<4xi16, #map0>
    -> memref<i16>, index, index, index

  return %base, %offset, %size, %stride :
      memref<i16>, index, index, index
}

// -----

// CHECK-LABEL: extract_strided_metadata_of_alloc_with_cst_offset
//       CHECK: %[[ALLOC:.*]] = memref.alloc
//       CHECK: %[[BASE:[^,]*]], {{.*}} = memref.extract_strided_metadata %[[ALLOC]]
//       CHECK: return %[[BASE]]
#map0 = affine_map<(d0) -> (d0 + 12)>
func.func @extract_strided_metadata_of_alloc_with_cst_offset(%arg : index)
    -> (memref<i16>, index, index, index) {

  %A = memref.alloc() : memref<4xi16, #map0>
  %base, %offset, %size, %stride = memref.extract_strided_metadata %A :
    memref<4xi16, #map0>
    -> memref<i16>, index, index, index

  return %base, %offset, %size, %stride :
      memref<i16>, index, index, index
}

// -----

// CHECK-LABEL: extract_strided_metadata_of_alloc_with_cst_offset_in_type
//       CHECK: %[[ALLOC:.*]] = memref.alloc
//       CHECK: %[[BASE:[^,]*]], {{.*}} = memref.extract_strided_metadata %[[ALLOC]]
//       CHECK: return %[[BASE]]
func.func @extract_strided_metadata_of_alloc_with_cst_offset_in_type(%arg : index)
    -> (memref<i16>, index, index, index) {

  %A = memref.alloc() : memref<4xi16, strided<[1], offset : 10>>
  %base, %offset, %size, %stride = memref.extract_strided_metadata %A :
    memref<4xi16, strided<[1], offset : 10>>
    -> memref<i16>, index, index, index

  return %base, %offset, %size, %stride :
      memref<i16>, index, index, index
}

// -----

// CHECK-LABEL: extract_strided_metadata_of_alloc_with_strided
//       CHECK: %[[ALLOC:.*]] = memref.alloc
//       CHECK: %[[BASE:[^,]*]], {{.*}} = memref.extract_strided_metadata %[[ALLOC]]
//       CHECK: return %[[BASE]]
func.func @extract_strided_metadata_of_alloc_with_strided(%arg : index)
    -> (memref<i16>, index, index, index) {

  %A = memref.alloc() : memref<4xi16, strided<[12]>>
  %base, %offset, %size, %stride = memref.extract_strided_metadata %A :
    memref<4xi16, strided<[12]>>
    -> memref<i16>, index, index, index

  return %base, %offset, %size, %stride :
      memref<i16>, index, index, index
}
