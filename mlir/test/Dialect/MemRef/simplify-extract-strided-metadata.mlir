// RUN: mlir-opt --simplify-extract-strided-metadata -split-input-file %s -o - | FileCheck %s

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
