// RUN: mlir-opt %s -test-vector-transfer-collapse-inner-most-dims -split-input-file | FileCheck %s

//-----------------------------------------------------------------------------
// 1. vector.transfer_read
//-----------------------------------------------------------------------------

func.func @contiguous_inner_most(%in: memref<1x1x8x1xf32, strided<[3072, 8, 1, 1], offset: ?>>) -> vector<1x8x1xf32>{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %0 = vector.transfer_read %in[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x1x8x1xf32, strided<[3072, 8, 1, 1], offset: ?>>, vector<1x8x1xf32>
  return %0 : vector<1x8x1xf32>
}

//      CHECK: func @contiguous_inner_most(%[[SRC:.+]]: memref<1x1x8x1xf32, strided<[3072, 8, 1, 1], offset: ?>>
//      CHECK:   %[[SRC_0:.+]] = memref.subview %[[SRC]]
// CHECK-SAME:    memref<1x1x8x1xf32, strided<[3072, 8, 1, 1], offset: ?>> to memref<1x1x8xf32, strided<[3072, 8, 1], offset: ?>>
//      CHECK:   %[[VEC:.+]] = vector.transfer_read %[[SRC_0]]
// CHECK-SAME:    memref<1x1x8xf32, strided<[3072, 8, 1], offset: ?>>, vector<1x8xf32>
//      CHECK:   %[[RESULT:.+]] = vector.shape_cast %[[VEC]]
//      CHECK:   return %[[RESULT]]

// Same as the top example within this split, but with the inner vector
// dim scalable. Note that this example only makes sense when "8 = [8]" (i.e.
// vscale = 1). This is assumed (implicitly) via the `in_bounds` attribute.

func.func @contiguous_inner_most_scalable_inner_dim(%in: memref<1x1x8x1xf32, strided<[3072, 8, 1, 1], offset: ?>>) -> vector<1x[8]x1xf32>{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %0 = vector.transfer_read %in[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x1x8x1xf32, strided<[3072, 8, 1, 1], offset: ?>>, vector<1x[8]x1xf32>
  return %0 : vector<1x[8]x1xf32>
}

//      CHECK: func @contiguous_inner_most_scalable_inner_dim(%[[SRC:.+]]: memref<1x1x8x1xf32, strided<[3072, 8, 1, 1], offset: ?>>
//      CHECK:   %[[SRC_0:.+]] = memref.subview %[[SRC]]
// CHECK-SAME:    memref<1x1x8x1xf32, strided<[3072, 8, 1, 1], offset: ?>> to memref<1x1x8xf32, strided<[3072, 8, 1], offset: ?>>
//      CHECK:   %[[VEC:.+]] = vector.transfer_read %[[SRC_0]]
// CHECK-SAME:    memref<1x1x8xf32, strided<[3072, 8, 1], offset: ?>>, vector<1x[8]xf32>
//      CHECK:   %[[RESULT:.+]] = vector.shape_cast %[[VEC]]
//      CHECK:   return %[[RESULT]]

// Same as the top example within this split, but the trailing unit dim was
// replaced with a dyn dim - not supported

func.func @non_unit_trailing_dim(%in: memref<1x1x8x?xf32, strided<[3072, 8, 1, 1], offset: ?>>) -> vector<1x8x1xf32>{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %0 = vector.transfer_read %in[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x1x8x?xf32, strided<[3072, 8, 1, 1], offset: ?>>, vector<1x8x1xf32>
  return %0 : vector<1x8x1xf32>
}

//  CHECK-LABEL: func @non_unit_trailing_dim
//    CHECK-NOT: memref.subview
//    CHECK-NOT: vector.shape_cast

// Same as the top example within this split, but with a scalable unit dim in
// the output vector - not supported (scalable 1 is _not_ a unit dimension).

func.func @negative_scalable_unit_dim(%in: memref<1x1x8x1xf32, strided<[3072, 8, 1, 1], offset: ?>>) -> vector<1x8x[1]xf32>{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %0 = vector.transfer_read %in[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x1x8x1xf32, strided<[3072, 8, 1, 1], offset: ?>>, vector<1x8x[1]xf32>
  return %0 : vector<1x8x[1]xf32>
}
//  CHECK-LABEL: func @negative_scalable_unit_dim
//    CHECK-NOT: memref.subview
//    CHECK-NOT: vector.shape_cast

// -----

func.func @contiguous_inner_most_dynamic_outer(%a: index, %b: index, %memref: memref<?x?x8x1xf32>) -> vector<8x1xf32> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %v = vector.transfer_read %memref[%a, %b, %c0, %c0], %pad {in_bounds = [true, true]} : memref<?x?x8x1xf32>, vector<8x1xf32>
  return %v : vector<8x1xf32>
}
// CHECK: func.func @contiguous_inner_most_dynamic_outer
// CHECK-SAME:   %[[IDX0:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[IDX1:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[SRC:[a-zA-Z0-9]+]]
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[PAD:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[D0:.+]] = memref.dim %[[SRC]], %[[C0]]
// CHECK:        %[[D1:.+]] = memref.dim %[[SRC]], %[[C1]]
// CHECK:        %[[VIEW:.+]] = memref.subview %[[SRC]][0, 0, 0, 0] [%[[D0]], %[[D1]], 8, 1] [1, 1, 1, 1]
// CHECK-SAME:     memref<?x?x8x1xf32> to memref<?x?x8xf32, strided<[?, 8, 1], offset: ?>>
// CHECK:        %[[VEC:.+]] = vector.transfer_read %[[VIEW]]
// CHECK-SAME:     memref<?x?x8xf32, strided<[?, 8, 1], offset: ?>>, vector<8xf32>
// CHECK:        %[[RESULT:.+]] = vector.shape_cast %[[VEC]]
// CHECK:        return %[[RESULT]]

// Same as the top example within this split, but with the outer vector
// dim scalable. Note that this example only makes sense when "8 = [8]" (i.e.
// vscale = 1). This is assumed (implicitly) via the `in_bounds` attribute.

func.func @contiguous_inner_most_outer_dim_dyn_scalable_inner_dim(%a: index, %b: index, %memref: memref<?x?x8x1xf32>) -> vector<[8]x1xf32> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %v = vector.transfer_read %memref[%a, %b, %c0, %c0], %pad {in_bounds = [true, true]} : memref<?x?x8x1xf32>, vector<[8]x1xf32>
  return %v : vector<[8]x1xf32>
}
// CHECK-LABEL:  func @contiguous_inner_most_outer_dim_dyn_scalable_inner_dim
// CHECK-SAME:   %[[IDX0:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[IDX1:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[SRC:[a-zA-Z0-9]+]]
// CHECK:         %[[VIEW:.+]] = memref.subview %[[SRC]]{{.*}} memref<?x?x8x1xf32> to memref<?x?x8xf32, strided<[?, 8, 1], offset: ?>>
// CHECK:         %[[VEC_READ:.+]] = vector.transfer_read %[[VIEW]]
// CHECK-SAME:    {in_bounds = [true]}
// CHECK-SAME:     memref<?x?x8xf32, strided<[?, 8, 1], offset: ?>>, vector<[8]xf32>
// CHECK:         vector.shape_cast %[[VEC_READ]]

// -----

func.func @contiguous_inner_most_dim_non_zero_idx(%A: memref<16x1xf32>, %i:index) -> (vector<8x1xf32>) {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32
  %1 = vector.transfer_read %A[%i, %c0], %f0 : memref<16x1xf32>, vector<8x1xf32>
  return %1 : vector<8x1xf32>
}
//      CHECK: func @contiguous_inner_most_dim_non_zero_idx(%[[SRC:.+]]: memref<16x1xf32>, %[[I:.+]]: index) -> vector<8x1xf32>
//      CHECK:   %[[SRC_0:.+]] = memref.subview %[[SRC]]
// CHECK-SAME:     memref<16x1xf32> to memref<16xf32, strided<[1]>>
//      CHECK:   %[[V:.+]] = vector.transfer_read %[[SRC_0]]
//      CHECK:   %[[RESULT:.+]] = vector.shape_cast %[[V]] : vector<8xf32> to vector<8x1xf32>
//      CHECK:   return %[[RESULT]]

// The index to be dropped is != 0 - this is currently not supported.
func.func @negative_contiguous_inner_most_dim_non_zero_idxs(%A: memref<16x1xf32>, %i:index) -> (vector<8x1xf32>) {
  %f0 = arith.constant 0.0 : f32
  %1 = vector.transfer_read %A[%i, %i], %f0 : memref<16x1xf32>, vector<8x1xf32>
  return %1 : vector<8x1xf32>
}
// CHECK-LABEL: func @negative_contiguous_inner_most_dim_non_zero_idxs
// CHECK-NOT:     memref.subview
// CHECK:         vector.transfer_read

// Same as the top example within this split, but with the outer vector
// dim scalable. Note that this example only makes sense when "8 = [8]" (i.e.
// vscale = 1). This is assumed (implicitly) via the `in_bounds` attribute.

func.func @contiguous_inner_most_dim_non_zero_idx_scalable_inner_dim(%A: memref<16x1xf32>, %i:index) -> (vector<[8]x1xf32>) {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32
  %1 = vector.transfer_read %A[%i, %c0], %f0 : memref<16x1xf32>, vector<[8]x1xf32>
  return %1 : vector<[8]x1xf32>
}
// CHECK-LABEL: func @contiguous_inner_most_dim_non_zero_idx_scalable_inner_dim(
// CHECK-SAME:    %[[SRC:.+]]: memref<16x1xf32>, %[[I:.+]]: index) -> vector<[8]x1xf32>
//       CHECK:   %[[SRC_0:.+]] = memref.subview %[[SRC]]
//  CHECK-SAME:     memref<16x1xf32> to memref<16xf32, strided<[1]>>
//       CHECK:   %[[V:.+]] = vector.transfer_read %[[SRC_0]]
//       CHECK:   %[[RESULT:.+]] = vector.shape_cast %[[V]] : vector<[8]xf32> to vector<[8]x1xf32>
//       CHECK:   return %[[RESULT]]

// -----

func.func @contiguous_inner_most_dim_with_subview(%A: memref<1000x1xf32>, %i:index, %ii:index) -> (vector<4x1xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %0 = memref.subview %A[%i, 0] [40, 1] [1, 1] : memref<1000x1xf32> to memref<40x1xf32, strided<[1, 1], offset: ?>>
  %1 = vector.transfer_read %0[%ii, %c0], %cst {in_bounds = [true, true]} : memref<40x1xf32, strided<[1, 1], offset: ?>>, vector<4x1xf32>
  return %1 : vector<4x1xf32>
}
//      CHECK: func @contiguous_inner_most_dim_with_subview(%[[SRC:.+]]: memref<1000x1xf32>, %[[II:.+]]: index, %[[J:.+]]: index) -> vector<4x1xf32>
//      CHECK:   %[[SRC_0:.+]] = memref.subview %[[SRC]]
//      CHECK:   %[[SRC_1:.+]] = memref.subview %[[SRC_0]]
//      CHECK:   %[[V:.+]] = vector.transfer_read %[[SRC_1]]
// CHECK-SAME:       {in_bounds = [true]}
// CHECK-SAME:       vector<4xf32>

// Same as the top example within this split, but with the outer vector
// dim scalable. Note that this example only makes sense when "4 = [4]" (i.e.
// vscale = 1). This is assumed (implicitly) via the `in_bounds` attribute.

func.func @contiguous_inner_most_dim_with_subview_scalable_inner_dim(%A: memref<1000x1xf32>, %i:index, %ii:index) -> (vector<[4]x1xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %0 = memref.subview %A[%i, 0] [40, 1] [1, 1] : memref<1000x1xf32> to memref<40x1xf32, strided<[1, 1], offset: ?>>
  %1 = vector.transfer_read %0[%ii, %c0], %cst {in_bounds = [true, true]} : memref<40x1xf32, strided<[1, 1], offset: ?>>, vector<[4]x1xf32>
  return %1 : vector<[4]x1xf32>
}
// CHECK-LABEL: func @contiguous_inner_most_dim_with_subview_scalable_inner_dim
//  CHECK-SAME:   %[[SRC:.+]]: memref<1000x1xf32>
//       CHECK:   %[[SRC_0:.+]] = memref.subview %[[SRC]]
//       CHECK:   %[[V:.+]] = vector.transfer_read %[[SRC_0]]
//  CHECK-SAME:       {in_bounds = [true]}
//  CHECK-SAME:       vector<[4]xf32>

// -----

func.func @contiguous_inner_most_dim_with_subview_2d(%A: memref<1000x1x1xf32>, %i:index, %ii:index) -> (vector<4x1x1xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %0 = memref.subview %A[%i, 0, 0] [40, 1, 1] [1, 1, 1] : memref<1000x1x1xf32> to memref<40x1x1xf32, strided<[1, 1, 1], offset: ?>>
  %1 = vector.transfer_read %0[%ii, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<40x1x1xf32, strided<[1, 1, 1], offset: ?>>, vector<4x1x1xf32>
  return %1 : vector<4x1x1xf32>
}
//      CHECK: func @contiguous_inner_most_dim_with_subview_2d(%[[SRC:.+]]: memref<1000x1x1xf32>, %[[II:.+]]: index, %[[J:.+]]: index) -> vector<4x1x1xf32>
//      CHECK:   %[[SRC_0:.+]] = memref.subview %[[SRC]]
//      CHECK:   %[[SRC_1:.+]] = memref.subview %[[SRC_0]]
//      CHECK:   %[[V:.+]] = vector.transfer_read %[[SRC_1]]
// CHECK-SAME:       {in_bounds = [true]}
// CHECK-SAME:       vector<4xf32>

// Same as the top example within this split, but with the outer vector
// dim scalable. Note that this example only makes sense when "4 = [4]" (i.e.
// vscale = 1). This is assumed (implicitly) via the `in_bounds` attribute.

func.func @contiguous_inner_most_dim_with_subview_2d_scalable_inner_dim(%A: memref<1000x1x1xf32>, %i:index, %ii:index) -> (vector<[4]x1x1xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %0 = memref.subview %A[%i, 0, 0] [40, 1, 1] [1, 1, 1] : memref<1000x1x1xf32> to memref<40x1x1xf32, strided<[1, 1, 1], offset: ?>>
  %1 = vector.transfer_read %0[%ii, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<40x1x1xf32, strided<[1, 1, 1], offset: ?>>, vector<[4]x1x1xf32>
  return %1 : vector<[4]x1x1xf32>
}
// CHECK-LABEL: func @contiguous_inner_most_dim_with_subview_2d_scalable_inner_dim(
//  CHECK-SAME:   %[[SRC:.+]]: memref<1000x1x1xf32>, %[[II:.+]]: index, %[[J:.+]]: index) -> vector<[4]x1x1xf32>
//       CHECK:   %[[SRC_0:.+]] = memref.subview %[[SRC]]
//       CHECK:   %[[SRC_1:.+]] = memref.subview %[[SRC_0]]
//       CHECK:   %[[V:.+]] = vector.transfer_read %[[SRC_1]]
//  CHECK-SAME:       {in_bounds = [true]}
//  CHECK-SAME:       vector<[4]xf32>
//       CHECK:  vector.shape_cast %[[V]]

// -----

// NOTE: This is an out-of-bounds access.

func.func @negative_non_unit_inner_vec_dim(%arg0: memref<4x1xf32>) -> vector<4x8xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<4x1xf32>, vector<4x8xf32>
  return %0 : vector<4x8xf32>
}
//      CHECK: func.func @negative_non_unit_inner_vec_dim
//  CHECK-NOT:   memref.subview
//      CHECK:   vector.transfer_read

// -----

func.func @negative_non_unit_inner_memref_dim(%arg0: memref<4x8xf32>) -> vector<4x1xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<4x8xf32>, vector<4x1xf32>
  return %0 : vector<4x1xf32>
}
//      CHECK: func.func @negative_non_unit_inner_memref_dim
//  CHECK-NOT:   memref.subview
//      CHECK:   vector.transfer_read

// -----

//-----------------------------------------------------------------------------
// 2. vector.transfer_write
//-----------------------------------------------------------------------------

func.func @drop_two_inner_most_dim_for_transfer_write(%arg0: memref<1x512x16x1x1xf32>, %arg1: vector<1x16x16x1x1xf32>, %arg2: index) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %arg1, %arg0[%c0, %arg2, %c0, %c0, %c0]
    {in_bounds = [true, true, true, true, true]}
    : vector<1x16x16x1x1xf32>, memref<1x512x16x1x1xf32>
  return
}
// CHECK:      func.func @drop_two_inner_most_dim_for_transfer_write
// CHECK-SAME:   %[[DEST:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[VEC:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[IDX:[a-zA-Z0-9]+]]
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK:        %[[SUBVIEW:.+]] = memref.subview %[[DEST]]
// CHECK-SAME:     memref<1x512x16x1x1xf32> to memref<1x512x16xf32, strided<[8192, 16, 1]>>
// CHECK:        %[[CAST:.+]] = vector.shape_cast %[[VEC]] : vector<1x16x16x1x1xf32> to vector<1x16x16xf32>
// CHECK:        vector.transfer_write %[[CAST]], %[[SUBVIEW]]
// CHECK-SAME:     [%[[C0]], %[[IDX]], %[[C0]]]

// -----

func.func @drop_inner_most_dim_for_transfer_write(%arg0: memref<1x512x16x1xf32, strided<[8192, 16, 1, 1], offset: ?>>, %arg1: vector<1x16x16x1xf32>, %arg2: index) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %arg1, %arg0[%c0, %arg2, %c0, %c0]
    {in_bounds = [true, true, true, true]}
    : vector<1x16x16x1xf32>, memref<1x512x16x1xf32, strided<[8192, 16, 1, 1], offset: ?>>
  return
}
// CHECK:      func.func @drop_inner_most_dim_for_transfer_write
// CHECK-SAME:   %[[DEST:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[VEC:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[IDX:[a-zA-Z0-9]+]]
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK:        %[[SUBVIEW:.+]] = memref.subview %[[DEST]]
// CHECK-SAME:     memref<1x512x16x1xf32, strided<[8192, 16, 1, 1], offset: ?>> to memref<1x512x16xf32, strided<[8192, 16, 1], offset: ?>>
// CHECK:        %[[CAST:.+]] = vector.shape_cast %[[VEC]] : vector<1x16x16x1xf32> to vector<1x16x16xf32>
// CHECK:        vector.transfer_write %[[CAST]], %[[SUBVIEW]]
// CHECK-SAME:     [%[[C0]], %[[IDX]], %[[C0]]]

// -----

func.func @outer_dyn_drop_inner_most_dim_for_transfer_write(%arg0: memref<?x512x16x1xf32, strided<[8192, 16, 1, 1], offset: ?>>, %arg1: vector<1x16x16x1xf32>, %arg2: index) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %arg1, %arg0[%arg2, %c0, %c0, %c0]
    {in_bounds = [true, true, true, true]}
    : vector<1x16x16x1xf32>, memref<?x512x16x1xf32, strided<[8192, 16, 1, 1], offset: ?>>
  return
}
// CHECK:      func.func @outer_dyn_drop_inner_most_dim_for_transfer_write
// CHECK-SAME:   %[[DEST:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[VEC:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[IDX:[a-zA-Z0-9]+]]
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[D0:.+]] = memref.dim %[[SRC]], %[[C0]]
// CHECK:        %[[SUBVIEW:.+]] = memref.subview %[[DEST]][0, 0, 0, 0] [%[[D0]], 512, 16, 1]
// CHECK-SAME:     memref<?x512x16x1xf32, strided<[8192, 16, 1, 1], offset: ?>> to memref<?x512x16xf32, strided<[8192, 16, 1], offset: ?>>
// CHECK:        %[[CAST:.+]] = vector.shape_cast %[[VEC]] : vector<1x16x16x1xf32> to vector<1x16x16xf32>
// CHECK:        vector.transfer_write %[[CAST]], %[[SUBVIEW]]
// CHECK-SAME:     [%[[IDX]], %[[C0]], %[[C0]]]

// -----

func.func @non_unit_strides(%arg0: memref<512x16x1xf32, strided<[8192, 16, 4], offset: ?>>, %arg1: vector<16x16x1xf32>, %arg2: index) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %arg1, %arg0[%arg2, %c0, %c0]
    {in_bounds = [true, true, true]}
    : vector<16x16x1xf32>, memref<512x16x1xf32, strided<[8192, 16, 4], offset: ?>>
  return
}
// The inner most unit dims can not be dropped if the strides are not ones.
// CHECK:     func.func @non_unit_strides
// CHECK-NOT:   memref.subview

// -----

func.func @leading_scalable_dimension_transfer_write(%dest : memref<24x1xf32>, %vec: vector<[4]x1xf32>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %vec, %dest[%c0, %c0] {in_bounds = [true, true]} : vector<[4]x1xf32>,  memref<24x1xf32>
  return
}
// CHECK:      func.func @leading_scalable_dimension_transfer_write
// CHECK-SAME:   %[[DEST:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[VEC:[a-zA-Z0-9]+]]
// CHECK:        %[[SUBVIEW:.+]] = memref.subview %[[DEST]][0, 0] [24, 1] [1, 1] : memref<24x1xf32> to memref<24xf32, strided<[1]>>
// CHECK:        %[[CAST:.+]] = vector.shape_cast %[[VEC]] : vector<[4]x1xf32> to vector<[4]xf32>
// CHECK:        vector.transfer_write %[[CAST]], %[[SUBVIEW]]{{.*}} {in_bounds = [true]} : vector<[4]xf32>, memref<24xf32, strided<[1]>>

// -----

// Negative test: [1] (scalable 1) is _not_ a unit dimension.
func.func @trailing_scalable_one_dim_transfer_write(%dest : memref<24x1xf32>, %vec: vector<4x[1]xf32>, %index: index) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %vec, %dest[%index, %c0] {in_bounds = [true, true]} : vector<4x[1]xf32>,  memref<24x1xf32>
  return
}
// CHECK:      func.func @trailing_scalable_one_dim_transfer_write
// CHECK-NOT:    vector.shape_cast
// CHECK:        vector.transfer_write {{.*}} : vector<4x[1]xf32>,  memref<24x1xf32>
// CHECK-NOT:    vector.shape_cast
