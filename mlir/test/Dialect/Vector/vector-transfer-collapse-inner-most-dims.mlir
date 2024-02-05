// RUN: mlir-opt %s -test-vector-transfer-collapse-inner-most-dims -split-input-file | FileCheck %s

func.func @contiguous_inner_most_view(%in: memref<1x1x8x1xf32, strided<[3072, 8, 1, 1], offset: ?>>) -> vector<1x8x1xf32>{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %0 = vector.transfer_read %in[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x1x8x1xf32, strided<[3072, 8, 1, 1], offset: ?>>, vector<1x8x1xf32>
  return %0 : vector<1x8x1xf32>
}
//      CHECK: func @contiguous_inner_most_view(%[[SRC:.+]]: memref<1x1x8x1xf32, strided<[3072, 8, 1, 1], offset: ?>>
//      CHECK:   %[[SRC_0:.+]] = memref.subview %[[SRC]]
// CHECK-SAME:    memref<1x1x8x1xf32, strided<[3072, 8, 1, 1], offset: ?>> to memref<1x1x8xf32, strided<[3072, 8, 1], offset: ?>>
//      CHECK:   %[[VEC:.+]] = vector.transfer_read %[[SRC_0]]
// CHECK-SAME:    memref<1x1x8xf32, strided<[3072, 8, 1], offset: ?>>, vector<1x8xf32>
//      CHECK:   %[[RESULT:.+]] = vector.shape_cast %[[VEC]]
//      CHECK:   return %[[RESULT]]

// -----

func.func @contiguous_inner_most_dim(%A: memref<16x1xf32>, %i:index, %j:index) -> (vector<8x1xf32>) {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32
  %1 = vector.transfer_read %A[%i, %j], %f0 : memref<16x1xf32>, vector<8x1xf32>
  return %1 : vector<8x1xf32>
}
//      CHECK: func @contiguous_inner_most_dim(%[[SRC:.+]]: memref<16x1xf32>, %[[I:.+]]: index, %[[J:.+]]: index) -> vector<8x1xf32>
//      CHECK:   %[[SRC_0:.+]] = memref.subview %[[SRC]]
// CHECK-SAME:     memref<16x1xf32> to memref<16xf32>
//      CHECK:   %[[V:.+]] = vector.transfer_read %[[SRC_0]]
//      CHECK:   %[[RESULT]] = vector.shape_cast %[[V]] : vector<8xf32> to vector<8x1xf32>
//      CHECK:   return %[[RESULT]]

// -----

func.func @contiguous_inner_most_dim_bounds(%A: memref<1000x1xf32>, %i:index, %ii:index) -> (vector<4x1xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %0 = memref.subview %A[%i, 0] [40, 1] [1, 1] : memref<1000x1xf32> to memref<40x1xf32, strided<[1, 1], offset: ?>>
  %1 = vector.transfer_read %0[%ii, %c0], %cst {in_bounds = [true, true]} : memref<40x1xf32, strided<[1, 1], offset: ?>>, vector<4x1xf32>
  return %1 : vector<4x1xf32>
}
//      CHECK: func @contiguous_inner_most_dim_bounds(%[[SRC:.+]]: memref<1000x1xf32>, %[[II:.+]]: index, %[[J:.+]]: index) -> vector<4x1xf32>
//      CHECK:   %[[SRC_0:.+]] = memref.subview %[[SRC]]
//      CHECK:   %[[SRC_1:.+]] = memref.subview %[[SRC_0]]
//      CHECK:   %[[V:.+]] = vector.transfer_read %[[SRC_1]]
// CHECK-SAME:       {in_bounds = [true]}
// CHECK-SAME:       vector<4xf32>

// -----

func.func @contiguous_inner_most_dim_bounds_2d(%A: memref<1000x1x1xf32>, %i:index, %ii:index) -> (vector<4x1x1xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %0 = memref.subview %A[%i, 0, 0] [40, 1, 1] [1, 1, 1] : memref<1000x1x1xf32> to memref<40x1x1xf32, strided<[1, 1, 1], offset: ?>>
  %1 = vector.transfer_read %0[%ii, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<40x1x1xf32, strided<[1, 1, 1], offset: ?>>, vector<4x1x1xf32>
  return %1 : vector<4x1x1xf32>
}
//      CHECK: func @contiguous_inner_most_dim_bounds_2d(%[[SRC:.+]]: memref<1000x1x1xf32>, %[[II:.+]]: index, %[[J:.+]]: index) -> vector<4x1x1xf32>
//      CHECK:   %[[SRC_0:.+]] = memref.subview %[[SRC]]
//      CHECK:   %[[SRC_1:.+]] = memref.subview %[[SRC_0]]
//      CHECK:   %[[V:.+]] = vector.transfer_read %[[SRC_1]]
// CHECK-SAME:       {in_bounds = [true]}
// CHECK-SAME:       vector<4xf32>

// -----

func.func @contiguous_inner_most_dim_out_of_bounds_2d(%arg0: memref<1x1xf32>) -> vector<4x8xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<1x1xf32>, vector<4x8xf32>
  return %0 : vector<4x8xf32>
}
// The inner most unit dim can not be dropped. In this context, we do not
// generate rank-reduced memref.subview ops.
//      CHECK: func.func @contiguous_inner_most_dim_out_of_bounds_2d
// CHECK-SAME:   %[[SRC:[a-zA-Z0-9]+]]
//  CHECK-NOT:   memref.subview
//      CHECK:   %[[READ:.+]] = vector.transfer_read %[[SRC]]
//      CHECK:   return %[[READ]] : vector<4x8xf32>

// -----

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
// CHECK-SAME:     memref<1x512x16x1x1xf32> to memref<1x512x16xf32>
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
