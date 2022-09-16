// RUN: mlir-opt %s -test-vector-transfer-collapse-inner-most-dims -split-input-file | FileCheck %s

func.func @contiguous_inner_most_view(%in: memref<1x1x8x1xf32, strided<[3072, 8, 1, 1], offset: ?>>) -> vector<1x8x1xf32>{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %0 = vector.transfer_read %in[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x1x8x1xf32, strided<[3072, 8, 1, 1], offset: ?>>, vector<1x8x1xf32>
  return %0 : vector<1x8x1xf32>
}
//      CHECK: func @contiguous_inner_most_view(%[[SRC:.+]]: memref<1x1x8x1xf32, strided<[3072, 8, 1, 1], offset: ?>>
//      CHECK:   %[[SRC_0:.+]] = memref.subview %[[SRC]]
// CHECK-SAME:    memref<1x1x8x1xf32, strided<[3072, 8, 1, 1], offset: ?>> to memref<1x1x8xf32
//      CHECK:   %[[VEC:.+]] = vector.transfer_read %[[SRC_0]]
// CHECK-SAME:    memref<1x1x8xf32, {{.*}}>, vector<1x8xf32>
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
  %0 = memref.subview %A[%i, 0] [40, 1] [1, 1] : memref<1000x1xf32> to memref<40x1xf32, affine_map<(d0, d1)[s0] -> (d0 + s0 + d1)>>
  %1 = vector.transfer_read %0[%ii, %c0], %cst {in_bounds = [true, true]} : memref<40x1xf32, affine_map<(d0, d1)[s0] -> (d0 + s0 + d1)>>, vector<4x1xf32>
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
  %0 = memref.subview %A[%i, 0, 0] [40, 1, 1] [1, 1, 1] : memref<1000x1x1xf32> to memref<40x1x1xf32, affine_map<(d0, d1, d2)[s0] -> (d0 + s0 + d1 + d2)>>
  %1 = vector.transfer_read %0[%ii, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<40x1x1xf32, affine_map<(d0, d1, d2)[s0] -> (d0 + s0 + d1 + d2)>>, vector<4x1x1xf32>
  return %1 : vector<4x1x1xf32>
}
//      CHECK: func @contiguous_inner_most_dim_bounds_2d(%[[SRC:.+]]: memref<1000x1x1xf32>, %[[II:.+]]: index, %[[J:.+]]: index) -> vector<4x1x1xf32>
//      CHECK:   %[[SRC_0:.+]] = memref.subview %[[SRC]]
//      CHECK:   %[[SRC_1:.+]] = memref.subview %[[SRC_0]]
//      CHECK:   %[[V:.+]] = vector.transfer_read %[[SRC_1]]
// CHECK-SAME:       {in_bounds = [true]}
// CHECK-SAME:       vector<4xf32>
