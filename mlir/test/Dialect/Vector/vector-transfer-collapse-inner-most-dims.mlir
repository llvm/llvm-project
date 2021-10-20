// RUN: mlir-opt %s -test-vector-transfer-collapse-inner-most-dims -split-input-file | FileCheck %s

#map1 = affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 3072 + s0 + d1 * 8 + d2 + d3)>
func @contiguous_inner_most_view(%in: memref<1x1x8x1xf32, #map1>) -> vector<1x8x1xf32>{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %0 = vector.transfer_read %in[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x1x8x1xf32, #map1>, vector<1x8x1xf32>
  return %0 : vector<1x8x1xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 3072 + s0 + d1 * 8 + d2 + d3)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2)[s0] -> (d0 * 3072 + s0 + d1 * 8 + d2)>
//      CHECK: func @contiguous_inner_most_view(%[[SRC:.+]]: memref<1x1x8x1xf32, #[[MAP0]]>
//      CHECK:   %[[SRC_0:.+]] = memref.subview %[[SRC]]
// CHECK-SAME:    memref<1x1x8x1xf32, #[[MAP0]]> to memref<1x1x8xf32, #[[MAP1]]>
//      CHECK:   %[[VEC:.+]] = vector.transfer_read %[[SRC_0]]
// CHECK-SAME:    memref<1x1x8xf32, #[[MAP1]]>, vector<1x8xf32>
//      CHECK:   %[[RESULT:.+]] = vector.shape_cast %[[VEC]]
//      CHECK:   return %[[RESULT]]

// -----

func @contiguous_inner_most_dim(%A: memref<16x1xf32>, %i:index, %j:index) -> (vector<8x1xf32>) {
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
