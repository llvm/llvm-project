// RUN: mlir-opt %s -test-vector-to-vector-conversion | FileCheck %s

// CHECK-LABEL: func @add4x2
//      CHECK: %[[V1:.*]] = vector.strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[V2:.*]] = vector.strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[V3:.*]] = vector.strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[V4:.*]] = vector.strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[A1:.*]] = addf %[[V1]], %[[V3]] : vector<2x2xf32>
// CHECK-NEXT: %[[A2:.*]] = addf %[[V2]], %[[V4]] : vector<2x2xf32>
// CHECK-NEXT: %[[R1:.*]] = vector.insert_strided_slice %[[A1]], %{{.*}}  {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x2xf32>
// CHECK-NEXT: %[[R2:.*]] = vector.insert_strided_slice %[[A2]], %[[R1]]  {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x2xf32>
// CHECK-NEXT: return %[[R2:.*]] : vector<4x2xf32>
func @add4x2(%0: vector<4x2xf32>) -> vector<4x2xf32> {
  %1 = addf %0, %0: vector<4x2xf32>
  return %1: vector<4x2xf32>
}

// CHECK-LABEL: func @add4x4
//      CHECK: vector.strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: vector.strided_slice %{{.*}} {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: vector.strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: vector.strided_slice %{{.*}} {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: vector.strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: vector.strided_slice %{{.*}} {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: vector.strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: vector.strided_slice %{{.*}} {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: addf %{{.*}}, %{{.*}} : vector<2x2xf32>
// CHECK-NEXT: addf %{{.*}}, %{{.*}} : vector<2x2xf32>
// CHECK-NEXT: addf %{{.*}}, %{{.*}} : vector<2x2xf32>
// CHECK-NEXT: addf %{{.*}}, %{{.*}} : vector<2x2xf32>
// CHECK-NEXT: vector.strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: vector.strided_slice %{{.*}} {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: vector.strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: vector.strided_slice %{{.*}} {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[A1:.*]] = addf %{{.*}}, %{{.*}} : vector<2x2xf32>
// CHECK-NEXT: %[[A2:.*]] = addf %{{.*}}, %{{.*}} : vector<2x2xf32>
// CHECK-NEXT: %[[A3:.*]] = addf %{{.*}}, %{{.*}} : vector<2x2xf32>
// CHECK-NEXT: %[[A4:.*]] = addf %{{.*}}, %{{.*}} : vector<2x2xf32>
// CHECK-NEXT: %[[R1:.*]] = vector.insert_strided_slice %[[A1]], %{{.*}}  {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// CHECK-NEXT: %[[R2:.*]] = vector.insert_strided_slice %[[A2]], %[[R1]]  {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// CHECK-NEXT: %[[R3:.*]] = vector.insert_strided_slice %[[A3]], %[[R2]]  {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// CHECK-NEXT: %[[R4:.*]] = vector.insert_strided_slice %[[A4]], %[[R3]]  {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// CHECK-NEXT: return %[[R4:.*]] : vector<4x4xf32>

func @add4x4(%0: vector<4x4xf32>, %1: vector<4x4xf32>) -> vector<4x4xf32> {
  %2 = addf %0, %1: vector<4x4xf32>
  %3 = addf %1, %2: vector<4x4xf32>
  return %3: vector<4x4xf32>
}


#contraction_accesses0 = [
  (i, j, k) -> (i, k),
  (i, j, k) -> (k, j),
  (i, j, k) -> (i, j)
]
#contraction_trait0 = {
  indexing_maps = #contraction_accesses0,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL: func @contraction4x4_ijk

// Reducing output vector [0, 0]

// CHECK:       %[[A0S00:.*]] = vector.strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x6xf32> to vector<2x2xf32>
// CHECK-NEXT:  %[[A1S00:.*]] = vector.strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<6x4xf32> to vector<2x2xf32>
// CHECK-NEXT:  %[[R0S00:.*]] = vector.strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[R1S00:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %[[A0S00]], %[[A1S00]], %[[R0S00]], %{{.*}}, %{{.*}} : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
// CHECK-NEXT:  %[[A0S02:.*]] = vector.strided_slice %{{.*}} {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x6xf32> to vector<2x2xf32>
// CHECK-NEXT:  %[[A1S20:.*]] = vector.strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<6x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[R2S00:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %[[A0S02]], %[[A1S20]], %[[R1S00]], %{{.*}}, %{{.*}} : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
// CHECK-NEXT:  %[[A0S04:.*]] = vector.strided_slice %{{.*}} {offsets = [0, 4], sizes = [2, 2], strides = [1, 1]} : vector<4x6xf32> to vector<2x2xf32>
// CHECK-NEXT:  %[[A1S40:.*]] = vector.strided_slice %{{.*}} {offsets = [4, 0], sizes = [2, 2], strides = [1, 1]} : vector<6x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[R3S00:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %[[A0S04]], %[[A1S40]], %[[R2S00]], %{{.*}}, %{{.*}} : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// Reducing output vector [0, 2]

// CHECK-NEXT:  %[[A1S02:.*]] = vector.strided_slice %{{.*}} {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<6x4xf32> to vector<2x2xf32>
// CHECK-NEXT:  %[[R0S02:.*]] = vector.strided_slice %{{.*}} {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT:  %[[R1S02:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %[[A0S00]], %[[A1S02]], %[[R0S02]], %{{.*}}, %{{.*}} : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
// CHECK-NEXT:  %[[A1S22:.*]] = vector.strided_slice %{{.*}} {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<6x4xf32> to vector<2x2xf32>
// CHECK-NEXT:  %[[R2S02:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %[[A0S02]], %[[A1S22]], %[[R1S02]], %{{.*}}, %{{.*}} : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
// CHECK-NEXT:  %[[A1S42:.*]] = vector.strided_slice %{{.*}} {offsets = [4, 2], sizes = [2, 2], strides = [1, 1]} : vector<6x4xf32> to vector<2x2xf32>
// CHECK-NEXT:  %[[R3S02:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %[[A0S04]], %[[A1S42]], %[[R2S02]], %{{.*}}, %{{.*}} : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// Reducing output vector [2, 0]

// CHECK-NEXT:  %[[A0S20:.*]] = vector.strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x6xf32> to vector<2x2xf32>
// CHECK-NEXT:  %[[R0S20:.*]] = vector.strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT:  %[[R1S20:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %[[A0S20]], %[[A1S00]], %[[R0S20]], %{{.*}}, %{{.*}} : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
// CHECK-NEXT:  %[[A0S22:.*]] = vector.strided_slice %{{.*}} {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x6xf32> to vector<2x2xf32>
// CHECK-NEXT:  %[[R2S20:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %[[A0S22]], %[[A1S20]], %[[R1S20]], %{{.*}}, %{{.*}} : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
// CHECK-NEXT:  %[[A0S24:.*]] = vector.strided_slice %{{.*}} {offsets = [2, 4], sizes = [2, 2], strides = [1, 1]} : vector<4x6xf32> to vector<2x2xf32>
// CHECK-NEXT:  %[[R3S20:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %[[A0S24]], %[[A1S40]], %[[R2S20]], %{{.*}}, %{{.*}} : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// Reducing output vector [2, 2]

// CHECK-NEXT:  %[[R0S22:.*]] = vector.strided_slice %{{.*}} {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT:  %[[R1S22:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %[[A0S20]], %[[A1S02]], %[[R0S22]], %{{.*}}, %{{.*}} : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
// CHECK-NEXT:  %[[R2S22:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %[[A0S22]], %[[A1S22]], %[[R1S22]], %{{.*}}, %{{.*}} : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
// CHECK-NEXT:  %[[R3S22:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %[[A0S24]], %[[A1S42]], %[[R2S22]], %{{.*}}, %{{.*}} : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// Insert output vector slices into 4x4 vector result.
// CHECK-NEXT:  %[[RES0:.*]] = vector.insert_strided_slice %[[R3S00]], %{{.*}} {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// CHECK-NEXT:  %[[RES1:.*]] = vector.insert_strided_slice %[[R3S02]], %[[RES0]] {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// CHECK-NEXT:  %[[RES2:.*]] = vector.insert_strided_slice %[[R3S20]], %[[RES1]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// CHECK-NEXT:  %[[RES3:.*]] = vector.insert_strided_slice %[[R3S22]], %[[RES2]] {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// CHECK-NEXT:  return %[[RES3]] : vector<4x4xf32>

func @contraction4x4_ijk(%arg0 : vector<4x6xf32>, %arg1 : vector<6x4xf32>,
                         %arg2 : vector<4x4xf32>, %arg3 : index)
                         -> (vector<4x4xf32>) {

  %lhsm = vector.make_index_tuple %arg3, %arg3 : tuple<index, index>
  %rhsm = vector.make_index_tuple %arg3, %arg3 : tuple<index, index>
  %0 = vector.contract #contraction_trait0 %arg0, %arg1, %arg2, %lhsm, %rhsm
      : vector<4x6xf32>, vector<6x4xf32> into vector<4x4xf32>

  return %0 : vector<4x4xf32>
}


#contraction_accesses1 = [
  (i, k, j) -> (i, k),
  (i, k, j) -> (k, j),
  (i, k, j) -> (i, j)
]
#contraction_trait1 = {
  indexing_maps = #contraction_accesses1,
  iterator_types = ["parallel", "reduction", "parallel"]
}

// CHECK-LABEL: func @contraction4x4_ikj

// Reducing output vector [0, 0]

// CHECK:       %[[A0S00:.*]] = vector.strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf32> to vector<2x2xf32>
// CHECK-NEXT:  %[[A1S00:.*]] = vector.strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<2x4xf32> to vector<2x2xf32>
// CHECK-NEXT:  %[[R0S00:.*]] = vector.strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT:  %[[R1S00:.*]] = vector.contract {indexing_maps = [#map2, #map3, #map0], iterator_types = ["parallel", "reduction", "parallel"]} %[[A0S00]], %[[A1S00]], %[[R0S00]], %{{.*}}, %{{.*}} : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// Reducing output vector [0, 2]

// CHECK-NEXT:  %[[A1S02:.*]] = vector.strided_slice %{{.*}} {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<2x4xf32> to vector<2x2xf32>
// CHECK-NEXT:  %[[R0S02:.*]] = vector.strided_slice %{{.*}} {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT:  %[[R1S02:.*]] = vector.contract {indexing_maps = [#map2, #map3, #map0], iterator_types = ["parallel", "reduction", "parallel"]} %[[A0S00]], %[[A1S02]], %[[R0S02]], %{{.*}}, %{{.*}} : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// Reducing output vector [2, 0]

// CHECK-NEXT:  %[[A0S20:.*]] = vector.strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf32> to vector<2x2xf32>
// CHECK-NEXT:  %[[R0S20:.*]] = vector.strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT:  %[[R1S20:.*]] = vector.contract {indexing_maps = [#map2, #map3, #map0], iterator_types = ["parallel", "reduction", "parallel"]} %[[A0S20]], %[[A1S00]], %[[R0S20]], %{{.*}}, %{{.*}} : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// Reducing output vector [2, 2]

// CHECK-NEXT:  %[[R0S22:.*]] = vector.strided_slice %{{.*}} {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT:  %[[R1S22:.*]] = vector.contract {indexing_maps = [#map2, #map3, #map0], iterator_types = ["parallel", "reduction", "parallel"]} %[[A0S20]], %[[A1S02]], %[[R0S22]], %{{.*}}, %{{.*}} : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// Insert output vector slices into 4x4 vector result.
// CHECK-NEXT:  %[[RES0:.*]] = vector.insert_strided_slice %[[R1S00]], %{{.*}} {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// CHECK-NEXT:  %[[RES1:.*]] = vector.insert_strided_slice %[[R1S02]], %[[RES0]] {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// CHECK-NEXT:  %[[RES2:.*]] = vector.insert_strided_slice %[[R1S20]], %[[RES1]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// CHECK-NEXT:  %[[RES3:.*]] = vector.insert_strided_slice %[[R1S22]], %[[RES2]] {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// CHECK-NEXT:  return %[[RES3]] : vector<4x4xf32>

func @contraction4x4_ikj(%arg0 : vector<4x2xf32>, %arg1 : vector<2x4xf32>,
                         %arg2 : vector<4x4xf32>, %arg3 : index)
                         -> (vector<4x4xf32>) {

  %lhsm = vector.make_index_tuple %arg3, %arg3 : tuple<index, index>
  %rhsm = vector.make_index_tuple %arg3, %arg3 : tuple<index, index>
  %0 = vector.contract #contraction_trait1 %arg0, %arg1, %arg2, %lhsm, %rhsm
      : vector<4x2xf32>, vector<2x4xf32> into vector<4x4xf32>

  return %0 : vector<4x4xf32>
}

// CHECK-LABEL: func @vector_transfers
// CHECK-COUNT-2: vector.transfer_read
// CHECK-COUNT-8: vector.strided_slice
// CHECK-COUNT-4: addf
// CHECK-COUNT-4: vector.insert_strided_slice
//         CHECK: vector.transfer_write

func @vector_transfers(%arg0: index, %arg1: index) {
  %cst = constant 0.000000e+00 : f32
  %0 = alloc(%arg0, %arg1) : memref<?x?xf32>
  %1 = alloc(%arg0, %arg1) : memref<?x?xf32>
  %2 = alloc(%arg0, %arg1) : memref<?x?xf32>
  %cst_0 = constant 1.000000e+00 : f32
  %cst_1 = constant 2.000000e+00 : f32
  affine.for %arg2 = 0 to %arg0 step 4 {
    affine.for %arg3 = 0 to %arg1 step 4 {
      %4 = vector.transfer_read %0[%arg2, %arg3], %cst  {permutation_map = (d0, d1) -> (d0, d1)} : memref<?x?xf32>, vector<4x4xf32>
      %5 = vector.transfer_read %1[%arg2, %arg3], %cst  {permutation_map = (d0, d1) -> (d0, d1)} : memref<?x?xf32>, vector<4x4xf32>
      %6 = addf %4, %5 : vector<4x4xf32>
      vector.transfer_write %6, %2[%arg2, %arg3] {permutation_map = (d0, d1) -> (d0, d1)} : vector<4x4xf32>, memref<?x?xf32>
    }
  }
  return
}
