// RUN: mlir-opt %s -test-vector-to-vector-lowering="unroll" | FileCheck %s

// CHECK-DAG: #[[MAP1:map[0-9]+]] = affine_map<(d0, d1, d2) -> (d1, d2)>

// CHECK-LABEL: func @add4x2
//      CHECK: %[[S1:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S2:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[A1:.*]] = arith.addf %[[S1]], %[[S2]] : vector<2x2xf32>
// CHECK-NEXT: %[[VEC0:.*]] = vector.insert_strided_slice %[[A1]], %{{.*}} {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x2xf32>
// CHECK-NEXT: %[[S3:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S4:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[A2:.*]] = arith.addf %[[S3]], %[[S4]] : vector<2x2xf32>
// CHECK-NEXT: %[[VEC1:.*]] = vector.insert_strided_slice %[[A2]], %[[VEC0]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x2xf32>
// CHECK-NEXT: return %[[VEC1:.*]] : vector<4x2xf32>

func @add4x2(%0: vector<4x2xf32>) -> vector<4x2xf32> {
  %1 = arith.addf %0, %0: vector<4x2xf32>
  return %1: vector<4x2xf32>
}

// CHECK-LABEL: func @add4x4
//      CHECK: %[[S1:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S2:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>

// CHECK-NEXT: %[[A1:.*]] = arith.addf %[[S1]], %[[S2]] : vector<2x2xf32>

// CHECK-NEXT: %[[S3:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S4:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>

// CHECK-NEXT: %[[A2:.*]] = arith.addf %[[S3]], %[[S4]] : vector<2x2xf32>

// CHECK-NEXT: %[[S5:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S6:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[A3:.*]] = arith.addf %[[S5]], %[[S6]] : vector<2x2xf32>

// CHECK-NEXT: %[[S7:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S8:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[A4:.*]] = arith.addf %[[S7]], %[[S8]] : vector<2x2xf32>

// CHECK-NEXT: %[[S9:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[A5:.*]] = arith.addf %[[S9]], %[[A1]] : vector<2x2xf32>
// CHECK-NEXT: %[[R1:.*]] = vector.insert_strided_slice %[[A5]], %{{.*}} {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>


// CHECK-NEXT: %[[S11:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[A6:.*]] = arith.addf %[[S11]], %[[A2]] : vector<2x2xf32>
// CHECK-NEXT: %[[R2:.*]] = vector.insert_strided_slice %[[A6]], %[[R1]] {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>

// CHECK-NEXT: %[[S13:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[A7:.*]] = arith.addf %[[S13]], %[[A3]] : vector<2x2xf32>
// CHECK-NEXT: %[[R3:.*]] = vector.insert_strided_slice %[[A7]], %[[R2]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>

// CHECK-NEXT: %[[S15:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[A8:.*]] = arith.addf %[[S15]], %[[A4]] : vector<2x2xf32>
// CHECK-NEXT: %[[R4:.*]] = vector.insert_strided_slice %[[A8]], %[[R3]] {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>

// CHECK-NEXT: return %[[R4]] : vector<4x4xf32>

func @add4x4(%0: vector<4x4xf32>, %1: vector<4x4xf32>) -> vector<4x4xf32> {
  %2 = arith.addf %0, %1: vector<4x4xf32>
  %3 = arith.addf %1, %2: vector<4x4xf32>
  return %3: vector<4x4xf32>
}

#contraction_accesses0 = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
#contraction_trait0 = {
  indexing_maps = #contraction_accesses0,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL: func @contraction4x4_ijk

// Reducing output vector [0, 0]

//      CHECK: %[[S1:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x6xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S4:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT: %[[S2:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<6x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S5:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT: %[[S3:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>

// CHECK-NEXT: %[[R1S00:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[S1]], %[[S2]], %[[S3]], %[[S4]], %[[S5]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// CHECK-NEXT: %[[S6:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x6xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S8:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT: %[[S7:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<6x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S9:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>

// CHECK-NEXT: %[[R2S00:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[S6]], %[[S7]], %[[R1S00]], %[[S8]], %[[S9]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// CHECK-NEXT: %[[S10:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 4], sizes = [2, 2], strides = [1, 1]} : vector<4x6xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S12:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT: %[[S11:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [4, 0], sizes = [2, 2], strides = [1, 1]} : vector<6x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S13:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT: %[[R3S00:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[S10]], %[[S11]], %[[R2S00]], %[[S12]], %[[S13]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// Reducing output vector [0, 2]
// CHECK-NEXT: %[[S14:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x6xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S17:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT: %[[S15:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<6x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S18:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT: %[[S16:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[R1S02:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[S14]], %[[S15]], %[[S16]], %[[S17]], %[[S18]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// CHECK-NEXT: %[[S19:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x6xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S21:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT: %[[S20:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<6x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S22:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT: %[[R2S02:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[S19]], %[[S20]], %[[R1S02]], %[[S21]], %[[S22]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// CHECK-NEXT: %[[S23:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 4], sizes = [2, 2], strides = [1, 1]} : vector<4x6xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S25:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT: %[[S24:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [4, 2], sizes = [2, 2], strides = [1, 1]} : vector<6x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S26:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT: %[[R3S02:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[S23]], %[[S24]], %[[R2S02]], %[[S25]], %[[S26]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// Reducing output vector [2, 0]

// CHECK-NEXT: %[[S27:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x6xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S30:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT: %[[S28:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<6x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S31:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT: %[[S29:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[R1S20:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[S27]], %[[S28]], %[[S29]], %[[S30]], %[[S31]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// CHECK-NEXT: %[[S32:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x6xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S34:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT: %[[S33:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<6x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S35:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT:  %[[R2S20:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[S32]], %[[S33]], %[[R1S20]], %[[S34]], %[[S35]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// CHECK-NEXT: %[[S36:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 4], sizes = [2, 2], strides = [1, 1]} : vector<4x6xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S38:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT: %[[S37:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [4, 0], sizes = [2, 2], strides = [1, 1]} : vector<6x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S39:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT:  %[[R3S20:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[S36]], %[[S37]], %[[R2S20]], %[[S38]], %[[S39]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// Reducing output vector [2, 2]

// CHECK-NEXT: %[[S40:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x6xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S43:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT: %[[S41:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<6x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S44:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT: %[[S42:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[R1S22:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[S40]], %[[S41]], %[[S42]], %[[S43]], %[[S44]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// CHECK-NEXT: %[[S45:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x6xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S47:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT: %[[S46:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<6x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S48:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT: %[[R2S22:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[S45]], %[[S46]], %[[R1S22]], %[[S47]], %[[S48]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// CHECK-NEXT: %[[S49:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 4], sizes = [2, 2], strides = [1, 1]} : vector<4x6xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S51:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT: %[[S50:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [4, 2], sizes = [2, 2], strides = [1, 1]} : vector<6x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S52:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT: %[[R3S22:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[S49]], %[[S50]], %[[R2S22]], %[[S51]], %[[S52]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// CHECK-NEXT: %[[VEC1:.*]] = vector.insert_strided_slice %[[R3S00]], %{{.*}} {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// CHECK-NEXT: %[[VEC2:.*]] = vector.insert_strided_slice %[[R3S02]], %[[VEC1]] {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// CHECK-NEXT: %[[VEC3:.*]] = vector.insert_strided_slice %[[R3S20]], %[[VEC2]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// CHECK-NEXT: %[[VEC4:.*]] = vector.insert_strided_slice %[[R3S22]], %[[VEC3]] {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>

// CHECK-NEXT:  return %[[VEC4]] : vector<4x4xf32>

func @contraction4x4_ijk(%arg0 : vector<4x6xf32>, %arg1 : vector<6x4xf32>,
                         %arg2 : vector<4x4xf32>, %arg3 : index)
                         -> (vector<4x4xf32>) {
  %lhsm = vector.constant_mask [4, 6] : vector<4x6xi1>
  %rhsm = vector.constant_mask [6, 4] : vector<6x4xi1>
  %0 = vector.contract #contraction_trait0 %arg0, %arg1, %arg2, %lhsm, %rhsm
      : vector<4x6xf32>, vector<6x4xf32> into vector<4x4xf32>

  return %0 : vector<4x4xf32>
}

#contraction_accesses1 = [
  affine_map<(i, k, j) -> (i, k)>,
  affine_map<(i, k, j) -> (k, j)>,
  affine_map<(i, k, j) -> (i, j)>
]
#contraction_trait1 = {
  indexing_maps = #contraction_accesses1,
  iterator_types = ["parallel", "reduction", "parallel"]
}

// CHECK-LABEL: func @contraction4x4_ikj

// Reducing output vector [0, 0]

//      CHECK: %[[S1:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S4:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT: %[[S2:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<2x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S5:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT: %[[S3:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[R1S00:.*]] = vector.contract {indexing_maps = [#map2, #map3, #map0], iterator_types = ["parallel", "reduction", "parallel"], kind = #vector.kind<add>} %[[S1]], %[[S2]], %[[S3]], %[[S4]], %[[S5]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// Reducing output vector [0, 2]

// CHECK-NEXT: %[[S6:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S9:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT: %[[S7:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<2x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S10:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT: %[[S8:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[R1S02:.*]] = vector.contract {indexing_maps = [#map2, #map3, #map0], iterator_types = ["parallel", "reduction", "parallel"], kind = #vector.kind<add>} %[[S6]], %[[S7]], %[[S8]], %[[S9]], %[[S10]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// Reducing output vector [2, 0]

// CHECK-NEXT: %[[S11:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S14:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT: %[[S12:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<2x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S15:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT: %[[S13:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[R1S20:.*]] = vector.contract {indexing_maps = [#map2, #map3, #map0], iterator_types = ["parallel", "reduction", "parallel"], kind = #vector.kind<add>} %[[S11]], %[[S12]], %[[S13]], %[[S14]], %[[S15]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// Reducing output vector [2, 2]

// CHECK-NEXT: %[[S16:.*]] = vector.extract_strided_slice %arg0 {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S19:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT: %[[S17:.*]] = vector.extract_strided_slice %arg1 {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<2x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[S20:.*]] = vector.constant_mask [2, 2] : vector<2x2xi1>
// CHECK-NEXT: %[[S18:.*]] = vector.extract_strided_slice %arg2 {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[R1S22:.*]] = vector.contract {indexing_maps = [#map2, #map3, #map0], iterator_types = ["parallel", "reduction", "parallel"], kind = #vector.kind<add>} %[[S16]], %[[S17]], %[[S18]], %[[S19]], %[[S20]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// CHECK-NEXT: %[[VEC0:.*]] = vector.insert_strided_slice %[[R1S00]], %{{.*}} {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// CHECK-NEXT: %[[VEC1:.*]] = vector.insert_strided_slice %[[R1S02]], %[[VEC0]] {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// CHECK-NEXT: %[[VEC2:.*]] = vector.insert_strided_slice %[[R1S20]], %[[VEC1]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// CHECK-NEXT: %[[VEC3:.*]] = vector.insert_strided_slice %[[R1S22]], %[[VEC2]] {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// CHECK-NEXT:  return %[[VEC3]] : vector<4x4xf32>

func @contraction4x4_ikj(%arg0 : vector<4x2xf32>, %arg1 : vector<2x4xf32>,
                         %arg2 : vector<4x4xf32>, %arg3 : index)
                         -> (vector<4x4xf32>) {
  %lhsm = vector.constant_mask [4, 2] : vector<4x2xi1>
  %rhsm = vector.constant_mask [2, 4] : vector<2x4xi1>
  %0 = vector.contract #contraction_trait1 %arg0, %arg1, %arg2, %lhsm, %rhsm
      : vector<4x2xf32>, vector<2x4xf32> into vector<4x4xf32>

  return %0 : vector<4x4xf32>
}

// CHECK-LABEL: func @contraction4x4_ikj_xfer_read

// CHECK-DAG:      %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:      %[[C0:.*]] = arith.constant 0 : index

// Check LHS vector.transfer read is split for each user.

//      CHECK: %[[VTR0:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[C0]]], %{{.*}} : memref<4x2xf32>, vector<2x2xf32>
// CHECK-NEXT: %[[VTR1:.*]] = vector.transfer_read %{{.*}}[%[[C2]], %[[C0]]], %{{.*}} : memref<4x2xf32>, vector<2x2xf32>

// CHECK-NEXT: %[[VTR2:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[C0]]], %{{.*}} : memref<2x4xf32>, vector<2x2xf32>
// CHECK-NEXT: %[[VTR3:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[C2]]], %{{.*}} : memref<2x4xf32>, vector<2x2xf32>

// CHECK-NEXT: %[[VTR4:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[C0]]], %{{.*}} : memref<4x4xf32>, vector<2x2xf32>
// CHECK-NEXT: %[[VTR5:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[C2]]], %{{.*}} : memref<4x4xf32>, vector<2x2xf32>
// CHECK-NEXT: %[[VTR6:.*]] = vector.transfer_read %{{.*}}[%[[C2]], %[[C0]]], %{{.*}} : memref<4x4xf32>, vector<2x2xf32>
// CHECK-NEXT: %[[VTR7:.*]] = vector.transfer_read %{{.*}}[%[[C2]], %[[C2]]], %{{.*}} : memref<4x4xf32>, vector<2x2xf32>

// CHECK-NEXT: %[[R0:.*]] = vector.contract {indexing_maps = [#map2, #map3, #map0], iterator_types = ["parallel", "reduction", "parallel"], kind = #vector.kind<add>} %[[VTR0]], %[[VTR2]], %[[VTR4]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
// CHECK-NEXT: %[[R1:.*]] = vector.contract {indexing_maps = [#map2, #map3, #map0], iterator_types = ["parallel", "reduction", "parallel"], kind = #vector.kind<add>} %[[VTR0]], %[[VTR3]], %[[VTR5]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
// CHECK-NEXT: %[[R2:.*]] = vector.contract {indexing_maps = [#map2, #map3, #map0], iterator_types = ["parallel", "reduction", "parallel"], kind = #vector.kind<add>} %[[VTR1]], %[[VTR2]], %[[VTR6]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
// CHECK-NEXT: %[[R3:.*]] = vector.contract {indexing_maps = [#map2, #map3, #map0], iterator_types = ["parallel", "reduction", "parallel"], kind = #vector.kind<add>} %[[VTR1]], %[[VTR3]], %[[VTR7]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// CHECK-NEXT: vector.transfer_write %[[R0]], %{{.*}}[%[[C0]], %[[C0]]] {in_bounds = [true, true]} : vector<2x2xf32>, memref<4x4xf32>
// CHECK-NEXT: vector.transfer_write %[[R1]], %{{.*}}[%[[C0]], %[[C2]]] {in_bounds = [true, true]} : vector<2x2xf32>, memref<4x4xf32>
// CHECK-NEXT: vector.transfer_write %[[R2]], %{{.*}}[%[[C2]], %[[C0]]] {in_bounds = [true, true]} : vector<2x2xf32>, memref<4x4xf32>
// CHECK-NEXT: vector.transfer_write %[[R3]], %{{.*}}[%[[C2]], %[[C2]]] {in_bounds = [true, true]} : vector<2x2xf32>, memref<4x4xf32>
// CHECK-NEXT: return

func @contraction4x4_ikj_xfer_read(%arg0 : memref<4x2xf32>,
                                   %arg1 : memref<2x4xf32>,
                                   %arg2 : memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32

  %0 = vector.transfer_read %arg0[%c0, %c0], %cf0
    { permutation_map = affine_map<(d0, d1) -> (d0, d1)> }
      : memref<4x2xf32>, vector<4x2xf32>

  %1 = vector.transfer_read %arg1[%c0, %c0], %cf0
    { permutation_map = affine_map<(d0, d1) -> (d0, d1)> }
    : memref<2x4xf32>, vector<2x4xf32>

  %2 = vector.transfer_read %arg2[%c0, %c0], %cf0
    { permutation_map = affine_map<(d0, d1) -> (d0, d1)> }
      : memref<4x4xf32>, vector<4x4xf32>

  %3 = vector.contract #contraction_trait1 %0, %1, %2
      : vector<4x2xf32>, vector<2x4xf32> into vector<4x4xf32>

  vector.transfer_write %3, %arg2[%c0, %c0]
    {permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
      : vector<4x4xf32>, memref<4x4xf32>
  return
}

// TODO: Update test with VTR split transform.
// CHECK-LABEL: func @vector_transfers
// CHECK-COUNT-8: vector.transfer_read
// CHECK-COUNT-4: arith.addf
// CHECK-COUNT-4: vector.transfer_write

func @vector_transfers(%arg0: index, %arg1: index) {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
  %1 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
  %2 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
  %cst_0 = arith.constant 1.000000e+00 : f32
  %cst_1 = arith.constant 2.000000e+00 : f32
  affine.for %arg2 = 0 to %arg0 step 4 {
    affine.for %arg3 = 0 to %arg1 step 4 {
      %4 = vector.transfer_read %0[%arg2, %arg3], %cst {permutation_map = affine_map<(d0, d1) -> (d0, d1)>} : memref<?x?xf32>, vector<4x4xf32>
      %5 = vector.transfer_read %1[%arg2, %arg3], %cst {permutation_map = affine_map<(d0, d1) -> (d0, d1)>} : memref<?x?xf32>, vector<4x4xf32>
      %6 = arith.addf %4, %5 : vector<4x4xf32>
      vector.transfer_write %6, %2[%arg2, %arg3] {permutation_map = affine_map<(d0, d1) -> (d0, d1)>} : vector<4x4xf32>, memref<?x?xf32>
    }
  }
  return
}

// CHECK-LABEL: func @cancelling_shape_cast_ops
//  CHECK-SAME: %[[A0:.*0]]: vector<2x4xf32>
//       CHECK: return %[[A0]] : vector<2x4xf32>
func @cancelling_shape_cast_ops(%arg0 : vector<2x4xf32>) -> vector<2x4xf32> {
  %0 = vector.shape_cast %arg0 : vector<2x4xf32> to vector<8xf32>
  %1 = vector.shape_cast %0 : vector<8xf32> to vector<2x4xf32>
  return %1 : vector<2x4xf32>
}

// CHECK-LABEL: func @elementwise_unroll
//  CHECK-SAME: (%[[ARG0:.*]]: memref<4x4xf32>, %[[ARG1:.*]]: memref<4x4xf32>)
//       CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//       CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[VT0:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[VT1:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C2]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[VT2:.*]] = vector.transfer_read %[[ARG0]][%[[C2]], %[[C0]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[VT3:.*]] = vector.transfer_read %[[ARG0]][%[[C2]], %[[C2]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[VT4:.*]] = vector.transfer_read %[[ARG1]][%[[C0]], %[[C0]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[VT5:.*]] = vector.transfer_read %[[ARG1]][%[[C0]], %[[C2]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[VT6:.*]] = vector.transfer_read %[[ARG1]][%[[C2]], %[[C0]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[VT7:.*]] = vector.transfer_read %[[ARG1]][%[[C2]], %[[C2]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[CMP0:.*]] = arith.cmpf ult, %[[VT0]], %[[VT4]] : vector<2x2xf32>
//       CHECK:   %[[CMP1:.*]] = arith.cmpf ult, %[[VT1]], %[[VT5]] : vector<2x2xf32>
//       CHECK:   %[[CMP2:.*]] = arith.cmpf ult, %[[VT2]], %[[VT6]] : vector<2x2xf32>
//       CHECK:   %[[CMP3:.*]] = arith.cmpf ult, %[[VT3]], %[[VT7]] : vector<2x2xf32>
//       CHECK:   %[[VT0:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[VT1:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C2]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[VT2:.*]] = vector.transfer_read %[[ARG0]][%[[C2]], %[[C0]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[VT3:.*]] = vector.transfer_read %[[ARG0]][%[[C2]], %[[C2]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[VT4:.*]] = vector.transfer_read %[[ARG1]][%[[C0]], %[[C0]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[VT5:.*]] = vector.transfer_read %[[ARG1]][%[[C0]], %[[C2]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[VT6:.*]] = vector.transfer_read %[[ARG1]][%[[C2]], %[[C0]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[VT7:.*]] = vector.transfer_read %[[ARG1]][%[[C2]], %[[C2]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[SEL0:.*]] = arith.select %[[CMP0]], %[[VT0]], %[[VT4]] : vector<2x2xi1>, vector<2x2xf32>
//       CHECK:   %[[SEL1:.*]] = arith.select %[[CMP1]], %[[VT1]], %[[VT5]] : vector<2x2xi1>, vector<2x2xf32>
//       CHECK:   %[[SEL2:.*]] = arith.select %[[CMP2]], %[[VT2]], %[[VT6]] : vector<2x2xi1>, vector<2x2xf32>
//       CHECK:   %[[SEL3:.*]] = arith.select %[[CMP3]], %[[VT3]], %[[VT7]] : vector<2x2xi1>, vector<2x2xf32>
//       CHECK:   vector.transfer_write %[[SEL0]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
//       CHECK:   vector.transfer_write %[[SEL1]], %[[ARG0]][%[[C0]], %[[C2]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
//       CHECK:   vector.transfer_write %[[SEL2]], %[[ARG0]][%[[C2]], %[[C0]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
//       CHECK:   vector.transfer_write %[[SEL3]], %[[ARG0]][%[[C2]], %[[C2]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
func @elementwise_unroll(%arg0 : memref<4x4xf32>, %arg1 : memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0], %cf0 : memref<4x4xf32>, vector<4x4xf32>
  %1 = vector.transfer_read %arg1[%c0, %c0], %cf0 : memref<4x4xf32>, vector<4x4xf32>
  %cond = arith.cmpf ult, %0, %1 : vector<4x4xf32>
  // Vector transfer split pattern only support single user right now.
  %2 = vector.transfer_read %arg0[%c0, %c0], %cf0 : memref<4x4xf32>, vector<4x4xf32>
  %3 = vector.transfer_read %arg1[%c0, %c0], %cf0 : memref<4x4xf32>, vector<4x4xf32>
  %4 = arith.select %cond, %2, %3 : vector<4x4xi1>, vector<4x4xf32>
  vector.transfer_write %4, %arg0[%c0, %c0] : vector<4x4xf32>, memref<4x4xf32>
  return
}

// Check that vector.transfer read/write are split based on contract unrolling.
//      CHECK: %[[VTR0:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[C0]]], %{{.*}} : tensor<4x2xf32>, vector<2x2xf32>
// CHECK-NEXT: %[[VTR1:.*]] = vector.transfer_read %{{.*}}[%[[C2]], %[[C0]]], %{{.*}} : tensor<4x2xf32>, vector<2x2xf32>

// CHECK-NEXT: %[[VTR2:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[C0]]], %{{.*}} : tensor<2x4xf32>, vector<2x2xf32>
// CHECK-NEXT: %[[VTR3:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[C2]]], %{{.*}} : tensor<2x4xf32>, vector<2x2xf32>

// CHECK-NEXT: %[[VTR4:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[C0]]], %{{.*}} : tensor<4x4xf32>, vector<2x2xf32>
// CHECK-NEXT: %[[VTR5:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[C2]]], %{{.*}} : tensor<4x4xf32>, vector<2x2xf32>
// CHECK-NEXT: %[[VTR6:.*]] = vector.transfer_read %{{.*}}[%[[C2]], %[[C0]]], %{{.*}} : tensor<4x4xf32>, vector<2x2xf32>
// CHECK-NEXT: %[[VTR7:.*]] = vector.transfer_read %{{.*}}[%[[C2]], %[[C2]]], %{{.*}} : tensor<4x4xf32>, vector<2x2xf32>

// CHECK-NEXT: %[[R0:.*]] = vector.contract {indexing_maps = [#map2, #map3, #map0], iterator_types = ["parallel", "reduction", "parallel"], kind = #vector.kind<add>} %[[VTR0]], %[[VTR2]], %[[VTR4]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
// CHECK-NEXT: %[[R1:.*]] = vector.contract {indexing_maps = [#map2, #map3, #map0], iterator_types = ["parallel", "reduction", "parallel"], kind = #vector.kind<add>} %[[VTR0]], %[[VTR3]], %[[VTR5]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
// CHECK-NEXT: %[[R2:.*]] = vector.contract {indexing_maps = [#map2, #map3, #map0], iterator_types = ["parallel", "reduction", "parallel"], kind = #vector.kind<add>} %[[VTR1]], %[[VTR2]], %[[VTR6]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
// CHECK-NEXT: %[[R3:.*]] = vector.contract {indexing_maps = [#map2, #map3, #map0], iterator_types = ["parallel", "reduction", "parallel"], kind = #vector.kind<add>} %[[VTR1]], %[[VTR3]], %[[VTR7]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// CHECK-NEXT: %[[VTW0:.*]] = vector.transfer_write %[[R0]], %{{.*}}[%[[C0]], %[[C0]]] {in_bounds = [true, true]} : vector<2x2xf32>, tensor<4x4xf32>
// CHECK-NEXT: %[[VTW1:.*]] = vector.transfer_write %[[R1]], %[[VTW0]][%[[C0]], %[[C2]]] {in_bounds = [true, true]} : vector<2x2xf32>, tensor<4x4xf32>
// CHECK-NEXT: %[[VTW2:.*]] = vector.transfer_write %[[R2]], %[[VTW1]][%[[C2]], %[[C0]]] {in_bounds = [true, true]} : vector<2x2xf32>, tensor<4x4xf32>
// CHECK-NEXT: %[[VTW3:.*]] = vector.transfer_write %[[R3]], %[[VTW2]][%[[C2]], %[[C2]]] {in_bounds = [true, true]} : vector<2x2xf32>, tensor<4x4xf32>
// CHECK-NEXT: return %[[VTW3]] : tensor<4x4xf32>

func @contraction4x4_ikj_xfer_read_tensor(%arg0 : tensor<4x2xf32>,
                                          %arg1 : tensor<2x4xf32>,
                                          %arg2 : tensor<4x4xf32>) ->
  tensor<4x4xf32> {
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0], %cf0 :
    tensor<4x2xf32>, vector<4x2xf32>
  %1 = vector.transfer_read %arg1[%c0, %c0], %cf0 :
    tensor<2x4xf32>, vector<2x4xf32>
  %2 = vector.transfer_read %arg2[%c0, %c0], %cf0 :
    tensor<4x4xf32>, vector<4x4xf32>
  %3 = vector.contract #contraction_trait1 %0, %1, %2
      : vector<4x2xf32>, vector<2x4xf32> into vector<4x4xf32>
  %r = vector.transfer_write %3, %arg2[%c0, %c0]
      : vector<4x4xf32>, tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}

// CHECK-LABEL: func @bubble_down_bitcast_in_extract
//  CHECK-SAME: %[[SRC:.+]]: vector<4xf32>
func @bubble_down_bitcast_in_extract(%src: vector<4xf32>) -> (f16, f16) {
  %0 = vector.bitcast %src : vector<4xf32> to vector<8xf16>
  // CHECK: %[[EXTRACT1:.+]] = vector.extract %[[SRC]][1] : vector<4xf32>
  // CHECK:    %[[CAST1:.+]] = vector.bitcast %[[EXTRACT1]] : vector<1xf32> to vector<2xf16>
  // CHECK: %[[EXTRACT2:.+]] = vector.extract %[[CAST1]][1] : vector<2xf16>
  %1 = vector.extract %0[3] : vector<8xf16>
  // CHECK: %[[EXTRACT3:.+]] = vector.extract %[[SRC]][2] : vector<4xf32>
  // CHECK:    %[[CAST2:.+]] = vector.bitcast %[[EXTRACT3]] : vector<1xf32> to vector<2xf16>
  // CHECK: %[[EXTRACT4:.+]] = vector.extract %[[CAST2]][0] : vector<2xf16>
  %2 = vector.extract %0[4] : vector<8xf16>
  // CHECK: return %[[EXTRACT2]], %[[EXTRACT4]]
  return %1, %2: f16, f16
}

// CHECK-LABEL: func @bubble_down_bitcast_in_strided_slice_extract
//  CHECK-SAME: %[[SRC:.+]]: vector<4xf32>
func @bubble_down_bitcast_in_strided_slice_extract(%arg0: vector<4xf32>) -> vector<4xf16> {
  // CHECK: %[[EXTRACT:.+]] = vector.extract_strided_slice %[[SRC]] {offsets = [2], sizes = [2], strides = [1]} : vector<4xf32> to vector<2xf32>
  // CHECK: %[[CAST:.+]] = vector.bitcast %[[EXTRACT]] : vector<2xf32> to vector<4xf16>
  %cast = vector.bitcast %arg0: vector<4xf32> to vector<8xf16>
  %0 = vector.extract_strided_slice %cast {offsets = [4], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
  // CHECK: return %[[CAST]]
  return %0: vector<4xf16>
}

// CHECK-LABEL: func @bubble_down_bitcast_in_strided_slice_extract_full_last_dim
//  CHECK-SAME: %[[SRC:.+]]: vector<4x2xf32>
func @bubble_down_bitcast_in_strided_slice_extract_full_last_dim(%arg0: vector<4x2xf32>) -> vector<2x4xf16> {
  // CHECK: %[[EXTRACT:.+]] = vector.extract_strided_slice %[[SRC]] {offsets = [1], sizes = [2], strides = [1]} : vector<4x2xf32> to vector<2x2xf32>
  // CHECK: %[[CAST:.+]] = vector.bitcast %[[EXTRACT]] : vector<2x2xf32> to vector<2x4xf16>
  %cast = vector.bitcast %arg0: vector<4x2xf32> to vector<4x4xf16>
  %0 = vector.extract_strided_slice %cast {offsets = [1], sizes = [2], strides = [1]} : vector<4x4xf16> to vector<2x4xf16>
  // CHECK: return %[[CAST]]
  return %0: vector<2x4xf16>
}

// CHECK-LABEL: func @bubble_down_bitcast_in_strided_slice_extract_odd_offset
func @bubble_down_bitcast_in_strided_slice_extract_odd_offset(%arg0: vector<4xf32>) -> vector<4xf16> {
  // CHECK: vector.bitcast
  // CHECK-NEXT: vector.extract_strided_slice
  %cast = vector.bitcast %arg0: vector<4xf32> to vector<8xf16>
  %0 = vector.extract_strided_slice %cast {offsets = [3], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
  return %0: vector<4xf16>
}

// CHECK-LABEL: func @bubble_down_bitcast_in_strided_slice_extract_odd_size
func @bubble_down_bitcast_in_strided_slice_extract_odd_size(%arg0: vector<4xf32>) -> vector<3xf16> {
  // CHECK: vector.bitcast
  // CHECK-NEXT: vector.extract_strided_slice
  %cast = vector.bitcast %arg0: vector<4xf32> to vector<8xf16>
  %0 = vector.extract_strided_slice %cast {offsets = [0], sizes = [3], strides = [1]} : vector<8xf16> to vector<3xf16>
  return %0: vector<3xf16>
}

// CHECK-LABEL: func @bubble_up_bitcast_in_strided_slice_insert
//  CHECK-SAME: (%[[DST:.+]]: vector<8xf16>, %[[SRC1:.+]]: vector<4xf16>, %[[SRC2:.+]]: vector<4xf16>)
func @bubble_up_bitcast_in_strided_slice_insert(%dst: vector<8xf16>, %src1: vector<4xf16>, %src2: vector<4xf16>) -> vector<4xf32> {
  // CHECK-DAG: %[[CAST_SRC1:.+]] = vector.bitcast %[[SRC1]] : vector<4xf16> to vector<2xf32>
  // CHECK-DAG: %[[CAST_SRC2:.+]] = vector.bitcast %[[SRC2]] : vector<4xf16> to vector<2xf32>
  // CHECK-DAG: %[[CAST_DST:.+]] = vector.bitcast %[[DST]] : vector<8xf16> to vector<4xf32>
  // CHECK: %[[INSERT1:.+]] = vector.insert_strided_slice %[[CAST_SRC1]], %[[CAST_DST]] {offsets = [0], strides = [1]} : vector<2xf32> into vector<4xf32>
  // CHECK: %[[INSERT2:.+]] = vector.insert_strided_slice %[[CAST_SRC2]], %[[INSERT1]] {offsets = [2], strides = [1]} : vector<2xf32> into vector<4xf32>
  %0 = vector.insert_strided_slice %src1, %dst {offsets = [0], strides = [1]} : vector<4xf16> into vector<8xf16>
  %1 = vector.insert_strided_slice %src2, %0   {offsets = [4], strides = [1]} : vector<4xf16> into vector<8xf16>
  %cast = vector.bitcast %1: vector<8xf16> to vector<4xf32>
  // CHECK: return %[[INSERT2]]
  return %cast: vector<4xf32>
}

// CHECK-LABEL: func @bubble_up_bitcast_in_strided_slice_insert_odd_offset
func @bubble_up_bitcast_in_strided_slice_insert_odd_offset(%dst: vector<8xf16>, %src: vector<4xf16>) -> vector<4xf32> {
  // CHECK: vector.insert_strided_slice
  // CHECK-NEXT: vector.bitcast
  %0 = vector.insert_strided_slice %src, %dst {offsets = [3], strides = [1]} : vector<4xf16> into vector<8xf16>
  %cast = vector.bitcast %0: vector<8xf16> to vector<4xf32>
  return %cast: vector<4xf32>
}

// CHECK-LABEL: func @bubble_up_bitcast_in_strided_slice_insert_different_rank
func @bubble_up_bitcast_in_strided_slice_insert_different_rank(%dst: vector<16x4x8xf16>, %src: vector<2x4xf16>) -> vector<16x4x4xf32> {
  // CHECK: vector.insert_strided_slice
  // CHECK-NEXT: vector.bitcast
  %0 = vector.insert_strided_slice %src, %dst {offsets = [0, 0, 2], strides = [1, 1]} : vector<2x4xf16> into vector<16x4x8xf16>
  %cast = vector.bitcast %0: vector<16x4x8xf16> to vector<16x4x4xf32>
  return %cast: vector<16x4x4xf32>
}
