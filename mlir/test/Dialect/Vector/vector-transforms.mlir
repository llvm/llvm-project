// RUN: mlir-opt %s -test-vector-to-vector-conversion="unroll" | FileCheck %s

// CHECK-DAG: #[[MAP1:map[0-9]+]] = affine_map<(d0, d1, d2) -> (d1, d2)>

// CHECK-LABEL: func @add4x2
//      CHECK: %[[ES1:.*]] = vector.extract_slices %{{.*}}, [2, 2], [1, 1] : vector<4x2xf32> into tuple<vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[ES2:.*]] = vector.extract_slices %{{.*}}, [2, 2], [1, 1] : vector<4x2xf32> into tuple<vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG1:.*]] = vector.tuple_get %[[ES1]], 0 : tuple<vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG2:.*]] = vector.tuple_get %[[ES2]], 0 : tuple<vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[A1:.*]] = addf %[[TG1]], %[[TG2]] : vector<2x2xf32>
// CHECK-NEXT: %[[TG3:.*]] = vector.tuple_get %[[ES1]], 1 : tuple<vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG4:.*]] = vector.tuple_get %[[ES2]], 1 : tuple<vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[A2:.*]] = addf %[[TG3]], %[[TG4]] : vector<2x2xf32>
// CHECK-NEXT: %[[R1:.*]] = vector.tuple %[[A1]], %[[A2]] : vector<2x2xf32>, vector<2x2xf32>
// CHECK-NEXT: %[[R2:.*]] = vector.insert_slices %[[R1]], [2, 2], [1, 1] : tuple<vector<2x2xf32>, vector<2x2xf32>> into vector<4x2xf32>
// CHECK-NEXT: return %[[R2:.*]] : vector<4x2xf32>

func @add4x2(%0: vector<4x2xf32>) -> vector<4x2xf32> {
  %1 = addf %0, %0: vector<4x2xf32>
  return %1: vector<4x2xf32>
}

// CHECK-LABEL: func @add4x4
//      CHECK: %[[ES1:.*]] = vector.extract_slices %{{.*}}, [2, 2], [1, 1] : vector<4x4xf32> into tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[ES2:.*]] = vector.extract_slices %{{.*}}, [2, 2], [1, 1] : vector<4x4xf32> into tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>

// CHECK-NEXT: %[[TG1:.*]] = vector.tuple_get %[[ES1]], 0 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG2:.*]] = vector.tuple_get %[[ES2]], 0 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[A1:.*]] = addf %[[TG1]], %[[TG2]] : vector<2x2xf32>

// CHECK-NEXT: %[[TG3:.*]] = vector.tuple_get %[[ES1]], 1 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG4:.*]] = vector.tuple_get %[[ES2]], 1 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[A2:.*]] = addf %[[TG3]], %[[TG4]] : vector<2x2xf32>

// CHECK-NEXT: %[[TG5:.*]] = vector.tuple_get %[[ES1]], 2 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG6:.*]] = vector.tuple_get %[[ES2]], 2 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[A3:.*]] = addf %[[TG5]], %[[TG6]] : vector<2x2xf32>

// CHECK-NEXT: %[[TG7:.*]] = vector.tuple_get %[[ES1]], 3 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG8:.*]] = vector.tuple_get %[[ES2]], 3 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[A4:.*]] = addf %[[TG7]], %[[TG8]] : vector<2x2xf32>

// CHECK-NEXT: %[[ES3:.*]] = vector.extract_slices %{{.*}}, [2, 2], [1, 1] : vector<4x4xf32> into tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>

// CHECK-NEXT: %[[TG9:.*]] = vector.tuple_get %[[ES3]], 0 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[A5:.*]] = addf %[[TG9]], %[[A1]] : vector<2x2xf32>

// CHECK-NEXT: %[[TG11:.*]] = vector.tuple_get %[[ES3]], 1 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[A6:.*]] = addf %[[TG11]], %[[A2]] : vector<2x2xf32>

// CHECK-NEXT: %[[TG13:.*]] = vector.tuple_get %[[ES3]], 2 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[A7:.*]] = addf %[[TG13]], %[[A3]] : vector<2x2xf32>

// CHECK-NEXT: %[[TG15:.*]] = vector.tuple_get %[[ES3]], 3 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[A8:.*]] = addf %[[TG15]], %[[A4]] : vector<2x2xf32>

// CHECK-NEXT: %[[R3:.*]] = vector.tuple %[[A5]], %[[A6]], %[[A7]], %[[A8]] : vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>
// CHECK-NEXT: %[[R4:.*]] = vector.insert_slices %[[R3]], [2, 2], [1, 1] : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>> into vector<4x4xf32>
// CHECK-NEXT: return %[[R4]] : vector<4x4xf32>

func @add4x4(%0: vector<4x4xf32>, %1: vector<4x4xf32>) -> vector<4x4xf32> {
  %2 = addf %0, %1: vector<4x4xf32>
  %3 = addf %1, %2: vector<4x4xf32>
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

//      CHECK: %[[LMASK:.*]] = vector.constant_mask [4, 6] : vector<4x6xi1>
// CHECK-NEXT: %[[RMASK:.*]] = vector.constant_mask [6, 4] : vector<6x4xi1>

// Reducing output vector [0, 0]

// CHECK-NEXT: %[[ES1:.*]] = vector.extract_slices %{{.*}}, [2, 2], [1, 1] : vector<4x6xf32> into tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[ES2:.*]] = vector.extract_slices %{{.*}}, [2, 2], [1, 1] : vector<6x4xf32> into tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[ES3:.*]] = vector.extract_slices %{{.*}}, [2, 2], [1, 1] : vector<4x4xf32> into tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[ES4:.*]] = vector.extract_slices %[[LMASK]], [2, 2], [1, 1] : vector<4x6xi1> into tuple<vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>>
// CHECK-NEXT: %[[ES5:.*]] = vector.extract_slices %[[RMASK]], [2, 2], [1, 1] : vector<6x4xi1> into tuple<vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>>

// CHECK-NEXT: %[[TG1:.*]] = vector.tuple_get %[[ES1]], 0 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG2:.*]] = vector.tuple_get %[[ES2]], 0 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG3:.*]] = vector.tuple_get %[[ES3]], 0 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG4:.*]] = vector.tuple_get %[[ES4]], 0 : tuple<vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>>
// CHECK-NEXT: %[[TG5:.*]] = vector.tuple_get %[[ES5]], 0 : tuple<vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>>
// CHECK-NEXT: %[[R1S00:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[TG1]], %[[TG2]], %[[TG3]], %[[TG4]], %[[TG5]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// CHECK-NEXT: %[[TG6:.*]] = vector.tuple_get %[[ES1]], 1 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG7:.*]] = vector.tuple_get %[[ES2]], 2 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG8:.*]] = vector.tuple_get %[[ES4]], 1 : tuple<vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>>
// CHECK-NEXT: %[[TG9:.*]] = vector.tuple_get %[[ES5]], 2 : tuple<vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>>
// CHECK-NEXT: %[[R2S00:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[TG6]], %[[TG7]], %[[R1S00]], %[[TG8]], %[[TG9]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// CHECK-NEXT: %[[TG10:.*]] = vector.tuple_get %[[ES1]], 2 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG11:.*]] = vector.tuple_get %[[ES2]], 4 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG12:.*]] = vector.tuple_get %[[ES4]], 2 : tuple<vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>>
// CHECK-NEXT: %[[TG13:.*]] = vector.tuple_get %[[ES5]], 4 : tuple<vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>>
// CHECK-NEXT: %[[R3S00:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[TG10]], %[[TG11]], %[[R2S00]], %[[TG12]], %[[TG13]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// Reducing output vector [0, 2]

// CHECK-NEXT: %[[TG14:.*]] = vector.tuple_get %[[ES2]], 1 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG15:.*]] = vector.tuple_get %[[ES3]], 1 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG16:.*]] = vector.tuple_get %[[ES5]], 1 : tuple<vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>>
// CHECK-NEXT: %[[R1S02:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[TG1]], %[[TG14]], %[[TG15]], %[[TG4]], %[[TG16]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// CHECK-NEXT: %[[TG17:.*]] = vector.tuple_get %[[ES2]], 3 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG18:.*]] = vector.tuple_get %[[ES5]], 3 : tuple<vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>>
// CHECK-NEXT: %[[R2S02:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[TG6]], %[[TG17]], %[[R1S02]], %[[TG8]], %[[TG18]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// CHECK-NEXT: %[[TG19:.*]] = vector.tuple_get %[[ES2]], 5 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG20:.*]] = vector.tuple_get %[[ES5]], 5 : tuple<vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>>
// CHECK-NEXT: %[[R3S02:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[TG10]], %[[TG19]], %[[R2S02]], %[[TG12]], %[[TG20]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// Reducing output vector [2, 0]

// CHECK-NEXT: %[[TG21:.*]] = vector.tuple_get %[[ES1]], 3 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG22:.*]] = vector.tuple_get %[[ES3]], 2 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG23:.*]] = vector.tuple_get %[[ES4]], 3 : tuple<vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>>
// CHECK-NEXT: %[[R1S20:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[TG21]], %[[TG2]], %[[TG22]], %[[TG23]], %[[TG5]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// CHECK-NEXT: %[[TG24:.*]] = vector.tuple_get %[[ES1]], 4 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG25:.*]] = vector.tuple_get %[[ES4]], 4 : tuple<vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>>
// CHECK-NEXT:  %[[R2S20:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[TG24]], %[[TG7]], %[[R1S20]], %[[TG25]], %[[TG9]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// CHECK-NEXT: %[[TG26:.*]] = vector.tuple_get %[[ES1]], 5 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG27:.*]] = vector.tuple_get %[[ES4]], 5 : tuple<vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>>
// CHECK-NEXT:  %[[R3S20:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[TG26]], %[[TG11]], %[[R2S20]], %[[TG27]], %[[TG13]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// Reducing output vector [2, 2]

// CHECK-NEXT: %[[TG28:.*]] = vector.tuple_get %[[ES3]], 3 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[R1S22:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[TG21]], %[[TG14]], %[[TG28]], %[[TG23]], %[[TG16]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
// CHECK-NEXT: %[[R2S22:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[TG24]], %[[TG17]], %[[R1S22]], %[[TG25]], %[[TG18]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
// CHECK-NEXT: %[[R3S22:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[TG26]], %[[TG19]], %[[R2S22]], %[[TG27]], %[[TG20]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// CHECK-NEXT: %[[RES0:.*]] = vector.tuple %[[R3S00]], %[[R3S02]], %[[R3S20]], %[[R3S22]] : vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>
// CHECK-NEXT: %[[RES1:.*]] = vector.insert_slices %[[RES0]], [2, 2], [1, 1] : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>> into vector<4x4xf32>
// CHECK-NEXT:  return %[[RES1]] : vector<4x4xf32>

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


//      CHECK: %[[LMASK:.*]] = vector.constant_mask [4, 2] : vector<4x2xi1>
// CHECK-NEXT: %[[RMASK:.*]] = vector.constant_mask [2, 4] : vector<2x4xi1>

// Reducing output vector [0, 0]

// CHECK-NEXT: %[[ES1:.*]] = vector.extract_slices %{{.*}}, [2, 2], [1, 1] : vector<4x2xf32> into tuple<vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[ES2:.*]] = vector.extract_slices %{{.*}}, [2, 2], [1, 1] : vector<2x4xf32> into tuple<vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[ES3:.*]] = vector.extract_slices %{{.*}}, [2, 2], [1, 1] : vector<4x4xf32> into tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[ES4:.*]] = vector.extract_slices %{{.*}}, [2, 2], [1, 1] : vector<4x2xi1> into tuple<vector<2x2xi1>, vector<2x2xi1>>
// CHECK-NEXT: %[[ES5:.*]] = vector.extract_slices %{{.*}}, [2, 2], [1, 1] : vector<2x4xi1> into tuple<vector<2x2xi1>, vector<2x2xi1>>

// CHECK-NEXT: %[[TG1:.*]] = vector.tuple_get %[[ES1]], 0 : tuple<vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG2:.*]] = vector.tuple_get %[[ES2]], 0 : tuple<vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG3:.*]] = vector.tuple_get %[[ES3]], 0 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG4:.*]] = vector.tuple_get %[[ES4]], 0 : tuple<vector<2x2xi1>, vector<2x2xi1>>
// CHECK-NEXT: %[[TG5:.*]] = vector.tuple_get %[[ES5]], 0 : tuple<vector<2x2xi1>, vector<2x2xi1>>
// CHECK-NEXT:  %[[R1S00:.*]] = vector.contract {indexing_maps = [#map2, #map3, #map0], iterator_types = ["parallel", "reduction", "parallel"], kind = #vector.kind<add>} %[[TG1]], %[[TG2]], %[[TG3]], %[[TG4]], %[[TG5]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// Reducing output vector [0, 2]

// CHECK-NEXT: %[[TG6:.*]] = vector.tuple_get %[[ES2]], 1 : tuple<vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG7:.*]] = vector.tuple_get %[[ES3]], 1 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG8:.*]] = vector.tuple_get %[[ES5]], 1 : tuple<vector<2x2xi1>, vector<2x2xi1>>
// CHECK-NEXT:  %[[R1S02:.*]] = vector.contract {indexing_maps = [#map2, #map3, #map0], iterator_types = ["parallel", "reduction", "parallel"], kind = #vector.kind<add>} %[[TG1]], %[[TG6]], %[[TG7]], %[[TG4]], %[[TG8]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// Reducing output vector [2, 0]

// CHECK-NEXT: %[[TG9:.*]] = vector.tuple_get %[[ES1]], 1 : tuple<vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG10:.*]] = vector.tuple_get %[[ES3]], 2 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG11:.*]] = vector.tuple_get %[[ES4]], 1 : tuple<vector<2x2xi1>, vector<2x2xi1>>
// CHECK-NEXT:  %[[R1S20:.*]] = vector.contract {indexing_maps = [#map2, #map3, #map0], iterator_types = ["parallel", "reduction", "parallel"], kind = #vector.kind<add>} %[[TG9]], %[[TG2]], %[[TG10]], %[[TG11]], %[[TG5]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// Reducing output vector [2, 2]

// CHECK-NEXT: %[[TG12:.*]] = vector.tuple_get %[[ES3]], 3 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT:  %[[R1S22:.*]] = vector.contract {indexing_maps = [#map2, #map3, #map0], iterator_types = ["parallel", "reduction", "parallel"], kind = #vector.kind<add>} %[[TG9]], %[[TG6]], %[[TG12]], %[[TG11]], %[[TG8]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// CHECK-NEXT: %[[RES0:.*]] = vector.tuple %[[R1S00]], %[[R1S02]], %[[R1S20]], %[[R1S22]] : vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>
// CHECK-NEXT: %[[RES1:.*]] = vector.insert_slices %[[RES0]], [2, 2], [1, 1] : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>> into vector<4x4xf32>
// CHECK-NEXT:  return %[[RES1]] : vector<4x4xf32>

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

// CHECK:      %[[C0:.*]] = constant 0 : index
// CHECK:      %[[C2:.*]] = constant 2 : index

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

// CHECK-NEXT: vector.transfer_write %[[R0]], %{{.*}}[%[[C0]], %[[C0]]] {masked = [false, false]} : vector<2x2xf32>, memref<4x4xf32>
// CHECK-NEXT: vector.transfer_write %[[R1]], %{{.*}}[%[[C0]], %[[C2]]] {masked = [false, false]} : vector<2x2xf32>, memref<4x4xf32>
// CHECK-NEXT: vector.transfer_write %[[R2]], %{{.*}}[%[[C2]], %[[C0]]] {masked = [false, false]} : vector<2x2xf32>, memref<4x4xf32>
// CHECK-NEXT: vector.transfer_write %[[R3]], %{{.*}}[%[[C2]], %[[C2]]] {masked = [false, false]} : vector<2x2xf32>, memref<4x4xf32>
// CHECK-NEXT: return

func @contraction4x4_ikj_xfer_read(%arg0 : memref<4x2xf32>,
                                   %arg1 : memref<2x4xf32>,
                                   %arg2 : memref<4x4xf32>) {
  %c0 = constant 0 : index
  %cf0 = constant 0.0 : f32

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
// CHECK-COUNT-4: addf
// CHECK-COUNT-4: vector.transfer_write

func @vector_transfers(%arg0: index, %arg1: index) {
  %cst = constant 0.000000e+00 : f32
  %0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
  %1 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
  %2 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
  %cst_0 = constant 1.000000e+00 : f32
  %cst_1 = constant 2.000000e+00 : f32
  affine.for %arg2 = 0 to %arg0 step 4 {
    affine.for %arg3 = 0 to %arg1 step 4 {
      %4 = vector.transfer_read %0[%arg2, %arg3], %cst {permutation_map = affine_map<(d0, d1) -> (d0, d1)>} : memref<?x?xf32>, vector<4x4xf32>
      %5 = vector.transfer_read %1[%arg2, %arg3], %cst {permutation_map = affine_map<(d0, d1) -> (d0, d1)>} : memref<?x?xf32>, vector<4x4xf32>
      %6 = addf %4, %5 : vector<4x4xf32>
      vector.transfer_write %6, %2[%arg2, %arg3] {permutation_map = affine_map<(d0, d1) -> (d0, d1)>} : vector<4x4xf32>, memref<?x?xf32>
    }
  }
  return
}

// CHECK-LABEL: func @tuple_get(%arg0: vector<4xf32>, %arg1: vector<8xf32>)
//       CHECK: return %arg1

func @tuple_get(%arg0: vector<4xf32>, %arg1: vector<8xf32>) -> vector<8xf32> {
  %0 = vector.tuple %arg0, %arg1 : vector<4xf32>, vector<8xf32>
  %1 = vector.tuple_get %0, 1 : tuple<vector<4xf32>, vector<8xf32>>
  return %1 : vector<8xf32>
}

// CHECK-LABEL: func @tuple_get_producer_consumer
// CHECK-SAME: %[[A0:.*0]]: vector<2x4xf32>,
// CHECK-SAME: %[[A1:.*1]]: vector<2x4xf32>,
// CHECK-SAME: %[[A2:.*2]]: vector<2x4xf32>,
// CHECK-SAME: %[[A3:.*3]]: vector<2x4xf32>,
// CHECK-SAME: %[[A4:.*4]]: vector<2x4xf32>,
// CHECK-SAME: %[[A5:.*5]]: vector<2x4xf32>,
// CHECK-SAME: %[[A6:.*6]]: vector<2x4xf32>,
// CHECK-SAME: %[[A7:.*7]]: vector<2x4xf32>
//      CHECK: return %[[A7]] : vector<2x4xf32>

func @tuple_get_producer_consumer(
  %arg0 : vector<2x4xf32>, %arg1 : vector<2x4xf32>,
  %arg2 : vector<2x4xf32>, %arg3 : vector<2x4xf32>,
  %arg4 : vector<2x4xf32>, %arg5 : vector<2x4xf32>,
  %arg6 : vector<2x4xf32>, %arg7 : vector<2x4xf32>) -> vector<2x4xf32> {
  %0 = vector.tuple %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7
    : vector<2x4xf32>, vector<2x4xf32>, vector<2x4xf32>, vector<2x4xf32>,
      vector<2x4xf32>, vector<2x4xf32>, vector<2x4xf32>, vector<2x4xf32>
  // %arg7 == %0 at tupleIndex = 7, offsets = [0, 0]
  %1 = vector.insert_slices %0, [2, 4], [1, 1]
    : tuple<vector<2x4xf32>, vector<2x4xf32>, vector<2x4xf32>, vector<2x4xf32>,
            vector<2x4xf32>, vector<2x4xf32>, vector<2x4xf32>, vector<2x4xf32>>
      into vector<4x16xf32>
  // %arg7 == %1 at tupleIndex = -1, offsets = [2, 12]
  %2 = vector.extract_slices %1, [4, 8], [1, 1]
    : vector<4x16xf32> into tuple<vector<4x8xf32>, vector<4x8xf32>>
  // %arg7 == %2 at tupleIndex = 1, offsets = [2, 4]
  %3 = vector.shape_cast %2 : tuple<vector<4x8xf32>, vector<4x8xf32>> to
                              tuple<vector<1x1x4x8xf32>, vector<1x1x4x8xf32>>
  // %arg7 = %3 at tupleIndex = 1, offsets = [0, 0, 2, 4]
  %4 = vector.tuple_get %3, 1 : tuple<vector<1x1x4x8xf32>, vector<1x1x4x8xf32>>
  // %arg7 == %4 at tupleIndex = -1, offsets = [0, 0, 2, 4]
  %5 = vector.shape_cast %4 : vector<1x1x4x8xf32> to vector<4x8xf32>
  // %arg7 == %5 at tupleIndex = -1, offsets = [2, 4]
  %6 = vector.extract_slices %5, [2, 4], [1, 1]
    : vector<4x8xf32> into
      tuple<vector<2x4xf32>, vector<2x4xf32>, vector<2x4xf32>, vector<2x4xf32>>
  // %arg7 == %6 at tupleIndex = 3, offsets = [0, 0]
  %7 = vector.tuple_get %6, 3
    : tuple<vector<2x4xf32>, vector<2x4xf32>, vector<2x4xf32>, vector<2x4xf32>>
  // %arg7 == %7
  return %7 : vector<2x4xf32>
}

// CHECK-LABEL: func @tuple_get_producer_consumer_swizzle
// CHECK-SAME: %[[A0:.*0]]: vector<2x4xf32>,
// CHECK-SAME: %[[A1:.*1]]: vector<2x4xf32>,
// CHECK-SAME: %[[A2:.*2]]: vector<2x4xf32>,
// CHECK-SAME: %[[A3:.*3]]: vector<2x4xf32>,
// CHECK-SAME: %[[A4:.*4]]: vector<2x4xf32>,
// CHECK-SAME: %[[A5:.*5]]: vector<2x4xf32>,
// CHECK-SAME: %[[A6:.*6]]: vector<2x4xf32>,
// CHECK-SAME: %[[A7:.*7]]: vector<2x4xf32>
//      CHECK: return %[[A7]] : vector<2x4xf32>

func @tuple_get_producer_consumer_swizzle(
  %arg0 : vector<2x4xf32>, %arg1 : vector<2x4xf32>,
  %arg2 : vector<2x4xf32>, %arg3 : vector<2x4xf32>,
  %arg4 : vector<2x4xf32>, %arg5 : vector<2x4xf32>,
  %arg6 : vector<2x4xf32>, %arg7 : vector<2x4xf32>) -> vector<2x4xf32> {
  %0 = vector.tuple %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7
    : vector<2x4xf32>, vector<2x4xf32>, vector<2x4xf32>, vector<2x4xf32>,
      vector<2x4xf32>, vector<2x4xf32>, vector<2x4xf32>, vector<2x4xf32>
  // %arg7 == %0 at tupleIndex = 7, offsets = [0, 0]
  %1 = vector.insert_slices %0, [2, 4], [1, 1]
    : tuple<vector<2x4xf32>, vector<2x4xf32>, vector<2x4xf32>, vector<2x4xf32>,
            vector<2x4xf32>, vector<2x4xf32>, vector<2x4xf32>, vector<2x4xf32>>
      into vector<4x16xf32>
  // %arg7 == %1 at tupleIndex = -1, offsets = [2, 12]
  %2 = vector.extract_slices %1, [4, 8], [1, 1]
    : vector<4x16xf32> into tuple<vector<4x8xf32>, vector<4x8xf32>>
  // %arg7 == %2 at tupleIndex = 1, offsets = [2, 4]
  %3= vector.shape_cast %2 : tuple<vector<4x8xf32>, vector<4x8xf32>> to
                             tuple<vector<1x1x4x8xf32>, vector<1x1x4x8xf32>>
  // %arg7 = %3 at tupleIndex = 1, offsets = [0, 0, 2, 4]

  // Extract tuple elements.
  %4 = vector.tuple_get %3, 0 : tuple<vector<1x1x4x8xf32>, vector<1x1x4x8xf32>>
  %5 = vector.tuple_get %3, 1 : tuple<vector<1x1x4x8xf32>, vector<1x1x4x8xf32>>
  // %arg7 == %5 at tupleIndex = -1, offsets = [0, 0, 2, 4]

  // Swizzle tuple elements.
  %6 = vector.tuple %5, %4 : vector<1x1x4x8xf32>, vector<1x1x4x8xf32>
  // %arg7 == %6 at tupleIndex = 0, offsets = [0, 0, 2, 4]
  %7 = vector.shape_cast %6 : tuple<vector<1x1x4x8xf32>, vector<1x1x4x8xf32>> to
                              tuple<vector<4x8xf32>, vector<4x8xf32>>
  // %arg7 = %7 at tupleIndex = 0, offsets = [2, 4]
  %8 = vector.tuple_get %7, 0 : tuple<vector<4x8xf32>, vector<4x8xf32>>
  // %arg7 == %8 at tupleIndex = -1, offsets = [2, 4]
  %9 = vector.extract_slices %8, [2, 4], [1, 1]
    : vector<4x8xf32> into
      tuple<vector<2x4xf32>, vector<2x4xf32>, vector<2x4xf32>, vector<2x4xf32>>
  // %arg7 == %9 at tupleIndex = 3, offsets = [0, 0]
  %10 = vector.tuple_get %9, 3
    : tuple<vector<2x4xf32>, vector<2x4xf32>, vector<2x4xf32>, vector<2x4xf32>>
  // %arg7 == %10
  return %10 : vector<2x4xf32>
}

// CHECK-LABEL: func @cancelling_shape_cast_ops
//  CHECK-SAME: %[[A0:.*0]]: vector<2x4xf32>
//       CHECK: return %[[A0]] : vector<2x4xf32>
func @cancelling_shape_cast_ops(%arg0 : vector<2x4xf32>) -> vector<2x4xf32> {
  %0 = vector.shape_cast %arg0 : vector<2x4xf32> to vector<8xf32>
  %1 = vector.shape_cast %0 : vector<8xf32> to vector<2x4xf32>
  return %1 : vector<2x4xf32>
}

// CHECK-LABEL: func @vector_transfers_vector_element_type
//      CHECK: %[[C0:.*]] = constant 0 : index
//      CHECK: %[[C1:.*]] = constant 1 : index
//      CHECK: %[[VTR0:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[C0]], %[[C0]]], %{{.*}} {masked = [false, false]} : memref<6x2x1xvector<2x4xf32>>, vector<1x1x2x4xf32>
// CHECK-NEXT: %[[VTR1:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[C1]], %[[C0]]], %{{.*}} {masked = [false, false]} : memref<6x2x1xvector<2x4xf32>>, vector<1x1x2x4xf32>
// CHECK-NEXT: vector.transfer_write %[[VTR0]], %{{.*}}[%[[C0]], %[[C0]], %[[C0]]] {masked = [false, false]} : vector<1x1x2x4xf32>, memref<6x2x1xvector<2x4xf32>>
// CHECK-NEXT: vector.transfer_write %[[VTR1]], %{{.*}}[%[[C0]], %[[C1]], %[[C0]]] {masked = [false, false]} : vector<1x1x2x4xf32>, memref<6x2x1xvector<2x4xf32>>

func @vector_transfers_vector_element_type() {
  %c0 = constant 0 : index
  %cf0 = constant 0.000000e+00 : f32
  %vf0 = splat %cf0 : vector<2x4xf32>

  %0 = memref.alloc() : memref<6x2x1xvector<2x4xf32>>

  %1 = vector.transfer_read %0[%c0, %c0, %c0], %vf0
      {permutation_map = affine_map<(d0, d1, d2) -> (d1, d2)>}
        : memref<6x2x1xvector<2x4xf32>>, vector<2x1x2x4xf32>

  %2 = vector.extract_slices %1, [1, 1, 2, 4], [1, 1, 1, 1]
    : vector<2x1x2x4xf32> into tuple<vector<1x1x2x4xf32>, vector<1x1x2x4xf32>>
  %3 = vector.tuple_get %2, 0 : tuple<vector<1x1x2x4xf32>, vector<1x1x2x4xf32>>
  %4 = vector.tuple_get %2, 1 : tuple<vector<1x1x2x4xf32>, vector<1x1x2x4xf32>>
  %5 = vector.tuple %3, %4 : vector<1x1x2x4xf32>, vector<1x1x2x4xf32>
  %6 = vector.insert_slices %5, [1, 1, 2, 4], [1, 1, 1, 1]
    : tuple<vector<1x1x2x4xf32>, vector<1x1x2x4xf32>> into vector<2x1x2x4xf32>

  vector.transfer_write %6, %0[%c0, %c0, %c0]
    {permutation_map = affine_map<(d0, d1, d2) -> (d1, d2)>}
      : vector<2x1x2x4xf32>, memref<6x2x1xvector<2x4xf32>>

  return
}

// Test that ShapeCastOp on tuple of vectors, decomposes to multiple
// ShapeCastOps on vectors.
// CHECK-LABEL: func @shape_cast_decomposition
//       CHECK: %[[V0:.*]] = vector.shape_cast %{{.*}} : vector<5x4x2xf32> to vector<20x2xf32>
//  CHECK-NEXT: %[[V1:.*]] = vector.shape_cast %{{.*}} : vector<3x4x2xf32> to vector<12x2xf32>
//  CHECK-NEXT: return %[[V0]], %[[V1]] : vector<20x2xf32>, vector<12x2xf32>

func @shape_cast_decomposition(%arg0 : vector<5x4x2xf32>,
                               %arg1 : vector<3x4x2xf32>)
  -> (vector<20x2xf32>, vector<12x2xf32>) {
  %0 = vector.tuple %arg0, %arg1 : vector<5x4x2xf32>, vector<3x4x2xf32>
  %1 = vector.shape_cast %0 : tuple<vector<5x4x2xf32>, vector<3x4x2xf32>> to
                              tuple<vector<20x2xf32>, vector<12x2xf32>>
  %2 = vector.tuple_get %1, 0 : tuple<vector<20x2xf32>, vector<12x2xf32>>
  %3 = vector.tuple_get %1, 1 : tuple<vector<20x2xf32>, vector<12x2xf32>>
  return %2, %3 : vector<20x2xf32>, vector<12x2xf32>
}

// Test that cancelling ShapeCastOps are canonicalized away.
// EX:
//
//  The following MLIR with cancelling ShapeCastOps:
//
//   %0 = source : vector<5x4x2xf32>
//   %1 = shape_cast %0 : vector<5x4x2xf32> to vector<20x2xf32>
//   %2 = shape_cast %1 : vector<20x2xf32> to vector<5x4x2xf32>
//   %3 = user %2 : vector<5x4x2xf32>
//
//  Should canonicalize to the following:
//
//
//   %0 = source : vector<5x4x2xf32>
//   %1 = user %0 : vector<5x4x2xf32>
//

// ShapeCastOps on vectors.
// CHECK-LABEL: func @shape_cast_fold
//       CHECK: return %{{.*}},  %{{.*}} : vector<5x4x2xf32>, vector<3x4x2xf32>

func @shape_cast_fold(%arg0 : vector<5x4x2xf32>, %arg1 : vector<3x4x2xf32>)
  -> (vector<5x4x2xf32>, vector<3x4x2xf32>) {
  %0 = vector.tuple %arg0, %arg1 : vector<5x4x2xf32>, vector<3x4x2xf32>

  %1 = vector.shape_cast %0 : tuple<vector<5x4x2xf32>, vector<3x4x2xf32>> to
                              tuple<vector<20x2xf32>, vector<12x2xf32>>

  %2 = vector.tuple_get %1, 0 : tuple<vector<20x2xf32>, vector<12x2xf32>>
  %3 = vector.tuple_get %1, 1 : tuple<vector<20x2xf32>, vector<12x2xf32>>

  %4 = vector.tuple %2, %3 : vector<20x2xf32>, vector<12x2xf32>
  %5 = vector.shape_cast %4 : tuple<vector<20x2xf32>, vector<12x2xf32>> to
                              tuple<vector<5x4x2xf32>, vector<3x4x2xf32>>

  %6 = vector.tuple_get %5, 0 : tuple<vector<5x4x2xf32>, vector<3x4x2xf32>>
  %7 = vector.tuple_get %5, 1 : tuple<vector<5x4x2xf32>, vector<3x4x2xf32>>

  return %6, %7 : vector<5x4x2xf32>, vector<3x4x2xf32>
}

// CHECK-LABEL: func @elementwise_unroll
//  CHECK-SAME: (%[[ARG0:.*]]: memref<4x4xf32>, %[[ARG1:.*]]: memref<4x4xf32>)
//       CHECK:   %[[C0:.*]] = constant 0 : index
//       CHECK:   %[[C2:.*]] = constant 2 : index
//       CHECK:   %[[VT0:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[VT1:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C2]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[VT2:.*]] = vector.transfer_read %[[ARG0]][%[[C2]], %[[C0]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[VT3:.*]] = vector.transfer_read %[[ARG0]][%[[C2]], %[[C2]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[VT4:.*]] = vector.transfer_read %[[ARG1]][%[[C0]], %[[C0]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[VT5:.*]] = vector.transfer_read %[[ARG1]][%[[C0]], %[[C2]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[VT6:.*]] = vector.transfer_read %[[ARG1]][%[[C2]], %[[C0]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[VT7:.*]] = vector.transfer_read %[[ARG1]][%[[C2]], %[[C2]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[CMP0:.*]] = cmpf ult, %[[VT0]], %[[VT4]] : vector<2x2xf32>
//       CHECK:   %[[CMP1:.*]] = cmpf ult, %[[VT1]], %[[VT5]] : vector<2x2xf32>
//       CHECK:   %[[CMP2:.*]] = cmpf ult, %[[VT2]], %[[VT6]] : vector<2x2xf32>
//       CHECK:   %[[CMP3:.*]] = cmpf ult, %[[VT3]], %[[VT7]] : vector<2x2xf32>
//       CHECK:   %[[VT0:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[VT1:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C2]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[VT2:.*]] = vector.transfer_read %[[ARG0]][%[[C2]], %[[C0]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[VT3:.*]] = vector.transfer_read %[[ARG0]][%[[C2]], %[[C2]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[VT4:.*]] = vector.transfer_read %[[ARG1]][%[[C0]], %[[C0]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[VT5:.*]] = vector.transfer_read %[[ARG1]][%[[C0]], %[[C2]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[VT6:.*]] = vector.transfer_read %[[ARG1]][%[[C2]], %[[C0]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[VT7:.*]] = vector.transfer_read %[[ARG1]][%[[C2]], %[[C2]]], {{.*}} : memref<4x4xf32>, vector<2x2xf32>
//       CHECK:   %[[SEL0:.*]] = select %[[CMP0]], %[[VT0]], %[[VT4]] : vector<2x2xi1>, vector<2x2xf32>
//       CHECK:   %[[SEL1:.*]] = select %[[CMP1]], %[[VT1]], %[[VT5]] : vector<2x2xi1>, vector<2x2xf32>
//       CHECK:   %[[SEL2:.*]] = select %[[CMP2]], %[[VT2]], %[[VT6]] : vector<2x2xi1>, vector<2x2xf32>
//       CHECK:   %[[SEL3:.*]] = select %[[CMP3]], %[[VT3]], %[[VT7]] : vector<2x2xi1>, vector<2x2xf32>
//       CHECK:   vector.transfer_write %[[SEL0]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
//       CHECK:   vector.transfer_write %[[SEL1]], %[[ARG0]][%[[C0]], %[[C2]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
//       CHECK:   vector.transfer_write %[[SEL2]], %[[ARG0]][%[[C2]], %[[C0]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
//       CHECK:   vector.transfer_write %[[SEL3]], %[[ARG0]][%[[C2]], %[[C2]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
func @elementwise_unroll(%arg0 : memref<4x4xf32>, %arg1 : memref<4x4xf32>) {
  %c0 = constant 0 : index
  %cf0 = constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0], %cf0 : memref<4x4xf32>, vector<4x4xf32>
  %1 = vector.transfer_read %arg1[%c0, %c0], %cf0 : memref<4x4xf32>, vector<4x4xf32>
  %cond = cmpf ult, %0, %1 : vector<4x4xf32>
  // Vector transfer split pattern only support single user right now.
  %2 = vector.transfer_read %arg0[%c0, %c0], %cf0 : memref<4x4xf32>, vector<4x4xf32>
  %3 = vector.transfer_read %arg1[%c0, %c0], %cf0 : memref<4x4xf32>, vector<4x4xf32>
  %4 = select %cond, %2, %3 : vector<4x4xi1>, vector<4x4xf32>
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

// CHECK-NEXT: %[[VTW0:.*]] = vector.transfer_write %[[R0]], %{{.*}}[%[[C0]], %[[C0]]] {masked = [false, false]} : vector<2x2xf32>, tensor<4x4xf32>
// CHECK-NEXT: %[[VTW1:.*]] = vector.transfer_write %[[R1]], %[[VTW0]][%[[C0]], %[[C2]]] {masked = [false, false]} : vector<2x2xf32>, tensor<4x4xf32>
// CHECK-NEXT: %[[VTW2:.*]] = vector.transfer_write %[[R2]], %[[VTW1]][%[[C2]], %[[C0]]] {masked = [false, false]} : vector<2x2xf32>, tensor<4x4xf32>
// CHECK-NEXT: %[[VTW3:.*]] = vector.transfer_write %[[R3]], %[[VTW2]][%[[C2]], %[[C2]]] {masked = [false, false]} : vector<2x2xf32>, tensor<4x4xf32>
// CHECK-NEXT: return %[[VTW3]] : tensor<4x4xf32>

func @contraction4x4_ikj_xfer_read_tensor(%arg0 : tensor<4x2xf32>,
                                          %arg1 : tensor<2x4xf32>,
                                          %arg2 : tensor<4x4xf32>) ->
  tensor<4x4xf32> {
  %c0 = constant 0 : index
  %cf0 = constant 0.0 : f32
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

// CHECK-LABEL: func @cast_away_extract_strided_slice_leading_one_dims
func @cast_away_extract_strided_slice_leading_one_dims(%arg0: vector<1x8x8xf16>) -> vector<1x1x8xf16> {
  // CHECK:     %[[SRC:.+]] = vector.shape_cast %{{.*}} : vector<1x8x8xf16> to vector<8x8xf16>
  // CHECK: %[[EXTRACT:.+]] = vector.extract_strided_slice %[[SRC]] {offsets = [4], sizes = [1], strides = [1]} : vector<8x8xf16> to vector<1x8xf16>
  %0 = vector.extract_strided_slice %arg0 {offsets = [0, 4], sizes = [1, 1], strides = [1, 1]} : vector<1x8x8xf16> to vector<1x1x8xf16>
  // CHECK:     %[[RET:.+]] = vector.shape_cast %[[EXTRACT]] : vector<1x8xf16> to vector<1x1x8xf16>
  // CHECK: return %[[RET]]
  return %0: vector<1x1x8xf16>
}

// CHECK-LABEL: func @cast_away_insert_strided_slice_leading_one_dims
func @cast_away_insert_strided_slice_leading_one_dims(%arg0: vector<1x8xf16>, %arg1: vector<1x8x8xf16>) -> vector<1x8x8xf16> {
  // CHECK:    %[[SRC:.+]] = vector.shape_cast %{{.*}} : vector<1x8xf16> to vector<8xf16>
  // CHECK:    %[[DST:.+]] = vector.shape_cast %{{.*}} : vector<1x8x8xf16> to vector<8x8xf16>
  // CHECK: %[[INSERT:.+]] = vector.insert_strided_slice %[[SRC]], %[[DST]] {offsets = [0, 0], strides = [1]} : vector<8xf16> into vector<8x8xf16>
  %0 = vector.insert_strided_slice %arg0, %arg1 {offsets = [0, 0, 0], strides = [1, 1]} : vector<1x8xf16> into vector<1x8x8xf16>
  // CHECK:    %[[RET:.+]] = vector.shape_cast %[[INSERT]] : vector<8x8xf16> to vector<1x8x8xf16>
  // CHECK: return %[[RET]]
  return %0: vector<1x8x8xf16>
}

// CHECK-LABEL: func @cast_away_insert_strided_slice_leading_one_dims_one_element
func @cast_away_insert_strided_slice_leading_one_dims_one_element(%arg0: vector<1x1xf16>, %arg1: vector<1x1x1xf16>) -> vector<1x1x1xf16> {
  // CHECK: vector.shape_cast %{{.+}} : vector<1x1xf16> to vector<1xf16>
  // CHECK: vector.shape_cast %{{.+}} : vector<1x1x1xf16> to vector<1xf16>
  %0 = vector.insert_strided_slice %arg0, %arg1 {offsets = [0, 0, 0], strides = [1, 1]} : vector<1x1xf16> into vector<1x1x1xf16>
  return %0: vector<1x1x1xf16>
}

// CHECK-LABEL: func @cast_away_transfer_read_leading_one_dims
func @cast_away_transfer_read_leading_one_dims(%arg0: memref<1x4x8x16xf16>) -> vector<1x4xf16> {
  // CHECK: %[[C0:.+]] = constant 0 : index
  %c0 = constant 0 : index
  // CHECK: %[[F0:.+]] = constant 0.000000e+00 : f16
  %f0 = constant 0. : f16
  // CHECK: %[[READ:.+]] = vector.transfer_read %{{.*}}[%[[C0]], %[[C0]], %[[C0]], %[[C0]]], %[[F0]] {masked = [false]} : memref<1x4x8x16xf16>, vector<4xf16>
  // CHECK: %[[CAST:.+]] = vector.shape_cast %[[READ]] : vector<4xf16> to vector<1x4xf16>
  %0 = vector.transfer_read %arg0[%c0, %c0, %c0, %c0], %f0 {masked = [false, false]} : memref<1x4x8x16xf16>, vector<1x4xf16>
  // CHECK: return %[[CAST]]
  return %0: vector<1x4xf16>
}

// CHECK-LABEL: func @cast_away_transfer_read_leading_one_dims_one_element
func @cast_away_transfer_read_leading_one_dims_one_element(%arg0: memref<1x1x1x1xf16>) -> vector<1x1xf16> {
  %c0 = constant 0 : index
  %f0 = constant 0. : f16
  // CHECK: vector.shape_cast %{{.+}} : vector<1xf16> to vector<1x1xf16>
  %0 = vector.transfer_read %arg0[%c0, %c0, %c0, %c0], %f0 {masked = [false, false]} : memref<1x1x1x1xf16>, vector<1x1xf16>
  return %0: vector<1x1xf16>
}

// CHECK-LABEL: func @cast_away_transfer_write_leading_one_dims
func @cast_away_transfer_write_leading_one_dims(%arg0: memref<1x4x8x16xf16>, %arg1: vector<1x4xf16>) {
  // CHECK: %[[C0:.+]] = constant 0 : index
  %c0 = constant 0 : index
  // CHECK: %[[CAST:.+]] = vector.shape_cast %{{.*}} : vector<1x4xf16> to vector<4xf16>
  // CHECK: vector.transfer_write %[[CAST]], %{{.*}}[%[[C0]], %[[C0]], %[[C0]], %[[C0]]] {masked = [false]} : vector<4xf16>, memref<1x4x8x16xf16>

  vector.transfer_write %arg1, %arg0[%c0, %c0, %c0, %c0] {masked = [false, false]} : vector<1x4xf16>, memref<1x4x8x16xf16>
  return
}

// CHECK-LABEL: func @cast_away_transfer_write_leading_one_dims_one_element
func @cast_away_transfer_write_leading_one_dims_one_element(%arg0: memref<1x1x1x1xf16>, %arg1: vector<1x1xf16>) {
  %c0 = constant 0 : index
  // CHECK: vector.shape_cast %{{.+}} : vector<1x1xf16> to vector<1xf16>
  vector.transfer_write %arg1, %arg0[%c0, %c0, %c0, %c0] {masked = [false, false]} : vector<1x1xf16>, memref<1x1x1x1xf16>
  return
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
