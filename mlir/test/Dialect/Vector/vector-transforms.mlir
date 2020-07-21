// RUN: mlir-opt %s -test-vector-to-vector-conversion | FileCheck %s
// RUN: mlir-opt %s -test-vector-unrolling-patterns | FileCheck %s

// CHECK-DAG: #[[MAP0:map[0-9]+]] = affine_map<(d0, d1) -> (d0, d1)>
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
// CHECK-NEXT: %[[R1S00:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %[[TG1]], %[[TG2]], %[[TG3]], %[[TG4]], %[[TG5]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// CHECK-NEXT: %[[TG6:.*]] = vector.tuple_get %[[ES1]], 1 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG7:.*]] = vector.tuple_get %[[ES2]], 2 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG8:.*]] = vector.tuple_get %[[ES4]], 1 : tuple<vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>>
// CHECK-NEXT: %[[TG9:.*]] = vector.tuple_get %[[ES5]], 2 : tuple<vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>>
// CHECK-NEXT: %[[R2S00:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %[[TG6]], %[[TG7]], %[[R1S00]], %[[TG8]], %[[TG9]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// CHECK-NEXT: %[[TG10:.*]] = vector.tuple_get %[[ES1]], 2 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG11:.*]] = vector.tuple_get %[[ES2]], 4 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG12:.*]] = vector.tuple_get %[[ES4]], 2 : tuple<vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>>
// CHECK-NEXT: %[[TG13:.*]] = vector.tuple_get %[[ES5]], 4 : tuple<vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>>
// CHECK-NEXT: %[[R3S00:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %[[TG10]], %[[TG11]], %[[R2S00]], %[[TG12]], %[[TG13]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// Reducing output vector [0, 2]

// CHECK-NEXT: %[[TG14:.*]] = vector.tuple_get %[[ES2]], 1 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG15:.*]] = vector.tuple_get %[[ES3]], 1 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG16:.*]] = vector.tuple_get %[[ES5]], 1 : tuple<vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>>
// CHECK-NEXT: %[[R1S02:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %[[TG1]], %[[TG14]], %[[TG15]], %[[TG4]], %[[TG16]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// CHECK-NEXT: %[[TG17:.*]] = vector.tuple_get %[[ES2]], 3 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG18:.*]] = vector.tuple_get %[[ES5]], 3 : tuple<vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>>
// CHECK-NEXT: %[[R2S02:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %[[TG6]], %[[TG17]], %[[R1S02]], %[[TG8]], %[[TG18]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// CHECK-NEXT: %[[TG19:.*]] = vector.tuple_get %[[ES2]], 5 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG20:.*]] = vector.tuple_get %[[ES5]], 5 : tuple<vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>>
// CHECK-NEXT: %[[R3S02:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %[[TG10]], %[[TG19]], %[[R2S02]], %[[TG12]], %[[TG20]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// Reducing output vector [2, 0]

// CHECK-NEXT: %[[TG21:.*]] = vector.tuple_get %[[ES1]], 3 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG22:.*]] = vector.tuple_get %[[ES3]], 2 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG23:.*]] = vector.tuple_get %[[ES4]], 3 : tuple<vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>>
// CHECK-NEXT: %[[R1S20:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %[[TG21]], %[[TG2]], %[[TG22]], %[[TG23]], %[[TG5]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// CHECK-NEXT: %[[TG24:.*]] = vector.tuple_get %[[ES1]], 4 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG25:.*]] = vector.tuple_get %[[ES4]], 4 : tuple<vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>>
// CHECK-NEXT:  %[[R2S20:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %[[TG24]], %[[TG7]], %[[R1S20]], %[[TG25]], %[[TG9]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// CHECK-NEXT: %[[TG26:.*]] = vector.tuple_get %[[ES1]], 5 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG27:.*]] = vector.tuple_get %[[ES4]], 5 : tuple<vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>, vector<2x2xi1>>
// CHECK-NEXT:  %[[R3S20:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %[[TG26]], %[[TG11]], %[[R2S20]], %[[TG27]], %[[TG13]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// Reducing output vector [2, 2]

// CHECK-NEXT: %[[TG28:.*]] = vector.tuple_get %[[ES3]], 3 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[R1S22:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %[[TG21]], %[[TG14]], %[[TG28]], %[[TG23]], %[[TG16]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
// CHECK-NEXT: %[[R2S22:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %[[TG24]], %[[TG17]], %[[R1S22]], %[[TG25]], %[[TG18]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
// CHECK-NEXT: %[[R3S22:.*]] = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %[[TG26]], %[[TG19]], %[[R2S22]], %[[TG27]], %[[TG20]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

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
// CHECK-NEXT:  %[[R1S00:.*]] = vector.contract {indexing_maps = [#map2, #map3, #map0], iterator_types = ["parallel", "reduction", "parallel"]} %[[TG1]], %[[TG2]], %[[TG3]], %[[TG4]], %[[TG5]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// Reducing output vector [0, 2]

// CHECK-NEXT: %[[TG6:.*]] = vector.tuple_get %[[ES2]], 1 : tuple<vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG7:.*]] = vector.tuple_get %[[ES3]], 1 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG8:.*]] = vector.tuple_get %[[ES5]], 1 : tuple<vector<2x2xi1>, vector<2x2xi1>>
// CHECK-NEXT:  %[[R1S02:.*]] = vector.contract {indexing_maps = [#map2, #map3, #map0], iterator_types = ["parallel", "reduction", "parallel"]} %[[TG1]], %[[TG6]], %[[TG7]], %[[TG4]], %[[TG8]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// Reducing output vector [2, 0]

// CHECK-NEXT: %[[TG9:.*]] = vector.tuple_get %[[ES1]], 1 : tuple<vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG10:.*]] = vector.tuple_get %[[ES3]], 2 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT: %[[TG11:.*]] = vector.tuple_get %[[ES4]], 1 : tuple<vector<2x2xi1>, vector<2x2xi1>>
// CHECK-NEXT:  %[[R1S20:.*]] = vector.contract {indexing_maps = [#map2, #map3, #map0], iterator_types = ["parallel", "reduction", "parallel"]} %[[TG9]], %[[TG2]], %[[TG10]], %[[TG11]], %[[TG5]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

// Reducing output vector [2, 2]

// CHECK-NEXT: %[[TG12:.*]] = vector.tuple_get %[[ES3]], 3 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
// CHECK-NEXT:  %[[R1S22:.*]] = vector.contract {indexing_maps = [#map2, #map3, #map0], iterator_types = ["parallel", "reduction", "parallel"]} %[[TG9]], %[[TG6]], %[[TG12]], %[[TG11]], %[[TG8]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

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

// CHECK-NEXT: %[[R0:.*]] = vector.contract {indexing_maps = [#map2, #map3, #map0], iterator_types = ["parallel", "reduction", "parallel"]} %[[VTR0]], %[[VTR2]], %[[VTR4]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
// CHECK-NEXT: %[[R1:.*]] = vector.contract {indexing_maps = [#map2, #map3, #map0], iterator_types = ["parallel", "reduction", "parallel"]} %[[VTR0]], %[[VTR3]], %[[VTR5]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
// CHECK-NEXT: %[[R2:.*]] = vector.contract {indexing_maps = [#map2, #map3, #map0], iterator_types = ["parallel", "reduction", "parallel"]} %[[VTR1]], %[[VTR2]], %[[VTR6]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
// CHECK-NEXT: %[[R3:.*]] = vector.contract {indexing_maps = [#map2, #map3, #map0], iterator_types = ["parallel", "reduction", "parallel"]} %[[VTR1]], %[[VTR3]], %[[VTR7]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

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
  %0 = alloc(%arg0, %arg1) : memref<?x?xf32>
  %1 = alloc(%arg0, %arg1) : memref<?x?xf32>
  %2 = alloc(%arg0, %arg1) : memref<?x?xf32>
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

  %0 = alloc() : memref<6x2x1xvector<2x4xf32>>

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
