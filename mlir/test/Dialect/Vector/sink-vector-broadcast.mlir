// RUN: mlir-opt %s -test-sink-vector-broadcast -split-input-file | FileCheck %s

// CHECK-LABEL:   func.func @broadcast_scalar(
// CHECK-SAME:     %[[ARG_0:.*]]: index, %[[ARG_1:.*]]: index) -> vector<1x4xindex> {
// CHECK:           %[[ADD:.*]] = arith.addi %[[ARG_0]], %[[ARG_1]] : index
// CHECK:           %[[BCAST:.*]] = vector.broadcast %[[ADD]] : index to vector<1x4xindex>
// CHECK:           return %[[BCAST]] : vector<1x4xindex>
// CHECK:         }

func.func @broadcast_scalar( %arg1: index, %arg2: index) -> vector<1x4xindex> {
  %0 = vector.broadcast %arg1 : index to vector<1x4xindex>
  %1 = vector.broadcast %arg2 : index to vector<1x4xindex>
  %2 = arith.addi %0, %1 : vector<1x4xindex>
  return %2 : vector<1x4xindex>
}

// -----

// CHECK-LABEL:   func.func @broadcast_vector(
// CHECK-SAME:      %[[ARG_0:.*]]: vector<4xf32>,
// CHECK-SAME:      %[[ARG_1:.*]]: vector<4xf32>) -> vector<3x4xf32> {
// CHECK:           %[[ADDF:.*]] = arith.addf %[[ARG_0]], %[[ARG_1]] : vector<4xf32>
// CHECK:           %[[BCAST:.*]] = vector.broadcast %[[ADDF]] : vector<4xf32> to vector<3x4xf32>
// CHECK:           return %[[BCAST]] : vector<3x4xf32>
// CHECK:         }

func.func @broadcast_vector( %arg1: vector<4xf32>, %arg2: vector<4xf32>) -> vector<3x4xf32> {
  %arg1_bcast = vector.broadcast %arg1 : vector<4xf32> to vector<3x4xf32>
  %arg2_bcast = vector.broadcast %arg2 : vector<4xf32> to vector<3x4xf32>
  %2 = arith.addf %arg1_bcast, %arg2_bcast : vector<3x4xf32>
  return %2 : vector<3x4xf32>
}
// -----

// CHECK-LABEL:   func.func @broadcast_vector_and_scalar(
// CHECK-SAME:      %[[ARG_0:.*]]: i32,
// CHECK-SAME:      %[[ARG_1:.*]]: vector<4xi32>) -> vector<4xi32> {
// CHECK:           %[[BCAST:.*]] = vector.broadcast %[[ARG_0]] : i32 to vector<4xi32>
// CHECK:           %[[ADD:.*]] = arith.addi %[[BCAST]], %[[ARG_1]] : vector<4xi32>
// CHECK:           return %[[ADD]] : vector<4xi32>
// CHECK:         }

func.func @broadcast_vector_and_scalar( %arg1: i32, %arg2: vector<4xi32>) -> vector<4xi32> {
  %arg1_bcast = vector.broadcast %arg1 : i32 to vector<4xi32>
  %2 = arith.addi %arg1_bcast, %arg2 : vector<4xi32>
  return %2 : vector<4xi32>
}

// -----

#matmat_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
#matmat_trait = {
  indexing_maps = #matmat_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL:   func.func @broadcast_not_elementwise() -> vector<2x2xf32> {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant dense<1.000000e+00> : vector<2x2xf32>
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant dense<2.000000e+00> : vector<2x2xf32>
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant dense<3.000000e+00> : vector<2x2xf32>
// CHECK:           %[[VAL_3:.*]] = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
func.func @broadcast_not_elementwise() -> vector<2x2xf32> {
  %f1 = arith.constant 1.0: f32
  %f2 = arith.constant 2.0: f32
  %f3 = arith.constant 3.0: f32

  %A = vector.broadcast %f1 : f32 to vector<2x2xf32>
  %B = vector.broadcast %f2 : f32 to vector<2x2xf32>
  %C = vector.broadcast %f3 : f32 to vector<2x2xf32>
  %mm1 = vector.contract #matmat_trait %A, %B, %C
    : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

  return %mm1 : vector<2x2xf32>
}
