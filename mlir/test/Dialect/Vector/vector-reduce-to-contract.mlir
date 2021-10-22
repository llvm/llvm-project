// RUN: mlir-opt %s -test-vector-reduction-to-contract-patterns -split-input-file | FileCheck %s

// CHECK-DAG: #[[$map0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[$map1:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>

// CHECK-LABEL: multidimreduction_contract
//  CHECK-NEXT:   %[[C0:.+]] = arith.constant dense<0.000000e+00> : vector<8x16xf32>
//  CHECK-NEXT:   %[[R:.+]] = vector.contract {indexing_maps = [#[[$map0]], #[[$map0]], #[[$map1]]],
//  CHECK-SAME:   iterator_types = ["parallel", "reduction", "parallel"], kind = #vector.kind<add>}
//  CHECK-SAME:   %{{.*}}, %{{.*}}, %[[C0]] : vector<8x32x16xf32>, vector<8x32x16xf32> into vector<8x16xf32>
//  CHECK-NEXT:   return %[[R]] : vector<8x16xf32>
func @multidimreduction_contract(
  %arg0: vector<8x32x16xf32>,%arg1: vector<8x32x16xf32>) -> vector<8x16xf32> {
  %0 = arith.mulf %arg0, %arg1 : vector<8x32x16xf32>
  %1 = vector.multi_reduction #vector.kind<add>, %0 [1] : vector<8x32x16xf32> to vector<8x16xf32>
  return %1 : vector<8x16xf32>
}

// -----

// CHECK-DAG: #[[$map0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[$map1:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>

// CHECK-LABEL: multidimreduction_contract_int
//  CHECK-NEXT:   %[[C0:.+]] = arith.constant dense<0> : vector<8x16xi32>
//  CHECK-NEXT:   %[[R:.+]] = vector.contract {indexing_maps = [#[[$map0]], #[[$map0]], #[[$map1]]],
//  CHECK-SAME:   iterator_types = ["parallel", "reduction", "parallel"], kind = #vector.kind<add>}
//  CHECK-SAME:   %{{.*}}, %{{.*}}, %[[C0]] : vector<8x32x16xi32>, vector<8x32x16xi32> into vector<8x16xi32>
//  CHECK-NEXT:   return %[[R]] : vector<8x16xi32>
func @multidimreduction_contract_int(
  %arg0: vector<8x32x16xi32>,%arg1: vector<8x32x16xi32>) -> vector<8x16xi32> {
  %0 = arith.muli %arg0, %arg1 : vector<8x32x16xi32>
  %1 = vector.multi_reduction #vector.kind<add>, %0 [1] : vector<8x32x16xi32> to vector<8x16xi32>
  return %1 : vector<8x16xi32>
}

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-DAG: #[[$map0:.*]] = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
// CHECK-DAG: #[[$map1:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[$map2:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: contract_transpose
//  CHECK-SAME: (%[[ARG0:.+]]: vector<32x16x8xf32>,
//  CHECK-NEXT:   %[[C0:.+]] = arith.constant dense<0.000000e+00> : vector<8x32xf32>
//  CHECK-NEXT:   %[[R:.+]] = vector.contract {indexing_maps = [#[[$map0]], #[[$map1]], #[[$map2]]],
//  CHECK-SAME:   iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
//  CHECK-SAME:   %[[ARG0]], %{{.*}}, %[[C0]] : vector<32x16x8xf32>, vector<8x32x16xf32> into vector<8x32xf32>
//  CHECK-NEXT:   return %[[R]] : vector<8x32xf32>
func @contract_transpose(
  %arg0: vector<32x16x8xf32>, %arg1: vector<8x32x16xf32>) -> vector<8x32xf32> {
  %cst = arith.constant dense<0.000000e+00> : vector<8x32xf32>
  %0 = vector.transpose %arg0, [2, 0, 1] : vector<32x16x8xf32> to vector<8x32x16xf32>
  %1 = vector.contract {indexing_maps = [#map0, #map0, #map1],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>} %0, %arg1, %cst : vector<8x32x16xf32>, vector<8x32x16xf32> into vector<8x32xf32>
  return %1 : vector<8x32xf32>
}

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-DAG: #[[$map0:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[$map1:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[$map2:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: contract_broadcast
//  CHECK-SAME: (%[[ARG0:.+]]: vector<32x16xf32>,
//  CHECK-NEXT:   %[[C0:.+]] = arith.constant dense<0.000000e+00> : vector<8x32xf32>
//  CHECK-NEXT:   %[[R:.+]] = vector.contract {indexing_maps = [#[[$map0]], #[[$map1]], #[[$map2]]],
//  CHECK-SAME:   iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
//  CHECK-SAME:   %[[ARG0]], %{{.*}}, %[[C0]] : vector<32x16xf32>, vector<8x32x16xf32> into vector<8x32xf32>
//  CHECK-NEXT:   return %[[R]] : vector<8x32xf32>
func @contract_broadcast(
  %arg0: vector<32x16xf32>, %arg1: vector<8x32x16xf32>) -> vector<8x32xf32> {
  %cst = arith.constant dense<0.000000e+00> : vector<8x32xf32>
  %0 = vector.broadcast %arg0 : vector<32x16xf32> to vector<8x32x16xf32>
  %1 = vector.contract {indexing_maps = [#map0, #map0, #map1],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>} %0, %arg1, %cst : vector<8x32x16xf32>, vector<8x32x16xf32> into vector<8x32xf32>
  return %1 : vector<8x32xf32>
}
