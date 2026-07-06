// RUN: mlir-opt %s -test-vector-reduction-to-contract-patterns -split-input-file | FileCheck %s

// TODO: Separate tests for vector.multi_reduction -> vector.contract and
//  * pre-op + vector.contract -> vector.contract,
//  * vector.contract + post-op -> vector.contract.

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>

// CHECK-LABEL: multidimreduction_contract
//  CHECK-SAME: (%[[ARG0:.*]]: vector<8x32x16xf32>, %[[ARG1:.*]]: vector<8x32x16xf32>, %[[ARG2:.*]]: vector<8x16xf32>)
//  CHECK-NEXT:   %[[R:.+]] = vector.contract {indexing_maps = [#[[$MAP0]], #[[$MAP0]], #[[$MAP1]]],
//  CHECK-SAME:   iterator_types = ["parallel", "reduction", "parallel"], kind = #vector.kind<add>}
//  CHECK-SAME:   %[[ARG0]], %[[ARG1]], %[[ARG2]] : vector<8x32x16xf32>, vector<8x32x16xf32> into vector<8x16xf32>
//  CHECK-NEXT:   return %[[R]] : vector<8x16xf32>
func.func @multidimreduction_contract(
  %arg0: vector<8x32x16xf32>,%arg1: vector<8x32x16xf32>, %acc: vector<8x16xf32>) -> vector<8x16xf32> {
  %0 = arith.mulf %arg0, %arg1 : vector<8x32x16xf32>
  %1 = vector.multi_reduction <add>, %0, %acc [1] : vector<8x32x16xf32> to vector<8x16xf32>
  return %1 : vector<8x16xf32> }

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>

// CHECK-LABEL: multidimreduction_contract_int
//  CHECK-SAME: (%[[ARG0:.*]]: vector<8x32x16xi32>, %[[ARG1:.*]]: vector<8x32x16xi32>, %[[ARG2:.*]]: vector<8x16xi32>)
//  CHECK-NEXT:   %[[R:.+]] = vector.contract {indexing_maps = [#[[$MAP0]], #[[$MAP0]], #[[$MAP1]]],
//  CHECK-SAME:   iterator_types = ["parallel", "reduction", "parallel"], kind = #vector.kind<add>}
//  CHECK-SAME:   %[[ARG0]], %[[ARG1]], %[[ARG2]] : vector<8x32x16xi32>, vector<8x32x16xi32> into vector<8x16xi32>
//  CHECK-NEXT:   return %[[R]] : vector<8x16xi32>
func.func @multidimreduction_contract_int(
  %arg0: vector<8x32x16xi32>,%arg1: vector<8x32x16xi32>, %acc: vector<8x16xi32>) -> vector<8x16xi32> {
  %0 = arith.muli %arg0, %arg1 : vector<8x32x16xi32>
  %1 = vector.multi_reduction <add>, %0, %acc [1] : vector<8x32x16xi32> to vector<8x16xi32>
  return %1 : vector<8x16xi32>
}

// -----

//-----------------------------------------------------------------------------
// [Pattern: CombineContractABTranspose]
//-----------------------------------------------------------------------------

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: contract_transpose
//  CHECK-SAME: (%[[ARG0:.+]]: vector<32x16x8xf32>,
//  CHECK-NEXT:   %[[C0:.+]] = arith.constant dense<0.000000e+00> : vector<8x32xf32>
//  CHECK-NEXT:   %[[R:.+]] = vector.contract {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]],
//  CHECK-SAME:   iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
//  CHECK-SAME:   %[[ARG0]], %{{.*}}, %[[C0]] : vector<32x16x8xf32>, vector<8x32x16xf32> into vector<8x32xf32>
//  CHECK-NEXT:   return %[[R]] : vector<8x32xf32>
func.func @contract_transpose(
  %arg0: vector<32x16x8xf32>, %arg1: vector<8x32x16xf32>) -> vector<8x32xf32> {
  %cst = arith.constant dense<0.000000e+00> : vector<8x32xf32>
  %0 = vector.transpose %arg0, [2, 0, 1] : vector<32x16x8xf32> to vector<8x32x16xf32>
  %1 = vector.contract {indexing_maps = [#map0, #map0, #map1],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>} %0, %arg1, %cst : vector<8x32x16xf32>, vector<8x32x16xf32> into vector<8x32xf32>
  return %1 : vector<8x32xf32>
}

// -----

//-----------------------------------------------------------------------------
// [Pattern: CombineContractBroadcast]
//-----------------------------------------------------------------------------

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: contract_broadcast
//  CHECK-SAME: (%[[ARG0:.+]]: vector<32x16xf32>,
//  CHECK-NEXT:   %[[C0:.+]] = arith.constant dense<0.000000e+00> : vector<8x32xf32>
//  CHECK-NEXT:   %[[R:.+]] = vector.contract {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]],
//  CHECK-SAME:   iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
//  CHECK-SAME:   %[[ARG0]], %{{.*}}, %[[C0]] : vector<32x16xf32>, vector<8x32x16xf32> into vector<8x32xf32>
//  CHECK-NEXT:   return %[[R]] : vector<8x32xf32>
func.func @contract_broadcast(
  %arg0: vector<32x16xf32>, %arg1: vector<8x32x16xf32>) -> vector<8x32xf32> {
  %cst = arith.constant dense<0.000000e+00> : vector<8x32xf32>
  %0 = vector.broadcast %arg0 : vector<32x16xf32> to vector<8x32x16xf32>
  %1 = vector.contract {indexing_maps = [#map0, #map0, #map1],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>} %0, %arg1, %cst : vector<8x32x16xf32>, vector<8x32x16xf32> into vector<8x32xf32>
  return %1 : vector<8x32xf32>
}

// -----

// Same as above, but with a mask.

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: contract_broadcast_masked
// CHECK-SAME:      %[[ARG0:.*]]: vector<32x16xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: vector<8x32x16xf32>,
// CHECK-SAME:      %[[MASK:.*]]: vector<8x32x16xi1>) -> vector<8x32xf32> {
// CHECK:           %[[C0:.*]] = arith.constant dense<0.000000e+00> : vector<8x32xf32>
// CHECK:           %[[R:.*]] = vector.mask %[[MASK]] {
// CHECK-SAME:        vector.contract {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]],
// CHECK-SAME:        iterator_types = ["parallel", "parallel", "reduction"],
// CHECK-SAME:        kind = #vector.kind<add>}
// CHECK-SAME:        %[[ARG0]], %[[ARG1]], %[[C0]] : vector<32x16xf32>, vector<8x32x16xf32> into vector<8x32xf32>
// CHECK-SAME       } : vector<8x32x16xi1> -> vector<8x32xf32>
// CHECK:           return %[[R]] : vector<8x32xf32>
func.func @contract_broadcast_masked(
  %arg0: vector<32x16xf32>, %arg1: vector<8x32x16xf32>, %mask: vector<8x32x16xi1>) -> vector<8x32xf32> {
  %cst = arith.constant dense<0.000000e+00> : vector<8x32xf32>
  %0 = vector.broadcast %arg0 : vector<32x16xf32> to vector<8x32x16xf32>
  %1 = vector.mask %mask {
    vector.contract {indexing_maps = [#map0, #map0, #map1],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>
    } %0, %arg1, %cst : vector<8x32x16xf32>, vector<8x32x16xf32> into vector<8x32xf32>
  } : vector<8x32x16xi1> -> vector<8x32xf32>
  return %1 : vector<8x32xf32>
}

// -----

// Same as above, but with a scalable dim.

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: contract_broadcast_masked_scalable
// CHECK-SAME:      %[[ARG0:.*]]: vector<[32]x16xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: vector<8x[32]x16xf32>,
// CHECK-SAME:      %[[MASK:.*]]: vector<8x[32]x16xi1>) -> vector<8x32xf32> {
// CHECK:           %[[C0:.*]] = arith.constant dense<0.000000e+00> : vector<8x32xf32>
// CHECK:           %[[R:.*]] = vector.mask %[[MASK]] {
// CHECK-SAME:        vector.contract {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]],
// CHECK-SAME:        iterator_types = ["parallel", "parallel", "reduction"],
// CHECK-SAME:        kind = #vector.kind<add>}
// CHECK-SAME:        %[[ARG0]], %[[ARG1]], %[[C0]] : vector<[32]x16xf32>, vector<8x[32]x16xf32> into vector<8x32xf32>
// CHECK-SAME       } : vector<8x[32]x16xi1> -> vector<8x32xf32>
// CHECK:           return %[[R]] : vector<8x32xf32>
func.func @contract_broadcast_masked_scalable(
  %arg0: vector<[32]x16xf32>, %arg1: vector<8x[32]x16xf32>, %mask: vector<8x[32]x16xi1>) -> vector<8x32xf32> {
  %cst = arith.constant dense<0.000000e+00> : vector<8x32xf32>
  %0 = vector.broadcast %arg0 : vector<[32]x16xf32> to vector<8x[32]x16xf32>
  %1 = vector.mask %mask {
    vector.contract {indexing_maps = [#map0, #map0, #map1],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>
    } %0, %arg1, %cst : vector<8x[32]x16xf32>, vector<8x[32]x16xf32> into vector<8x32xf32>
  } : vector<8x[32]x16xi1> -> vector<8x32xf32>
  return %1 : vector<8x32xf32>
}

// -----

// Test that CombineContractBroadcast is able to combine a broadcast that
// creates a unit dim that is consumed by a reduction iterator, dropping that
// reduction iterator, as long as there is another reduction iterator left.

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: contract_broadcast_unit_dim_reduction
//  CHECK-SAME: (%[[ARG0:.+]]: vector<8x4xi32>, %[[ARG1:.+]]: vector<8x4xi32>, %[[ARG2:.+]]: vector<8x8xi32>)
//  CHECK: vector.contract
//  CHECK-SAME: indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
//  CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[ARG2]] : vector<8x4xi32>, vector<8x4xi32> into vector<8x8xi32>
func.func @contract_broadcast_unit_dim_reduction(%arg0 : vector<8x4xi32>, %arg1 : vector<8x4xi32>, %arg2 : vector<8x8xi32>) -> vector<8x8xi32> {
    %0 = vector.broadcast %arg0 : vector<8x4xi32> to vector<1x8x4xi32>
    %1 = vector.broadcast %arg1 : vector<8x4xi32> to vector<1x8x4xi32>
    %result = vector.contract {
        indexing_maps = [#map0, #map1, #map2],
        iterator_types = ["reduction", "parallel", "parallel", "reduction"],
        kind = #vector.kind<add>
      } %0, %1, %arg2 : vector<1x8x4xi32>, vector<1x8x4xi32> into vector<8x8xi32>
    return %result : vector<8x8xi32>
}

// -----

// Same as above, but with a mask.

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: contract_broadcast_unit_dim_reduction_masked
//  CHECK-SAME: (%[[ARG0:.+]]: vector<8x4xi32>, %[[ARG1:.+]]: vector<8x4xi32>, %[[ARG2:.+]]: vector<8x8xi32>, %[[MASK:.+]]: vector<1x8x8x4xi1>)
//  CHECK:      %[[MASK_SC:.*]] = vector.shape_cast %[[MASK]] : vector<1x8x8x4xi1> to vector<8x8x4xi1>
//  CHECK:      %[[R:.*]] = vector.mask %[[MASK_SC]] {
//  CHECK-SAME: vector.contract
//  CHECK-SAME: indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
//  CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[ARG2]] : vector<8x4xi32>, vector<8x4xi32> into vector<8x8xi32>
func.func @contract_broadcast_unit_dim_reduction_masked(%arg0 : vector<8x4xi32>, %arg1 : vector<8x4xi32>, %arg2 : vector<8x8xi32>, %mask: vector<1x8x8x4xi1>) -> vector<8x8xi32> {
    %0 = vector.broadcast %arg0 : vector<8x4xi32> to vector<1x8x4xi32>
    %1 = vector.broadcast %arg1 : vector<8x4xi32> to vector<1x8x4xi32>
    %result = vector.mask %mask {
      vector.contract {
        indexing_maps = [#map0, #map1, #map2],
        iterator_types = ["reduction", "parallel", "parallel", "reduction"],
        kind = #vector.kind<add>
      } %0, %1, %arg2 : vector<1x8x4xi32>, vector<1x8x4xi32> into vector<8x8xi32>
    } : vector<1x8x8x4xi1> -> vector<8x8xi32>
    return %result : vector<8x8xi32>
}

// -----

// Same as above, but with a scalable dim.

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: contract_broadcast_unit_dim_reduction_masked_scalable
//  CHECK-SAME: (%[[ARG0:.+]]: vector<8x4xi32>, %[[ARG1:.+]]: vector<[8]x4xi32>, %[[ARG2:.+]]: vector<8x[8]xi32>, %[[MASK:.+]]: vector<1x8x[8]x4xi1>)
//  CHECK:      %[[MASK_SC:.*]] = vector.shape_cast %[[MASK]] : vector<1x8x[8]x4xi1> to vector<8x[8]x4xi1>
//  CHECK:      %[[R:.*]] = vector.mask %[[MASK_SC]] {
//  CHECK-SAME: vector.contract
//  CHECK-SAME: indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
//  CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[ARG2]] : vector<8x4xi32>, vector<[8]x4xi32> into vector<8x[8]xi32>
func.func @contract_broadcast_unit_dim_reduction_masked_scalable(%arg0 : vector<8x4xi32>, %arg1 : vector<[8]x4xi32>, %arg2 : vector<8x[8]xi32>, %mask: vector<1x8x[8]x4xi1>) -> vector<8x[8]xi32> {
    %0 = vector.broadcast %arg0 : vector<8x4xi32> to vector<1x8x4xi32>
    %1 = vector.broadcast %arg1 : vector<[8]x4xi32> to vector<1x[8]x4xi32>
    %result = vector.mask %mask {
      vector.contract {
        indexing_maps = [#map0, #map1, #map2],
        iterator_types = ["reduction", "parallel", "parallel", "reduction"],
        kind = #vector.kind<add>
      } %0, %1, %arg2 : vector<1x8x4xi32>, vector<1x[8]x4xi32> into vector<8x[8]xi32>
    } : vector<1x8x[8]x4xi1> -> vector<8x[8]xi32>
    return %result : vector<8x[8]xi32>
}

// -----
// Test that CombineContractBroadcast will not combine a broadcast that creates
// a non-unit dim that is consumed by a reduction iterator.
// Moreover, the affine_map's are permuting the position of that reduction
// iterator with the position of a parallel iterator to ensure that
// the logic guarding that case does not mix up dimensions here.

#map0 = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
// CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>

// CHECK-LABEL: contract_broadcast_non_unit_dim_reduction_with_permutation
//  CHECK-SAME: (%[[ARG0:.+]]: vector<8x4xi32>, %[[ARG1:.+]]: vector<8x4xi32>, %[[ARG2:.+]]: vector<8x8xi32>)
//  CHECK: %[[BROADCAST0:.+]] = vector.broadcast %[[ARG0]] : vector<8x4xi32> to vector<2x8x4xi32>
//  CHECK: %[[BROADCAST1:.+]] = vector.broadcast %[[ARG1]] : vector<8x4xi32> to vector<2x8x4xi32>
//  CHECK: vector.contract
//  CHECK-SAME: indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME: iterator_types = ["parallel", "reduction", "parallel", "reduction"]
//  CHECK-SAME: %[[BROADCAST0]], %[[BROADCAST1]], %[[ARG2]] : vector<2x8x4xi32>, vector<2x8x4xi32> into vector<8x8xi32>
func.func @contract_broadcast_non_unit_dim_reduction_with_permutation(%arg0 : vector<8x4xi32>, %arg1 : vector<8x4xi32>, %arg2 : vector<8x8xi32>) -> vector<8x8xi32> {
    %0 = vector.broadcast %arg0 : vector<8x4xi32> to vector<2x8x4xi32>
    %1 = vector.broadcast %arg1 : vector<8x4xi32> to vector<2x8x4xi32>
    %result = vector.contract {
        indexing_maps = [#map0, #map1, #map2],
        iterator_types = ["parallel", "reduction", "parallel", "reduction"],
        kind = #vector.kind<add>
      } %0, %1, %arg2 : vector<2x8x4xi32>, vector<2x8x4xi32> into vector<8x8xi32>
    return %result : vector<8x8xi32>
}

// -----

// Test that CombineContractBroadcast is not combining this case, as that would
// result in dropping this contract's only reduction iterator.

#map0 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>

// CHECK-LABEL: contract_broadcast_unit_dim_reduction_as_only_reduction
//  CHECK-SAME: (%[[ARG0:.+]]: vector<8xi32>, %[[ARG1:.+]]: vector<8xi32>, %[[ARG2:.+]]: vector<8x8xi32>)
//  CHECK: %[[BROADCAST0:.+]] = vector.broadcast %[[ARG0]] : vector<8xi32> to vector<1x8xi32>
//  CHECK: %[[BROADCAST1:.+]] = vector.broadcast %[[ARG1]] : vector<8xi32> to vector<1x8xi32>
//  CHECK: vector.contract
//  CHECK-SAME: indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME: iterator_types = ["reduction", "parallel", "parallel"]
//  CHECK-SAME: %[[BROADCAST0]], %[[BROADCAST1]], %[[ARG2]] : vector<1x8xi32>, vector<1x8xi32> into vector<8x8xi32>
func.func @contract_broadcast_unit_dim_reduction_as_only_reduction(%arg0 : vector<8xi32>, %arg1 : vector<8xi32>, %arg2 : vector<8x8xi32>) -> vector<8x8xi32> {
    %0 = vector.broadcast %arg0 : vector<8xi32> to vector<1x8xi32>
    %1 = vector.broadcast %arg1 : vector<8xi32> to vector<1x8xi32>
    %result = vector.contract {
        indexing_maps = [#map0, #map1, #map2],
        iterator_types = ["reduction", "parallel", "parallel"],
        kind = #vector.kind<add>
      } %0, %1, %arg2 : vector<1x8xi32>, vector<1x8xi32> into vector<8x8xi32>
    return %result : vector<8x8xi32>
}

// -----

// Test that CombineContractBroadcast is not combining this case, as that would
// result in a dimension being unused in the LHS and RHS maps, which is illegal.

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1)>

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1, d2) -> (d1)>

// CHECK-LABEL: contract_broadcast_dimension_would_go_unused_in_lhs_rhs
//  CHECK-SAME: (%[[ARG0:.+]]: vector<1x2xi32>, %[[ARG1:.+]]: vector<2xi32>, %[[ARG2:.+]]: vector<1xi32>)
//  CHECK: %[[BROADCAST1:.+]] = vector.broadcast %[[ARG1]] : vector<2xi32> to vector<1x1x2xi32>
//  CHECK: vector.contract
//  CHECK-SAME: indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME: iterator_types = ["reduction", "parallel", "reduction"]
//  CHECK-SAME: %[[ARG0]], %[[BROADCAST1]], %[[ARG2]] : vector<1x2xi32>, vector<1x1x2xi32> into vector<1xi32>

func.func @contract_broadcast_dimension_would_go_unused_in_lhs_rhs(%arg0 : vector<1x2xi32>, %arg1 : vector<2xi32>, %arg2 : vector<1xi32>) -> vector<1xi32> {
  %1 = vector.broadcast %arg1 : vector<2xi32> to vector<1x1x2xi32>
  %result = vector.contract {
      indexing_maps = [#map0, #map1, #map2],
      iterator_types = ["reduction", "parallel", "reduction"],
      kind = #vector.kind<add>
  } %arg0, %1, %arg2 : vector<1x2xi32>, vector<1x1x2xi32> into vector<1xi32>
  return  %result : vector<1xi32>
}

// -----

// Test that CombineContractBroadcast is not combining this case, as that would
// result in a vector.contract without a reduction dimention pair, as the only
// reduction dimension would be used by only one side among LHS, RHS.
// This is arguably a convoluted edge case (the affine_maps here look weird!)
// but it is something that we actually ran into from linalg.matmul tests that
// were exercising 1x1 shapes, and using various drop-unit-dims patterns.

#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: contract_broadcast_would_have_no_reduction_dim_pair
//  CHECK-SAME: (%[[ARG0:.+]]: vector<1xf32>, %[[ARG1:.+]]: vector<1xf32>, %[[ARG2:.+]]: vector<1xf32>)
//  CHECK: %[[BROADCAST1:.+]] = vector.broadcast %[[ARG1]] : vector<1xf32> to vector<1x1xf32>
//  CHECK: vector.contract
//  CHECK-SAME: indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME: iterator_types = ["parallel", "reduction"]
//  CHECK-SAME: %[[ARG0]], %[[BROADCAST1]], %[[ARG2]] : vector<1xf32>, vector<1x1xf32> into vector<1xf32>

func.func @contract_broadcast_would_have_no_reduction_dim_pair(%arg0 : vector<1xf32>, %arg1 : vector<1xf32>, %arg2 : vector<1xf32>) -> vector<1xf32> {
  %1 = vector.broadcast %arg1 : vector<1xf32> to vector<1x1xf32>
  %result = vector.contract {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "reduction"],
    kind = #vector.kind<add>
  } %arg0, %1, %arg2 : vector<1xf32>, vector<1x1xf32> into vector<1xf32>
  return %result : vector<1xf32>
}


// -----

//-----------------------------------------------------------------------------
// [Pattern: CombineContractResultTranspose]
//-----------------------------------------------------------------------------

// CHECK-DAG: #[[$LHS_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
// CHECK-DAG: #[[$RHS_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
// CHECK-DAG: #[[$ACC_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>

// CHECK-LABEL: func.func @contract_result_transpose
//  CHECK-SAME: (%[[LHS:.+]]: vector<2x4x4xf32>, %[[RHS:.+]]: vector<4x8xf32>, %[[ACC:.+]]: vector<2x8x4xf32>)
//       CHECK:   %[[CONTRACT:.+]] = vector.contract
//  CHECK-SAME:     indexing_maps = [#[[$LHS_MAP]], #[[$RHS_MAP]], #[[$ACC_MAP]]]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "reduction"]
//  CHECK-SAME:     kind = #vector.kind<add>
//  CHECK-SAME:     %[[LHS]], %[[RHS]], %[[ACC]]
//       CHECK:   return %[[CONTRACT]]
func.func @contract_result_transpose(%lhs : vector<2x4x4xf32>, %rhs: vector<4x8xf32>, %acc: vector<2x8x4xf32>) -> vector<2x8x4xf32> {
  %accT = vector.transpose %acc, [0, 2, 1] : vector<2x8x4xf32> to vector<2x4x8xf32>
  %contract = vector.contract {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>,
      affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
    ],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>
  } %lhs, %rhs, %accT : vector<2x4x4xf32>, vector<4x8xf32> into vector<2x4x8xf32>
  %resT = vector.transpose %contract, [0, 2, 1] : vector<2x4x8xf32> to vector<2x8x4xf32>
  return %resT : vector<2x8x4xf32>
}
