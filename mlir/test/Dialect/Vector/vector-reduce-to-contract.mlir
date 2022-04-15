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
  %1 = vector.multi_reduction <add>, %0 [1] : vector<8x32x16xf32> to vector<8x16xf32>
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
  %1 = vector.multi_reduction <add>, %0 [1] : vector<8x32x16xi32> to vector<8x16xi32>
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

//===----------------------------------------------------------------------===//
// Reorder casting ops and vector ops. The casting ops have almost identical
// pattern, so only arith.extsi op is tested.
//===----------------------------------------------------------------------===//

// -----

func @broadcast_vector_extsi(%a : vector<4xi8>) -> vector<2x4xi32> {
  // CHECK: %[[EXT:.+]] = arith.extsi %{{.+}} : vector<4xi8> to vector<4xi32>
  // CHECK: vector.broadcast %[[EXT:.+]] : vector<4xi32> to vector<2x4xi32>
  %b = vector.broadcast %a : vector<4xi8> to vector<2x4xi8>
  %r = arith.extsi %b : vector<2x4xi8> to vector<2x4xi32>
  return %r : vector<2x4xi32>
}

// -----

func @broadcast_scalar_extsi(%a : i8) -> vector<2x4xi32> {
  // CHECK: %[[EXT:.+]] = arith.extsi %{{.+}} : i8 to i32
  // CHECK: vector.broadcast %[[EXT]] : i32 to vector<2x4xi32>
  %b = vector.broadcast %a : i8 to vector<2x4xi8>
  %r = arith.extsi %b : vector<2x4xi8> to vector<2x4xi32>
  return %r : vector<2x4xi32>
}

// -----

func @transpose_extsi(%a : vector<4x2xi8>) -> vector<2x4xi32> {
  // CHECK: %[[EXT:.+]] = arith.extsi %{{.+}} : vector<4x2xi8> to vector<4x2xi32>
  // CHECK: vector.transpose %[[EXT]], [1, 0] : vector<4x2xi32> to vector<2x4xi32>
  %b = vector.transpose %a, [1, 0]: vector<4x2xi8> to vector<2x4xi8>
  %r = arith.extsi %b : vector<2x4xi8> to vector<2x4xi32>
  return %r : vector<2x4xi32>
}

//===----------------------------------------------------------------------===//
// Reorder elementwise ops and vector ops.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @transpose_elementwise_same_type
//  CHECK-SAME: (%[[A:.+]]: vector<4x2xf32>, %[[B:.+]]: vector<4x2xf32>)
//       CHECK:   %[[ADD:.+]] = arith.addf %[[A]], %[[B]] : vector<4x2xf32>
//       CHECK:   %[[T:.+]] = vector.transpose %[[ADD]], [1, 0]
//       CHECK:   return %[[T]]

func @transpose_elementwise_same_type(%a : vector<4x2xf32>, %b : vector<4x2xf32>) -> vector<2x4xf32> {
  %at = vector.transpose %a, [1, 0]: vector<4x2xf32> to vector<2x4xf32>
  %bt = vector.transpose %b, [1, 0]: vector<4x2xf32> to vector<2x4xf32>
  %r = arith.addf %at, %bt : vector<2x4xf32>
  return %r : vector<2x4xf32>
}

// -----

// CHECK-LABEL: func @transpose_elementwise_diff_operand_types
//  CHECK-SAME: (%[[COND:.+]]: vector<4x2xi1>, %[[A:.+]]: vector<4x2xf32>, %[[B:.+]]: vector<4x2xf32>)
//       CHECK:   %[[S:.+]] = arith.select %[[COND]], %[[A]], %[[B]] : vector<4x2xi1>, vector<4x2xf32>
//       CHECK:   %[[T:.+]] = vector.transpose %[[S]], [1, 0] : vector<4x2xf32> to vector<2x4xf32>
//       CHECK:   return %[[T]]
func @transpose_elementwise_diff_operand_types(%cond: vector<4x2xi1>, %a : vector<4x2xf32>, %b : vector<4x2xf32>) -> vector<2x4xf32> {
  %condt = vector.transpose %cond, [1, 0]: vector<4x2xi1> to vector<2x4xi1>
  %at = vector.transpose %a, [1, 0]: vector<4x2xf32> to vector<2x4xf32>
  %bt = vector.transpose %b, [1, 0]: vector<4x2xf32> to vector<2x4xf32>
  %r = arith.select %condt, %at, %bt : vector<2x4xi1>, vector<2x4xf32>
  return %r : vector<2x4xf32>
}

// -----

// CHECK-LABEL: func @transpose_elementwise_diff_operand_result_type
//  CHECK-SAME: (%[[A:.+]]: vector<4x2xf32>, %[[B:.+]]: vector<4x2xf32>)
//       CHECK:   %[[CMP:.+]] = arith.cmpf olt, %[[A]], %[[B]] : vector<4x2xf32>
//       CHECK:   %[[T:.+]] = vector.transpose %[[CMP]], [1, 0] : vector<4x2xi1> to vector<2x4xi1>
//       CHECK:   return %[[T]]
func @transpose_elementwise_diff_operand_result_type(%a : vector<4x2xf32>, %b : vector<4x2xf32>) -> vector<2x4xi1> {
  %at = vector.transpose %a, [1, 0]: vector<4x2xf32> to vector<2x4xf32>
  %bt = vector.transpose %b, [1, 0]: vector<4x2xf32> to vector<2x4xf32>
  %r = arith.cmpf olt, %at, %bt : vector<2x4xf32>
  return %r : vector<2x4xi1>
}

// -----

// CHECK-LABEL: func @transpose_elementwise_splat_constant
//  CHECK-SAME: (%[[A:.+]]: vector<4x6x3x2xf32>)
//       CHECK:   %[[B:.+]] = arith.constant dense<5.000000e+00> : vector<4x6x3x2xf32>
//       CHECK:   %[[ADD:.+]] = arith.addf %[[A]], %[[B]] : vector<4x6x3x2xf32>
//       CHECK:   %[[T:.+]] = vector.transpose %[[ADD]], [1, 0, 3, 2] : vector<4x6x3x2xf32> to vector<6x4x2x3xf32>
//       CHECK:   return %[[T:.+]] : vector<6x4x2x3xf32>

func @transpose_elementwise_splat_constant(%a : vector<4x6x3x2xf32>) -> vector<6x4x2x3xf32> {
  %b = arith.constant dense<5.0> : vector<6x4x2x3xf32>
  %at = vector.transpose %a, [1, 0, 3, 2]: vector<4x6x3x2xf32> to vector<6x4x2x3xf32>
  %r = arith.addf %at, %b : vector<6x4x2x3xf32>
  return %r : vector<6x4x2x3xf32>
}

// -----

// CHECK-LABEL: func @transpose_elementwise_diff_map
//       CHECK:   vector.transpose
//       CHECK:   vector.transpose
//       CHECK:   arith.addf
func @transpose_elementwise_diff_map(%a : vector<4x6x3x2xf32>, %b: vector<6x2x4x3xf32>) -> vector<6x4x2x3xf32> {
  %at = vector.transpose %a, [1, 0, 3, 2]: vector<4x6x3x2xf32> to vector<6x4x2x3xf32>
  %bt = vector.transpose %b, [0, 2, 1, 3]: vector<6x2x4x3xf32> to vector<6x4x2x3xf32>
  %r = arith.addf %at, %bt : vector<6x4x2x3xf32>
  return %r : vector<6x4x2x3xf32>
}
