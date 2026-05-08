// The following test examples of linalg category ops lowered to linalg.generic
// and then lifted back up to category op.

// RUN: mlir-opt %s -split-input-file -linalg-morph-ops=category-to-generic \
// RUN: | mlir-opt -split-input-file -linalg-morph-ops=generic-to-category \
// RUN: | FileCheck %s

func.func @unary_ops(%A: memref<7x14x21xf32>, %Out: memref<7x14x21xf32>) {
  linalg.elementwise kind=#linalg.elementwise_kind<exp>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<log>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<abs>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<ceil>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<floor>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<negf>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<reciprocal>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<round>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<sqrt>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<rsqrt>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<square>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<tanh>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<erf>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  return
}

// CHECK-LABEL: unary_ops
// CHECK-SAME: %[[A:.+]]: memref<7x14x21xf32>, %[[OUT:.+]]: memref<7x14x21xf32>)
// CHECK-NOT: linalg.generic
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<exp>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<log>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<abs>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<ceil>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<floor>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<negf>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<reciprocal>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<round>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<sqrt>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<rsqrt>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<square>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<tanh>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<erf>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)

// -----

func.func @unary_ops_non_identity(%A: tensor<?xf32>, %Out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.elementwise
    kind=#linalg.elementwise_kind<log>
    indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>]
    ins(%A : tensor<?xf32>) outs(%Out : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-DAG: #[[MAP_BC:.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG: #[[MAP_TP:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK: unary_ops_non_identity
// CHECK-SAME: %[[A:.+]]: tensor<?xf32>, %[[OUT:.+]]: tensor<?x?xf32>)
// CHECK-NOT: linalg.generic
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<log>
// CHECK-SAME: indexing_maps = [#[[MAP_BC]], #[[MAP_TP]]]
// CHECK-SAME: ins(%[[A]] : tensor<?xf32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @contract_matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
    %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.contract indexing_maps = [#map, #map1, #map2]
    ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-DAG: #[[$MAP_A:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$MAP_B:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: contract_matmul
// CHECK-SAME: %[[A:.+]]: tensor<?x?xf32>, %[[B:.+]]: tensor<?x?xf32>,
// CHECK-SAME: %[[OUT:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.contract
// CHECK-SAME: indexing_maps = {{\[}}#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]{{\]}}
// CHECK-SAME: ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>


func.func @contract_matmul_memref(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
    %arg2: memref<?x?xf32>) {
  linalg.contract indexing_maps = [#map, #map1, #map2]
    ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>)
    outs(%arg2 : memref<?x?xf32>)
  return
}

// CHECK-LABEL: contract_matmul_memref
// CHECK-SAME: %[[A:.+]]: memref<?x?xf32>, %[[B:.+]]: memref<?x?xf32>,
// CHECK-SAME: %[[OUT:.+]]: memref<?x?xf32>)
// CHECK-NOT: linalg.generic
// CHECK: linalg.contract
// CHECK-SAME: indexing_maps = {{\[}}#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]{{\]}}
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<?x?xf32>, memref<?x?xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<?x?xf32>)

func.func @contract_matmul_bitcast_int_to_float(%arg0: tensor<16x8xi32>,
    %arg1: tensor<8x32xi32>, %arg2: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.contract indexing_maps = [#map, #map1, #map2]
    ins(%arg0, %arg1 : tensor<16x8xi32>, tensor<8x32xi32>)
    outs(%arg2 : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: contract_matmul_bitcast_int_to_float
// CHECK-SAME: %[[A:.+]]: tensor<16x8xi32>, %[[B:.+]]: tensor<8x32xi32>,
// CHECK-SAME: %[[OUT:.+]]: tensor<16x32xf32>) -> tensor<16x32xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.contract
// CHECK-SAME: indexing_maps = {{\[}}#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]{{\]}}
// CHECK-NOT: cast =
// CHECK-SAME: ins(%[[A]], %[[B]] : tensor<16x8xi32>, tensor<8x32xi32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<16x32xf32>) -> tensor<16x32xf32>

func.func @contract_matmul_unsigned_cast_float(%arg0: tensor<16x8xi16>,
    %arg1: tensor<8x32xi16>, %arg2: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.contract indexing_maps = [#map, #map1, #map2]
    {cast = #linalg.type_fn<cast_unsigned>}
    ins(%arg0, %arg1 : tensor<16x8xi16>, tensor<8x32xi16>)
    outs(%arg2 : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: contract_matmul_unsigned_cast_float
// CHECK-SAME: %[[A:.+]]: tensor<16x8xi16>, %[[B:.+]]: tensor<8x32xi16>,
// CHECK-SAME: %[[OUT:.+]]: tensor<16x32xf32>) -> tensor<16x32xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.contract
// CHECK-SAME: indexing_maps = {{\[}}#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]{{\]}}
// CHECK-SAME: cast = #linalg.type_fn<cast_unsigned>
// CHECK-SAME: ins(%[[A]], %[[B]] : tensor<16x8xi16>, tensor<8x32xi16>)
// CHECK-SAME: outs(%[[OUT]] : tensor<16x32xf32>) -> tensor<16x32xf32>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d2, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
func.func @contract_multi_reduction(%arg0: tensor<10x20x30xf32>,
    %arg1: tensor<30x20x40xf32>, %arg2: tensor<10x40xf32>) -> tensor<10x40xf32> {
  %0 = linalg.contract indexing_maps = [#map, #map1, #map2]
    ins(%arg0, %arg1 : tensor<10x20x30xf32>, tensor<30x20x40xf32>)
    outs(%arg2 : tensor<10x40xf32>) -> tensor<10x40xf32>
  return %0 : tensor<10x40xf32>
}

// CHECK-DAG: #[[$MAP_A:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[$MAP_B:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d2, d1)>
// CHECK-DAG: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

// CHECK-LABEL: contract_multi_reduction
// CHECK-NOT: linalg.generic
// CHECK: linalg.contract
// CHECK-SAME: indexing_maps = {{\[}}#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]{{\]}}
