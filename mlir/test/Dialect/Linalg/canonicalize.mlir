// RUN: mlir-opt %s -canonicalize -split-input-file | FileCheck %s

// CHECK-LABEL: func @memref_cast(
func @memref_cast(%a: index, %b: index) -> memref<?x?xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c8 = constant 8 : index
  %c16 = constant 16 : index
  %1 = memref.alloc (%b) : memref<?xi8>
  %2 = memref.view %1[%c0][] : memref<?xi8> to memref<16x16xf32>
  %3 = memref.cast %2 : memref<16x16xf32> to memref<?x?xf32>

  // CHECK:  linalg.matmul ins({{.*}}memref<16x16xf32>, memref<16x16xf32>) outs({{.*}}memref<16x16xf32>)
  linalg.matmul ins(%3, %3: memref<?x?xf32>, memref<?x?xf32>)
               outs(%3: memref<?x?xf32>)
  return %3: memref<?x?xf32>
}

// -----

func @collapsing_tensor_reshapes(%arg0 : tensor<?x?x?x?x?xf32>) -> tensor<?x?xf32>
{
  %0 = linalg.tensor_reshape %arg0
         [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>,
          affine_map<(d0, d1, d2, d3, d4) -> (d2)>,
          affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>] :
       tensor<?x?x?x?x?xf32> into tensor<?x?x?xf32>
  %1 = linalg.tensor_reshape %0
         [affine_map<(d0, d1, d2) -> (d0, d1)>,
          affine_map<(d0, d1, d2) -> (d2)>] :
       tensor<?x?x?xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
//   CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>
// CHECK-LABEL: collapsing_tensor_reshapes
//       CHECK:   linalg.tensor_reshape %{{.*}} [#[[$MAP0]], #[[$MAP1]]]
//   CHECK-NOT:   linalg.tensor_reshape

// -----

func @collapsing_tensor_reshapes_to_zero_dim(%arg0 : tensor<1x1x1xf32>)
                                             -> tensor<f32> {
  %0 = linalg.tensor_reshape %arg0 [affine_map<(d0, d1, d2) -> (d0, d1, d2)>] :
       tensor<1x1x1xf32> into tensor<1xf32>
  %1 = linalg.tensor_reshape %0 [] : tensor<1xf32> into tensor<f32>
  return %1 : tensor<f32>
}
// CHECK-LABEL: collapsing_tensor_reshapes_to_zero
//       CHECK:   linalg.tensor_reshape %{{.*}} []
//  CHECK-SAME:     tensor<1x1x1xf32> into tensor<f32>

// -----

func @collapsing_memref_reshapes_to_zero_dim(%arg0 : memref<1x1x1xf32>)
                                             -> memref<f32> {
  %0 = linalg.reshape %arg0 [affine_map<(d0, d1, d2) -> (d0, d1, d2)>] :
       memref<1x1x1xf32> into memref<1xf32>
  %1 = linalg.reshape %0 [] : memref<1xf32> into memref<f32>
  return %1 : memref<f32>
}
// CHECK-LABEL: collapsing_memref_reshapes_to_zero
//       CHECK:   linalg.reshape %{{.*}} []
//  CHECK-SAME:     memref<1x1x1xf32> into memref<f32>

// -----

func @expanding_tensor_reshapes(%arg0 : tensor<?x?xf32>) -> tensor<?x6x4x?x5xf32>
{
  %0 = linalg.tensor_reshape %arg0
         [affine_map<(d0, d1, d2) -> (d0, d1)>,
          affine_map<(d0, d1, d2) -> (d2)>] :
       tensor<?x?xf32> into tensor<?x4x?xf32>
  %1 = linalg.tensor_reshape %0
         [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>,
          affine_map<(d0, d1, d2, d3, d4) -> (d2)>,
          affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>] :
       tensor<?x4x?xf32> into tensor<?x6x4x?x5xf32>
  return %1 : tensor<?x6x4x?x5xf32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
//   CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>
// CHECK-LABEL: expanding_tensor_reshapes
//       CHECK:   linalg.tensor_reshape %{{.*}} [#[[$MAP0]], #[[$MAP1]]]
//   CHECK-NOT:   linalg.tensor_reshape

// -----

func @collapsing_memref_reshapes(%arg0 : memref<?x?x?x?x?xf32>) -> memref<?x?xf32>
{
  %0 = linalg.reshape %arg0
         [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>,
          affine_map<(d0, d1, d2, d3, d4) -> (d2)>,
          affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>] :
       memref<?x?x?x?x?xf32> into memref<?x?x?xf32>
  %1 = linalg.reshape %0
         [affine_map<(d0, d1, d2) -> (d0, d1)>,
          affine_map<(d0, d1, d2) -> (d2)>] :
       memref<?x?x?xf32> into memref<?x?xf32>
  return %1 : memref<?x?xf32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
//   CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>
// CHECK-LABEL: collapsing_memref_reshapes
//       CHECK:   linalg.reshape %{{.*}} [#[[$MAP0]], #[[$MAP1]]]
//   CHECK-NOT:   linalg.reshape

// -----

func @expanding_memref_reshapes(%arg0 : memref<?x?xf32>) -> memref<?x6x4x5x?xf32>
{
  %0 = linalg.reshape %arg0
         [affine_map<(d0, d1, d2) -> (d0, d1)>,
          affine_map<(d0, d1, d2) -> (d2)>] :
       memref<?x?xf32> into memref<?x4x?xf32>
  %1 = linalg.reshape %0
         [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>,
          affine_map<(d0, d1, d2, d3, d4) -> (d2)>,
          affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>] :
       memref<?x4x?xf32> into memref<?x6x4x5x?xf32>
  return %1 : memref<?x6x4x5x?xf32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
//   CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>
// CHECK-LABEL: expanding_memref_reshapes
//       CHECK:   linalg.reshape %{{.*}} [#[[$MAP0]], #[[$MAP1]]]
//   CHECK-NOT:   linalg.reshape

// -----

func @expanding_tensor_reshapes_to_zero_dim(%arg0 : tensor<f32>)
                                             -> tensor<1x1x1xf32> {
  %0 = linalg.tensor_reshape %arg0 [] : tensor<f32> into tensor<1xf32>
  %1 = linalg.tensor_reshape %0 [affine_map<(d0, d1, d2) -> (d0, d1, d2)>] :
       tensor<1xf32> into tensor<1x1x1xf32>
  return %1 : tensor<1x1x1xf32>
}
// CHECK-LABEL: expanding_tensor_reshapes_to_zero
//       CHECK:   linalg.tensor_reshape %{{.*}} []
//  CHECK-SAME:     tensor<f32> into tensor<1x1x1xf32>

// -----

func @expanding_memref_reshapes_to_zero_dim(%arg0 : memref<f32>)
                                             -> memref<1x1x1xf32> {
  %0 = linalg.reshape %arg0 [] : memref<f32> into memref<1xf32>
  %1 = linalg.reshape %0
         [affine_map<(d0, d1, d2) -> (d0, d1, d2)>] :
       memref<1xf32> into memref<1x1x1xf32>
  return %1 : memref<1x1x1xf32>
}
// CHECK-LABEL: expanding_memref_reshapes_to_zero
//       CHECK:   linalg.reshape %{{.*}} []
//  CHECK-SAME:     memref<f32> into memref<1x1x1xf32>

// -----

func @fold_tensor_reshape(%arg0 : tensor<12x4xf32>) -> tensor<12x4xf32>
{
  %0 = linalg.tensor_reshape %arg0
         [affine_map<(d0, d1, d2) -> (d0, d1)>,
          affine_map<(d0, d1, d2) -> (d2)>] :
       tensor<12x4xf32> into tensor<3x4x4xf32>
  %1 = linalg.tensor_reshape %0
         [affine_map<(d0, d1, d2) -> (d0, d1)>,
          affine_map<(d0, d1, d2) -> (d2)>] :
       tensor<3x4x4xf32> into tensor<12x4xf32>
  return %1 : tensor<12x4xf32>
}
// CHECK-LABEL: @fold_tensor_reshape
//   CHECK-NOT:   linalg.tensor_reshape

// -----

func @fold_tensor_reshape_dynamic(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32>
{
  %0 = linalg.tensor_reshape %arg0
         [affine_map<(d0, d1, d2) -> (d0, d1)>,
          affine_map<(d0, d1, d2) -> (d2)>] :
       tensor<?x?xf32> into tensor<?x4x?xf32>
  %1 = linalg.tensor_reshape %0
         [affine_map<(d0, d1, d2) -> (d0, d1)>,
          affine_map<(d0, d1, d2) -> (d2)>] :
       tensor<?x4x?xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: @fold_tensor_reshape_dynamic
//   CHECK-NOT:   linalg.tensor_reshape

// -----

func @fold_memref_reshape(%arg0 : memref<12x4xf32>) -> memref<12x4xf32>
{
  %0 = linalg.reshape %arg0
         [affine_map<(d0, d1, d2) -> (d0, d1)>,
          affine_map<(d0, d1, d2) -> (d2)>] :
       memref<12x4xf32> into memref<3x4x4xf32>
  %1 = linalg.reshape %0
         [affine_map<(d0, d1, d2) -> (d0, d1)>,
          affine_map<(d0, d1, d2) -> (d2)>] :
       memref<3x4x4xf32> into memref<12x4xf32>
  return %1 : memref<12x4xf32>
}
// CHECK-LABEL: @fold_memref_reshape
//   CHECK-NOT:   linalg.reshape

// -----

func @fold_memref_reshape_dynamic(%arg0 : memref<?x?xf32>) -> memref<?x?xf32>
{
  %0 = linalg.reshape %arg0
         [affine_map<(d0, d1, d2) -> (d0, d1)>,
          affine_map<(d0, d1, d2) -> (d2)>] :
       memref<?x?xf32> into memref<?x4x?xf32>
  %1 = linalg.reshape %0
         [affine_map<(d0, d1, d2) -> (d0, d1)>,
          affine_map<(d0, d1, d2) -> (d2)>] :
       memref<?x4x?xf32> into memref<?x?xf32>
  return %1 : memref<?x?xf32>
}
// CHECK-LABEL: @fold_memref_reshape_dynamic
//   CHECK-NOT:   linalg.reshape

// -----

#accesses = [
  affine_map<(i) -> (i)>
]

#trait = {
  indexing_maps = #accesses,
  iterator_types = ["parallel"]
}

func @dce_zero_memref(%arg0 : memref<0xf32>, %arg1: tensor<0xf32>) -> tensor<0xf32> {
  // memref<0x32> is expected to be dce'ed
  linalg.copy(%arg0, %arg0): memref<0xf32>, memref<0xf32>

  // tensor<0xf32> cannot be dce'ed
  %1 = linalg.generic #trait outs(%arg1 : tensor<0xf32>) {
  ^bb(%0: f32) :
    linalg.yield %0 : f32
  } -> tensor<0xf32>

  return %1: tensor<0xf32>
}
// CHECK-LABEL: @dce_zero_memref
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<0xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<0xf32>
//   CHECK-NOT:   linalg.copy
//  CHECK-NEXT:   return %[[ARG1]]

// -----

func @reshape_splat_constant_int32() -> tensor<2x4x2xi32>
{
  %c0 = constant dense<42> : tensor<2x8xi32>
  %0 = linalg.tensor_reshape %c0
         [affine_map<(d0, d1, d2) -> (d0)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>]
       : tensor<2x8xi32> into tensor<2x4x2xi32>
  return %0 : tensor<2x4x2xi32>
}
// CHECK-LABEL: @reshape_splat_constant_int32
//       CHECK:   %[[CST:.*]] = constant dense<{{.*}}> : tensor<2x4x2xi32>
//   CHECK-NOT:   linalg.tensor_reshape
//       CHECK:   return %[[CST]]

func @reshape_splat_constant_int16() -> tensor<2x4x2xi16>
{
  %c0 = constant dense<42> : tensor<2x8xi16>
  %0 = linalg.tensor_reshape %c0
         [affine_map<(d0, d1, d2) -> (d0)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>]
       : tensor<2x8xi16> into tensor<2x4x2xi16>
  return %0 : tensor<2x4x2xi16>
}
// CHECK-LABEL: @reshape_splat_constant_int16
//       CHECK:   %[[CST:.*]] = constant dense<{{.*}}> : tensor<2x4x2xi16>
//   CHECK-NOT:   linalg.tensor_reshape
//       CHECK:   return %[[CST]]

func @reshape_splat_constant_float32() -> tensor<2x4x2xf32>
{
  %c0 = constant dense<42.0> : tensor<2x8xf32>
  %0 = linalg.tensor_reshape %c0
         [affine_map<(d0, d1, d2) -> (d0)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>]
       : tensor<2x8xf32> into tensor<2x4x2xf32>
  return %0 : tensor<2x4x2xf32>
}
// CHECK-LABEL: @reshape_splat_constant_float32
//       CHECK:   %[[CST:.*]] = constant dense<{{.*}}> : tensor<2x4x2xf32>
//   CHECK-NOT:   linalg.tensor_reshape
//       CHECK:   return %[[CST]]

func @reshape_splat_constant_float64() -> tensor<2x4x2xf64>
{
  %c0 = constant dense<42.0> : tensor<2x8xf64>
  %0 = linalg.tensor_reshape %c0
         [affine_map<(d0, d1, d2) -> (d0)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>]
       : tensor<2x8xf64> into tensor<2x4x2xf64>
  return %0 : tensor<2x4x2xf64>
}
// CHECK-LABEL: @reshape_splat_constant_float64
//       CHECK:   %[[CST:.*]] = constant dense<{{.*}}> : tensor<2x4x2xf64>
//   CHECK-NOT:   linalg.tensor_reshape
//       CHECK:   return %[[CST]]

// -----

// CHECK-LABEL: func @tensor.cast(
func @tensor.cast(%a : tensor<3x4xf32>, %b : tensor<4x?xf32>, %c : tensor<3x?xf32>)
  -> tensor<3x?xf32>
{
  %ta = tensor.cast %a : tensor<3x4xf32> to tensor<?x?xf32>
  %tb = tensor.cast %b : tensor<4x?xf32> to tensor<?x?xf32>
  %tc = tensor.cast %c : tensor<3x?xf32> to tensor<?x?xf32>

  //      CHECK:  linalg.matmul ins({{.*}}tensor<3x4xf32>, tensor<4x?xf32>)
  // CHECK-SAME:    outs({{.*}}tensor<3x?xf32>) -> tensor<3x?xf32>
  %0 = linalg.matmul ins(%ta, %tb: tensor<?x?xf32>, tensor<?x?xf32>)
                    outs(%tc: tensor<?x?xf32>) -> tensor<?x?xf32>

  %1 = tensor.cast %0 : tensor<?x?xf32> to tensor<3x?xf32>

  return %1: tensor<3x?xf32>
}

// -----

// CHECK-LABEL: func @linalg_effects(
//  CHECK-SAME:     %[[A:[a-z0-9]*]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[B:[a-z0-9]*]]: memref<?x?xf32>
//  CHECK-SAME:     %[[C:[a-z0-9]*]]: tensor<?x?xf32>
func @linalg_effects(%a : tensor<?x?xf32>, %b : memref<?x?xf32>, %c : tensor<?x?xf32>) {
  // CHECK-NOT:   %{{.*}} = linalg.matmul
  %t = linalg.matmul ins(%a, %b : tensor<?x?xf32>, memref<?x?xf32>)
                    outs(%c : tensor<?x?xf32>) -> tensor<?x?xf32>

  // CHECK-NOT:   %{{.*}} = linalg.matmul
  linalg.matmul ins(%a, %c : tensor<?x?xf32>, tensor<?x?xf32>)
               outs(%b : memref<?x?xf32>)
  return
}
// -----

func @init_tensor_canonicalize() -> (tensor<4x5x?xf32>) {
  %c6 = constant 6 : index
  %0 = linalg.init_tensor [4, 5, %c6] : tensor<4x5x?xf32>
  return %0 : tensor<4x5x?xf32>
}
// CHECK: func @init_tensor_canonicalize
// CHECK:   %[[T0:.+]] = linalg.init_tensor [4, 5, 6] : tensor<4x5x6xf32>
// CHECK:   %[[T1:.+]] = tensor.cast %[[T0]] : tensor<4x5x6xf32> to tensor<4x5x?xf32>
// CHECK:   return %[[T1]]

// -----

func @init_tensor_static_dim() -> (index, index) {
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c6 = constant 6 : index
  %0 = linalg.init_tensor [4, 5, %c6] : tensor<4x5x?xf32>
  %1 = dim %0, %c2 : tensor<4x5x?xf32>
  %2 = dim %0, %c0 : tensor<4x5x?xf32>
  return %1, %2 : index, index
}
//      CHECK: func @init_tensor_static_dim
//  CHECK-DAG:   %[[C4:.+]] = constant 4 : index
//  CHECK-DAG:   %[[C6:.+]] = constant 6 : index
//      CHECK:   return %[[C6]], %[[C4]]

// -----

func @init_tensor_dynamic_dim(%arg0 : index) -> (index) {
  %c2 = constant 2 : index
  %0 = linalg.init_tensor [4, 5, %arg0] : tensor<4x5x?xf32>
  %1 = dim %0, %c2 : tensor<4x5x?xf32>
  return %1 : index
}
//      CHECK: func @init_tensor_dynamic_dim
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: index
//      CHECK:   return %[[ARG0]]

// -----

func @init_tensor_dynamic_dim2(%arg0 : index, %arg1 : index) -> (index, index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = linalg.init_tensor [%arg0, %arg1] : tensor<?x?xf32>
  %1 = dim %0, %c0 : tensor<?x?xf32>
  %2 = dim %0, %c1 : tensor<?x?xf32>
  return %1, %2 : index, index
}
//      CHECK: func @init_tensor_dynamic_dim2
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//      CHECK:   return %[[ARG0]], %[[ARG1]]

// -----

func @remove_dim_result_uses
  (%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
   %arg2 : tensor<?x?xf32>) -> (index) {
  %c0 = constant 0 : index
  %0 = linalg.generic
    {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                      affine_map<(d0, d1, d2) -> (d2, d1)>,
                      affine_map<(d0, d1, d2) -> (d0 + d1, d1)>],
     iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%arg2 : tensor<?x?xf32>) {
    ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32):
      %1 = mulf %arg3, %arg4 : f32
      %2 = addf %1, %arg5 : f32
      linalg.yield %2 : f32
    } -> tensor<?x?xf32>
  %3 = dim %0, %c0 : tensor<?x?xf32>
  return %3 : index
}
//       CHECK: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
//       CHECK: func @remove_dim_result_uses
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[T0:.+]] = dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[T1:.+]] = dim %[[ARG1]], %[[C1]]
//       CHECK:   %[[T2:.+]] = affine.apply #[[MAP]]()[%[[T0]], %[[T1]]]
//       CHECK:   return %[[T2]]

// -----

func @remove_dim_result_uses_outs
  (%arg0 : tensor<?xf32>, %arg1 : index) -> (index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %d0 = dim %arg0, %c0 : tensor<?xf32>
  %0 = linalg.init_tensor [%d0, %arg1] : tensor<?x?xf32>
  %1 = linalg.generic
    {indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                      affine_map<(d0, d1) -> (d0, d1)>],
     iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<?xf32>) outs(%0 : tensor<?x?xf32>) {
    ^bb0(%arg2: f32, %arg3: f32) :
      linalg.yield %arg2 : f32
    } -> tensor<?x?xf32>
  %2 = dim %1, %c1 : tensor<?x?xf32>
  return %2 : index
}
//      CHECK: func @remove_dim_result_uses_outs
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//      CHECK:   return %[[ARG1]]

// -----

func @remove_dim_result_uses_sequence
  (%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
   %arg2 : tensor<?x?xf32>) -> (index, index, index, index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = dim %0, %c0 : tensor<?x?xf32>
  %2 = dim %0, %c1 : tensor<?x?xf32>
  %3 = linalg.generic
    {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d0)>,
                      affine_map<(d0, d1, d2) -> (d0, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d2)>],
     iterator_types = ["parallel", "reduction", "parallel"]}
    ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%0 : tensor<?x?xf32>) {
    ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32):
      %4 = mulf %arg3, %arg4 : f32
      %5 = addf %4, %arg5 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?xf32>
  %6 = dim %3, %c0 : tensor<?x?xf32>
  %7 = dim %3, %c1 : tensor<?x?xf32>
  return %1, %2, %6, %7 : index, index, index, index
}
// CHECK-LABEL: func @remove_dim_result_uses_sequence
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[T0:.+]] = dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[T1:.+]] = dim %[[ARG1]], %[[C1]]
//   CHECK-DAG:   %[[T2:.+]] = dim %[[ARG0]], %[[C1]]
//   CHECK-DAG:   %[[T3:.+]] = dim %[[ARG1]], %[[C1]]
//       CHECK:   return %[[T0]], %[[T1]], %[[T2]], %[[T3]]

// -----

func @keep_result_dim_uses_sequence2
  (%arg0 : tensor<?xf32>, %arg1 : index) -> (index, index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %d0 = dim %arg0, %c0 : tensor<?xf32>
  %0 = linalg.init_tensor [%d0, %arg1] : tensor<?x?xf32>
  %1 = linalg.generic
    {indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                      affine_map<(d0, d1) -> (d0, d1)>],
     iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<?xf32>) outs(%0 : tensor<?x?xf32>) {
    ^bb0(%arg2: f32, %arg3 : f32):
      linalg.yield %arg2 : f32
    } -> tensor<?x?xf32>
  %2 = dim %1, %c0 : tensor<?x?xf32>
  %3 = dim %1, %c1 : tensor<?x?xf32>
  return %2, %3 : index, index
}
//       CHECK: func @keep_result_dim_uses_sequence2
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[T0:.+]] = dim %[[ARG0]], %[[C0]]
//       CHECK:   return %[[T0]], %[[ARG1]]

// -----

#map = affine_map<(d0) -> (d0)>

func @init_tensor_dim_of_linalg_result(%arg_0 : tensor<?xf32>,
    %arg_1: tensor<?xf32>) -> (index, index) {
  %0, %1 = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%arg_0 : tensor<?xf32>)
    outs(%arg_0, %arg_1 : tensor<?xf32>, tensor<?xf32>) {
  ^bb0(%in: f32, %out_0: f32, %out_1: f32):
    linalg.yield %in, %in : f32, f32
  } -> tensor<?xf32>, tensor<?xf32>

  %c0 = constant 0 : index
  %num_elem_0 = dim %0, %c0 : tensor<?xf32>

  %num_elem_1 = dim %1, %c0 : tensor<?xf32>
  return %num_elem_0, %num_elem_1 : index, index
}
//      CHECK: func @init_tensor_dim_of_linalg_result(
// CHECK-SAME:   %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<?xf32>
// CHECK-SAME:   %[[ARG_1:[a-zA-Z0-9_]+]]: tensor<?xf32>)
//      CHECK:   %[[R0:.+]] = dim %[[ARG_0]]
//      CHECK:   %[[R1:.+]] = dim %[[ARG_0]]
//      CHECK:   return %[[R0]], %[[R1]]

// -----

func @init_tensor_reshape_expansion(%arg0 : index) -> tensor<2x3x5x4x?x7xf32> {
  %0 = linalg.init_tensor [6, 5, %arg0] : tensor<6x5x?xf32>
  %1 = linalg.tensor_reshape %0
    [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1)>,
     affine_map<(d0, d1, d2, d3, d4, d5) -> (d2)>,
     affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>] :
     tensor<6x5x?xf32> into tensor<2x3x5x4x?x7xf32>
  return %1 : tensor<2x3x5x4x?x7xf32>
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0] -> (s0 floordiv 28)>
//      CHECK: func @init_tensor_reshape_expansion
// CHECK-SAME:   %[[ARG0:.+]]: index
//      CHECK:   %[[T0:.+]] = affine.apply #[[MAP]]()[%[[ARG0]]]
//      CHECK:   %[[T1:.+]] = linalg.init_tensor [2, 3, 5, 4, %[[T0]], 7]
//      CHECK:   return %[[T1]]

// -----

func @init_tensor_reshape_collapse(%arg0 : index) -> tensor<6x5x?xf32> {
  %0 = linalg.init_tensor [2, 3, 5, 4, %arg0, 7] : tensor<2x3x5x4x?x7xf32>
  %1 = linalg.tensor_reshape %0
    [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1)>,
     affine_map<(d0, d1, d2, d3, d4, d5) -> (d2)>,
     affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>] :
    tensor<2x3x5x4x?x7xf32> into tensor<6x5x?xf32>
  return %1 : tensor<6x5x?xf32>
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0] -> (s0 * 28)>
//      CHECK: func @init_tensor_reshape_collapse
// CHECK-SAME:   %[[ARG0:.+]]: index
//      CHECK:   %[[T0:.+]] = affine.apply #[[MAP]]()[%[[ARG0]]]
//      CHECK:   %[[T1:.+]] = linalg.init_tensor [6, 5, %[[T0]]]
//      CHECK:   return %[[T1]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func @remove_no_op(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?x?xf32>)
  -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %0 = dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = linalg.init_tensor [%0, %1, %2] : tensor<?x?x?xf32>
  %4, %5 = linalg.generic {
    indexing_maps = [#map, #map, #map, #map],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
    outs(%3, %3 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) {
  ^bb0(%arg2 : f32, %arg3 : f32, %arg4 : f32, %arg5 : f32):
    linalg.yield %arg3, %arg2 : f32, f32
  } -> tensor<?x?x?xf32>, tensor<?x?x?xf32>
  return %4, %5 : tensor<?x?x?xf32>, tensor<?x?x?xf32>
}
// CHECK-LABEL: func @remove_no_op
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//       CHECK:     return %[[ARG1]], %[[ARG0]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func @keep_not_noop(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %cst = constant 1.000000e+00 : f32
  %0 = dim %arg0, %c0 : tensor<?x?xf32>
  %1 = dim %arg0, %c1 : tensor<?x?xf32>
  %2 = linalg.init_tensor [%0, %1] : tensor<?x?xf32>
  br ^bb1(%cst : f32)

^bb1(%arg1 : f32):
  %3 = linalg.generic
    {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<?x?xf32>) outs(%2 : tensor<?x?xf32>) {
    ^bb0(%arg2: f32, %arg3 : f32):
      linalg.yield %arg1 : f32
    } -> tensor<?x?xf32>
  return %3 : tensor<?x?xf32>
}
// CHECK-LABEL: func @keep_not_noop
//       CHECK:   %[[RESULT:.+]] = linalg.generic
//       CHECK:   return %[[RESULT]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func @keep_not_noop(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>)
  -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %cst = constant 1.000000e+00 : f32
  %0 = dim %arg0, %c0 : tensor<?x?xf32>
  %1 = dim %arg0, %c1 : tensor<?x?xf32>
  %2 = linalg.init_tensor [%0, %1] : tensor<?x?xf32>
  br ^bb1(%cst : f32)

^bb1(%arg2 : f32):
  %3:2 = linalg.generic
    {indexing_maps = [#map, #map, #map, #map],
     iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%2, %2 : tensor<?x?xf32>, tensor<?x?xf32>) {
    ^bb0(%arg3: f32, %arg4 : f32, %arg5 : f32, %arg6 : f32):
      linalg.yield %arg2, %arg4 : f32, f32
    } -> tensor<?x?xf32>, tensor<?x?xf32>
  return %3#0, %3#1 : tensor<?x?xf32>, tensor<?x?xf32>
}
// CHECK-LABEL: func @keep_not_noop
//       CHECK:   %[[RESULT:.+]]:2 = linalg.generic
//       CHECK:   return %[[RESULT]]#0, %[[RESULT]]#1

// -----

func @fold_init_tensor_with_subtensor
  (%arg0 : index, %arg1 : index) -> tensor<5x?x20xf32>
{
  %0 = linalg.init_tensor[%arg0, 10, 40] : tensor<?x10x40xf32>
  %1 = subtensor %0[0, 0, 0] [5, %arg1, 20] [1, 1, 1]
    : tensor<?x10x40xf32> to tensor<5x?x20xf32>
  return %1 : tensor<5x?x20xf32>
}
//      CHECK: func @fold_init_tensor_with_subtensor
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//      CHECK:   %[[T0:.+]] = linalg.init_tensor [5, %[[ARG1]], 20]
//      CHECK:   return %[[T0]]

// -----

#accesses = [
  affine_map<(i, j) -> (i, j)>
]

#trait = {
  indexing_maps = #accesses,
  iterator_types = ["parallel", "parallel"]
}

// CHECK-LABEL: func @dead_linalg_tensor
//   CHECK-NOT:   linalg.fill
//   CHECK-NOT:   linalg.matmul
//   CHECK-NOT:   linalg.generic
//   CHECK-NOT:   linalg.pad_tensor
//       CHECK:   return
func @dead_linalg_tensor(%arg0 : tensor<7x7xi32>, %arg1 : tensor<7x7xf32>,
                         %arg2: tensor<?x?xf32>, %high : index) {
  %c0_i32 = constant 0 : i32
  %c0 = constant 0 : index
  %cst = constant 0.000000e+00 : f32
  %0 = linalg.fill(%arg0, %c0_i32) : tensor<7x7xi32>, i32 -> tensor<7x7xi32>
  %1 = linalg.matmul ins(%arg1, %arg1: tensor<7x7xf32>, tensor<7x7xf32>)
                     outs(%arg1: tensor<7x7xf32>) -> tensor<7x7xf32>
  %2 = linalg.generic #trait outs(%arg0 : tensor<7x7xi32>) {
  ^bb(%3: i32) :
    linalg.yield %3 : i32
  } -> tensor<7x7xi32>
  %3 = linalg.pad_tensor %arg2 low[%c0, %c0] high[%high, %high] {
        ^bb0(%arg9: index, %arg10: index):  // no predecessors
          linalg.yield %cst : f32
  } : tensor<?x?xf32> to tensor<2x4xf32>
  return
}

// -----

func @dim_reshape_expansion(%arg0 : tensor<6x5x?xf32>) -> (index, index, index)
{
  %c1 = constant 1 : index
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %0 = linalg.tensor_reshape %arg0
    [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1)>,
     affine_map<(d0, d1, d2, d3, d4, d5) -> (d2)>,
     affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>] :
     tensor<6x5x?xf32> into tensor<2x3x5x4x?x7xf32>
  %1 = dim %0, %c1 : tensor<2x3x5x4x?x7xf32>
  %2 = dim %0, %c3 : tensor<2x3x5x4x?x7xf32>
  %3 = dim %0, %c4 : tensor<2x3x5x4x?x7xf32>
  return %1, %2, %3 : index, index, index
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0] -> (s0 floordiv 28)>
//      CHECK: func @dim_reshape_expansion
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<6x5x?xf32>
//  CHECK-DAG:   %[[C2:.+]] = constant 2 : index
//  CHECK-DAG:   %[[C3:.+]] = constant 3 : index
//  CHECK-DAG:   %[[C4:.+]] = constant 4 : index
//      CHECK:   %[[D0:.+]] = dim %[[ARG0]], %[[C2]]
//      CHECK:   %[[D1:.+]] = affine.apply #[[MAP]]()[%[[D0]]]
//      CHECK:   return %[[C3]], %[[C4]], %[[D1]]

// -----

func @dim_reshape_collapse(%arg0 : tensor<2x3x5x4x?x7xf32>) -> (index, index)
{
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %0 = linalg.tensor_reshape %arg0
    [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1)>,
     affine_map<(d0, d1, d2, d3, d4, d5) -> (d2)>,
     affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>] :
     tensor<2x3x5x4x?x7xf32> into tensor<6x5x?xf32>
  %1 = dim %0, %c1 : tensor<6x5x?xf32>
  %2 = dim %0, %c2 : tensor<6x5x?xf32>
  return %1, %2 : index, index
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0] -> (s0 * 28)>
//      CHECK: func @dim_reshape_collapse
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<2x3x5x4x?x7xf32>
//  CHECK-DAG:   %[[C4:.+]] = constant 4 : index
//  CHECK-DAG:   %[[C5:.+]] = constant 5 : index
//      CHECK:   %[[D0:.+]] = dim %[[ARG0]], %[[C4]]
//      CHECK:   %[[D1:.+]] = affine.apply #[[MAP]]()[%[[D0]]]
//      CHECK:   return %[[C5]], %[[D1]]
