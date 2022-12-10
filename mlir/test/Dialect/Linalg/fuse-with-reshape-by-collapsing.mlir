// RUN: mlir-opt %s -test-linalg-elementwise-fusion-patterns=fuse-with-reshape-by-collapsing -split-input-file | FileCheck %s
// RUN: mlir-opt %s -test-linalg-elementwise-fusion-patterns=fuse-with-reshape-by-collapsing-control -split-input-file | FileCheck %s --check-prefix=CONTROL

// Static problem sizes. Checks all aspects of fusion by collapsing. Rest of the
// tests only check a subset of conditions.
#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>
func.func @fuse_by_collapsing(%arg0 : tensor<2x12x5x336x9xi32>,
    %arg1 : tensor<2x3x4xi32>, %arg2 : tensor<5x6x7x8xi32>) -> tensor<2x3x4x5x6x7x8x9xi32> {
  %expand = tensor.expand_shape %arg0 [[0], [1, 2], [3], [4, 5, 6], [7]]
      : tensor<2x12x5x336x9xi32> into tensor<2x3x4x5x6x7x8x9xi32>
  %init = tensor.empty() : tensor<2x3x4x5x6x7x8x9xi32>
  %generic = linalg.generic {
    indexing_maps = [#map0, #map1, #map2, #map3],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]}
    ins(%expand, %arg1, %arg2 : tensor<2x3x4x5x6x7x8x9xi32>, tensor<2x3x4xi32>, tensor<5x6x7x8xi32>)
    outs(%init : tensor<2x3x4x5x6x7x8x9xi32>) {
      ^bb0(%b0 : i32, %b1 : i32, %b2 : i32, %b3 : i32):
        %t0 = arith.addi %b0, %b1 : i32
        %t1 = arith.addi %t0, %b2 : i32
        linalg.yield %t1 : i32
    } -> tensor<2x3x4x5x6x7x8x9xi32>
  return %generic : tensor<2x3x4x5x6x7x8x9xi32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
//      CHECK: func @fuse_by_collapsing(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<2x12x5x336x9xi32>
// CHECK-SAME:   %[[ARG1:.+]]: tensor<2x3x4xi32>
// CHECK-SAME:   %[[ARG2:.+]]: tensor<5x6x7x8xi32>
//  CHECK-DAG:   %[[INIT:.+]] = tensor.empty()
//  CHECK-DAG:   %[[ARG1_RESHAPE:.+]] = tensor.collapse_shape %[[ARG1]] {{\[}}[0], [1, 2]{{\]}}
//  CHECK-DAG:   %[[ARG2_RESHAPE:.+]] = tensor.collapse_shape %[[ARG2]] {{\[}}[0], [1, 2, 3]{{\]}}
//  CHECK-DAG:   %[[INIT_RESHAPE:.+]] = tensor.collapse_shape %[[INIT]] {{\[}}[0], [1, 2], [3], [4, 5, 6], [7]{{\]}}
//      CHECK:   %[[COLLAPSED_OP:.+]] = linalg.generic
// CHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]], #[[MAP0]]]
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1_RESHAPE]], %[[ARG2_RESHAPE]] :
// CHECK-SAME:       outs(%[[INIT_RESHAPE]] :
//      CHECK:   %[[RESULT_RESHAPE:.+]] = tensor.expand_shape %[[COLLAPSED_OP]] {{\[}}[0], [1, 2], [3], [4, 5, 6], [7]{{\]}}
//      CHECK:   return %[[RESULT_RESHAPE]]

//      CONTROL: func @fuse_by_collapsing(
// CONTROL-SAME:   %[[ARG0:.+]]: tensor<2x12x5x336x9xi32>
// CONTROL-SAME:   %[[ARG1:.+]]: tensor<2x3x4xi32>
// CONTROL-SAME:   %[[ARG2:.+]]: tensor<5x6x7x8xi32>
//      CONTROL:   %[[EXPAND:.+]] = tensor.expand_shape %[[ARG0]]
//      CONTROL:   %[[GENERIC:.+]] = linalg.generic
// CONTROL-SAME:       ins(%[[EXPAND]],
//      CONTROL:   return %[[GENERIC]]

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>
func.func @fuse_by_collapsing_indexing_op(%arg0 : tensor<2x12x5x336x9xi32>,
    %arg1 : tensor<2x3x4xi32>, %arg2 : tensor<5x6x7x8xi32>) -> tensor<2x3x4x5x6x7x8x9xi32> {
  %expand = tensor.expand_shape %arg0 [[0], [1, 2], [3], [4, 5, 6], [7]]
      : tensor<2x12x5x336x9xi32> into tensor<2x3x4x5x6x7x8x9xi32>
  %init = tensor.empty() : tensor<2x3x4x5x6x7x8x9xi32>
  %generic = linalg.generic {
    indexing_maps = [#map0, #map1, #map2, #map3],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]}
    ins(%expand, %arg1, %arg2 : tensor<2x3x4x5x6x7x8x9xi32>, tensor<2x3x4xi32>, tensor<5x6x7x8xi32>)
    outs(%init : tensor<2x3x4x5x6x7x8x9xi32>) {
      ^bb0(%b0 : i32, %b1 : i32, %b2 : i32, %b3 : i32):
        %iv0 = linalg.index 0: index
        %iv1 = linalg.index 1: index
        %t0 = arith.addi %iv0, %iv1 : index
        %iv2 = linalg.index 2 : index
        %t1 = arith.addi %t0, %iv2 : index
        %iv3 = linalg.index 3 : index
        %t2 = arith.addi %t1, %iv3 : index
        %iv4 = linalg.index 4 : index
        %t3 = arith.addi %t2, %iv4 : index
        %iv5 = linalg.index 5 : index
        %t4 = arith.addi %t3, %iv5 : index
        %iv6 = linalg.index 6 : index
        %t5 = arith.addi %t4, %iv6 : index
        %iv7 = linalg.index 7 : index
        %t6 = arith.addi %t5, %iv7 : index
        %yield = arith.index_cast %t6 : index to i32
        linalg.yield %yield : i32
    } -> tensor<2x3x4x5x6x7x8x9xi32>
  return %generic : tensor<2x3x4x5x6x7x8x9xi32>
}
// CHECK-LABEL: func @fuse_by_collapsing_indexing_op(
//   CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//   CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
//   CHECK-DAG:   %[[C7:.+]] = arith.constant 7 : index
//       CHECK:     %[[IV0:.+]] = linalg.index 0
//       CHECK:     %[[IV1:.+]] = linalg.index 1
//       CHECK:     %[[REM_IV1:.+]] = arith.remui %[[IV1]], %[[C4]]
//       CHECK:     %[[DIV_IV1:.+]] = arith.divui %[[IV1]], %[[C4]]
//       CHECK:     %[[IV2:.+]] = linalg.index 2
//       CHECK:     %[[IV3:.+]] = linalg.index 3
//       CHECK:     %[[REM1_IV3:.+]] = arith.remui %[[IV3]], %[[C8]]
//       CHECK:     %[[DIV1_IV3:.+]] = arith.divui %[[IV3]], %[[C8]]
//       CHECK:     %[[REM2_IV3:.+]] = arith.remui %[[DIV1_IV3]], %[[C7]]
//       CHECK:     %[[DIV2_IV3:.+]] = arith.divui %[[DIV1_IV3]], %[[C7]]
//       CHECK:     %[[IV4:.+]] = linalg.index 4
//       CHECK:     %[[T0:.+]] = arith.addi %[[IV0]], %[[DIV_IV1]]
//       CHECK:     %[[T1:.+]] = arith.addi %[[T0]], %[[REM_IV1]]
//       CHECK:     %[[T2:.+]] = arith.addi %[[T1]], %[[IV2]]
//       CHECK:     %[[T3:.+]] = arith.addi %[[T2]], %[[DIV2_IV3]]
//       CHECK:     %[[T4:.+]] = arith.addi %[[T3]], %[[REM2_IV3]]
//       CHECK:     %[[T5:.+]] = arith.addi %[[T4]], %[[REM1_IV3]]
//       CHECK:     %[[T6:.+]] = arith.addi %[[T5]], %[[IV4]]
//       CHECK:     %[[YIELD:.+]] = arith.index_cast %[[T6]]
//       CHECK:     linalg.yield %[[YIELD]]

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d7, d5, d6, d0, d1, d2, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d6, d0)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>
func.func @fuse_by_collapsing_change_reshape_order(%arg0 : tensor<9x56x2x60x6xi32>,
    %arg1 : tensor<7x8x2xi32>, %arg2 : tensor<6x3x4x5xi32>) -> tensor<2x3x4x5x6x7x8x9xi32> {
  %expand = tensor.expand_shape %arg0 [[0], [1, 2], [3], [4, 5, 6], [7]]
      : tensor<9x56x2x60x6xi32> into tensor<9x7x8x2x3x4x5x6xi32>
  %init = tensor.empty() : tensor<2x3x4x5x6x7x8x9xi32>
  %generic = linalg.generic {
    indexing_maps = [#map0, #map1, #map2, #map3],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]}
    ins(%expand, %arg1, %arg2 : tensor<9x7x8x2x3x4x5x6xi32>, tensor<7x8x2xi32>, tensor<6x3x4x5xi32>)
    outs(%init : tensor<2x3x4x5x6x7x8x9xi32>) {
      ^bb0(%b0 : i32, %b1 : i32, %b2 : i32, %b3 : i32):
        %t0 = arith.addi %b0, %b1 : i32
        %t1 = arith.addi %t0, %b2 : i32
        linalg.yield %t1 : i32
    } -> tensor<2x3x4x5x6x7x8x9xi32>
  return %generic : tensor<2x3x4x5x6x7x8x9xi32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d4, d3, d0, d1, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d3, d0)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d2, d1)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
//      CHECK: func @fuse_by_collapsing_change_reshape_order(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<9x56x2x60x6xi32>
// CHECK-SAME:   %[[ARG1:.+]]: tensor<7x8x2xi32>
// CHECK-SAME:   %[[ARG2:.+]]: tensor<6x3x4x5xi32>
//  CHECK-DAG:   %[[INIT:.+]] = tensor.empty()
//  CHECK-DAG:   %[[ARG1_RESHAPE:.+]] = tensor.collapse_shape %[[ARG1]] {{\[}}[0, 1], [2]{{\]}}
//  CHECK-DAG:   %[[ARG2_RESHAPE:.+]] = tensor.collapse_shape %[[ARG2]] {{\[}}[0], [1, 2, 3]{{\]}}
//  CHECK-DAG:   %[[INIT_RESHAPE:.+]] = tensor.collapse_shape %[[INIT]] {{\[}}[0], [1, 2, 3], [4], [5, 6], [7]{{\]}}
//      CHECK:   %[[COLLAPSED_OP:.+]] = linalg.generic
// CHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]], #[[MAP3]]]
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1_RESHAPE]], %[[ARG2_RESHAPE]] :
// CHECK-SAME:       outs(%[[INIT_RESHAPE]] :
//      CHECK:   %[[RESULT_RESHAPE:.+]] = tensor.expand_shape %[[COLLAPSED_OP]] {{\[}}[0], [1, 2, 3], [4], [5, 6], [7]{{\]}}
//      CHECK:   return %[[RESULT_RESHAPE]]

// -----

// Dynamic case. Only checks things not covered by `fuse_by_collapsing` test above.
#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d7, d5, d6, d0, d1, d2, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d6, d0)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>
func.func @fuse_by_collapsing_dynamic(%arg0 : tensor<?x?x?x?x?xi32>,
    %arg1 : tensor<?x?x?xi32>, %arg2 : tensor<?x?x?x?xi32>) -> tensor<?x3x?x5x?x7x?x?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %expand = tensor.expand_shape %arg0 [[0], [1, 2], [3], [4, 5, 6], [7]]
      : tensor<?x?x?x?x?xi32> into tensor<?x7x?x?x3x?x5x?xi32>
  %d0 = tensor.dim %arg1, %c2 : tensor<?x?x?xi32>
  %d2 = tensor.dim %arg2, %c2 : tensor<?x?x?x?xi32>
  %d4 = tensor.dim %arg2, %c0 : tensor<?x?x?x?xi32>
  %d6 = tensor.dim %arg1, %c1 : tensor<?x?x?xi32>
  %d7 = tensor.dim %arg0, %c0 : tensor<?x?x?x?x?xi32>
  %init = tensor.empty(%d0, %d2, %d4, %d6, %d7) : tensor<?x3x?x5x?x7x?x?xi32>
  %generic = linalg.generic {
    indexing_maps = [#map0, #map1, #map2, #map3],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]}
    ins(%expand, %arg1, %arg2 : tensor<?x7x?x?x3x?x5x?xi32>, tensor<?x?x?xi32>, tensor<?x?x?x?xi32>)
    outs(%init : tensor<?x3x?x5x?x7x?x?xi32>) {
      ^bb0(%b0 : i32, %b1 : i32, %b2 : i32, %b3 : i32):
        %iv0 = linalg.index 0: index
        %iv1 = linalg.index 1: index
        %t0 = arith.addi %iv0, %iv1 : index
        %iv2 = linalg.index 2 : index
        %t1 = arith.addi %t0, %iv2 : index
        %iv3 = linalg.index 3 : index
        %t2 = arith.addi %t1, %iv3 : index
        %iv4 = linalg.index 4 : index
        %t3 = arith.addi %t2, %iv4 : index
        %iv5 = linalg.index 5 : index
        %t4 = arith.addi %t3, %iv5 : index
        %iv6 = linalg.index 6 : index
        %t5 = arith.addi %t4, %iv6 : index
        %iv7 = linalg.index 7 : index
        %t6 = arith.addi %t5, %iv7 : index
        %yield = arith.index_cast %t6 : index to i32
        linalg.yield %yield : i32
    } -> tensor<?x3x?x5x?x7x?x?xi32>
  return %generic : tensor<?x3x?x5x?x7x?x?xi32>
}
//      CHECK: func @fuse_by_collapsing_dynamic(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?x?x?x?xi32>
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//  CHECK-DAG:   %[[C5:.+]] = arith.constant 5 : index
//      CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[ARG0]]
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[EXPAND]], %[[C2]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[EXPAND]], %[[C5]]
//      CHECK:   linalg.generic
//      CHECK:     %[[IV0:.+]] = linalg.index 1
//      CHECK:     %[[REM1_IV0:.+]] = arith.remui %[[IV0]], %[[C5]]
//      CHECK:     %[[DIV1_IV0:.+]] = arith.divui %[[IV0]], %[[C5]]
//      CHECK:     %[[REM2_IV0:.+]] = arith.remui %[[DIV1_IV0]], %[[D1]]
//      CHECK:     %[[DIV2_IV0:.+]] = arith.divui %[[DIV1_IV0]], %[[D1]]
//      CHECK:     %[[IV1:.+]] = linalg.index 3
//      CHECK:     %[[REM1_IV1:.+]] = arith.remui %[[IV1]], %[[D0]]
//      CHECK:     %[[DIV1_IV1:.+]] = arith.divui %[[IV1]], %[[D0]]

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3)>
func.func @fuse_reductions(%arg0 : tensor<2x?x5xf32>, %arg1 : tensor<2x5xf32>) -> tensor<2x5xf32> {
  %0 = tensor.expand_shape %arg0 [[0], [1, 2], [3]] : tensor<2x?x5xf32> into tensor<2x6x?x5xf32>
  %1 = linalg.generic {
      indexing_maps = [#map0, #map1],
      iterator_types = ["parallel", "reduction", "reduction", "parallel"]}
      ins(%0 : tensor<2x6x?x5xf32>) outs(%arg1 : tensor<2x5xf32>) {
        ^bb0(%b0 : f32, %b1 : f32):
          %2 = arith.addf %b0, %b1 : f32
          linalg.yield %2 : f32
      } -> tensor<2x5xf32>
  return %1 : tensor<2x5xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//      CHECK: func @fuse_reductions(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<2x?x5xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<2x5xf32>) -> tensor<2x5xf32>
//      CHECK:   %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME:       iterator_types = ["parallel", "reduction", "parallel"]
// CHECK-SAME:       ins(%[[ARG0]] : tensor<2x?x5xf32>)
// CHECK-SAME:       outs(%[[ARG1]] : tensor<2x5xf32>)

// -----

// Test no fusion because the folded dimensions are not all preserved.
#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
func.func @no_fuse_unpreserved_folding(%arg0 : tensor<2x12x5xf32>, %arg1 : tensor<2x3xf32>) -> tensor<2x3x4x5xf32> {
  %0 = tensor.expand_shape %arg0 [[0], [1, 2], [3]] : tensor<2x12x5xf32> into tensor<2x3x4x5xf32>
  %init = tensor.empty(): tensor<2x3x4x5xf32>
  %1 = linalg.generic {
      indexing_maps = [#map0, #map1, #map0],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%0, %arg1 : tensor<2x3x4x5xf32>, tensor<2x3xf32>) outs(%init : tensor<2x3x4x5xf32>) {
        ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
          %2 = arith.addf %b0, %b1 : f32
          linalg.yield %2 : f32
      } -> tensor<2x3x4x5xf32>
  return %1 : tensor<2x3x4x5xf32>
}
//      CHECK: func @no_fuse_unpreserved_folding
// CHECK-SAME:     %[[ARG0:.+]]: tensor<2x12x5xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<2x3xf32>
//      CHECK:   %[[RESHAPE:.+]] = tensor.expand_shape %[[ARG0]]
//      CHECK:   %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:       ins(%[[RESHAPE]], %[[ARG1]] :
//      CHECK:   return %[[GENERIC]]

// -----

// Test no fusion because the folded dimensions are not all preserved.
#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
func.func @no_fuse_unpreserved_folding_transpose(%arg0 : tensor<2x12x5xf32>, %arg1 : tensor<2xf32>) -> tensor<2x4x3x5xf32> {
  %0 = tensor.expand_shape %arg0 [[0], [1, 2], [3]] : tensor<2x12x5xf32> into tensor<2x3x4x5xf32>
  %init = tensor.empty() : tensor<2x4x3x5xf32>
  %1 = linalg.generic {
      indexing_maps = [#map0, #map1, #map2],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%0, %arg1 : tensor<2x3x4x5xf32>, tensor<2xf32>) outs(%init : tensor<2x4x3x5xf32>) {
        ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
          %2 = arith.addf %b0, %b1 : f32
          linalg.yield %2 : f32
      } -> tensor<2x4x3x5xf32>
  return %1 : tensor<2x4x3x5xf32>
}
//      CHECK: func @no_fuse_unpreserved_folding_transpose
// CHECK-SAME:     %[[ARG0:.+]]: tensor<2x12x5xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<2xf32>
//      CHECK:   %[[RESHAPE:.+]] = tensor.expand_shape %[[ARG0]]
//      CHECK:   %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:       ins(%[[RESHAPE]], %[[ARG1]] :
//      CHECK:   return %[[GENERIC]]

// -----

// Test no fusion because the iterator types of folded dims are not preserved.
#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d3)>
func.func @no_fuse_mismatched_iterator_types(%arg0 : tensor<2x12x5xf32>, %arg1 : tensor<2x3xf32>) -> tensor<2x5xf32> {
  %0 = tensor.expand_shape %arg0 [[0], [1, 2], [3]] : tensor<2x12x5xf32> into tensor<2x3x4x5xf32>
  %init = tensor.empty() : tensor<2x5xf32>
  %1 = linalg.generic {
      indexing_maps = [#map0, #map1, #map2],
      iterator_types = ["parallel", "reduction", "parallel", "parallel"]}
      ins(%0, %arg1 : tensor<2x3x4x5xf32>, tensor<2x3xf32>) outs(%init : tensor<2x5xf32>) {
        ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
          %2 = arith.addf %b0, %b1 : f32
          linalg.yield %2 : f32
      } -> tensor<2x5xf32>
  return %1 : tensor<2x5xf32>
}
//      CHECK: func @no_fuse_mismatched_iterator_types
// CHECK-SAME:     %[[ARG0:.+]]: tensor<2x12x5xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<2x3xf32>
//      CHECK:   %[[RESHAPE:.+]] = tensor.expand_shape %[[ARG0]]
//      CHECK:   %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:       ins(%[[RESHAPE]], %[[ARG1]] :
//      CHECK:   return %[[GENERIC]]

// -----

// Test control of fusion using control function
// Test no fusion because the folded dimensions are not all preserved.
#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @control_fusion(%arg0 : tensor<6xf32>, %arg1 : tensor<20xf32>) -> tensor<2x3x4x5xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1]] : tensor<6xf32> into tensor<2x3xf32>
  %1 = tensor.expand_shape %arg1 [[0, 1]] : tensor<20xf32> into tensor<4x5xf32>
    %init = tensor.empty() : tensor<2x3x4x5xf32>
  %2 = linalg.generic {
      indexing_maps = [#map0, #map1, #map2],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%0, %1 : tensor<2x3xf32>, tensor<4x5xf32>) outs(%init : tensor<2x3x4x5xf32>) {
        ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
          %3 = arith.addf %b0, %b1 : f32
          linalg.yield %3 : f32
      } -> tensor<2x3x4x5xf32>
  return %2 : tensor<2x3x4x5xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//      CHECK: func @control_fusion(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<6xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<20xf32>
//      CHECK:   %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:       iterator_types = ["parallel", "parallel"]
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%{{.+}}: tensor<6x20xf32>)
//      CHECK:   %[[RESHAPE1:.+]] = tensor.expand_shape %[[GENERIC]] {{\[}}[0], [1, 2]{{\]}}
//      CHECK:   %[[RESHAPE2:.+]] = tensor.expand_shape %[[RESHAPE1]] {{\[}}[0, 1], [2], [3]{{\]}}
//      CHECK:   return %[[RESHAPE2]]

//  CONTROL-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//  CONTROL-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2)>
//  CONTROL-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//      CONTROL: func @control_fusion(
// CONTROL-SAME:     %[[ARG0:.+]]: tensor<6xf32>
// CONTROL-SAME:     %[[ARG1:.+]]: tensor<20xf32>
//      CONTROL:     %[[EXPAND:.+]] = tensor.expand_shape %[[ARG0]]
//      CONTROL:     %[[INIT:.+]] = tensor.empty()
//      CONTROL:     %[[INIT_RESHAPE:.+]] = tensor.collapse_shape %[[INIT]] {{\[}}[0], [1], [2, 3]{{\]}}
//      CONTROL:     %[[GENERIC:.+]] = linalg.generic
// CONTROL-SAME:         ins(%[[EXPAND]], %[[ARG1]] :
// CONTROL-SAME:         outs(%[[INIT_RESHAPE]] :
//      CONTROL:     %[[RESULT:.+]] = tensor.expand_shape %[[GENERIC]] {{\[}}[0], [1], [2, 3]{{\]}}

// -----

// Corner case that isnt handled currently.
#map = affine_map<(d0) -> (d0)>
func.func @zero_D_test(%arg0: tensor<f32>) -> tensor<1xf32> {
  %0 = tensor.expand_shape %arg0 [] : tensor<f32> into tensor<1xf32>
  %init = tensor.empty() : tensor<1xf32>
  %1 = linalg.generic {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel"]}
      ins(%0: tensor<1xf32>) outs(%init : tensor<1xf32>) {
        ^bb0(%b0 : f32, %b1 : f32):
          linalg.yield %b0: f32
      } -> tensor<1xf32>
  return %1 : tensor<1xf32>
}
//      CHECK: func @zero_D_test
// CHECK-SAME:     %[[ARG0:.+]]: tensor<f32>
//      CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[ARG0]]
//      CHECK:   %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:       ins(%[[EXPAND]] :
//      CHECK:   return %[[GENERIC]]

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @fuse_only_one_reassociation(%arg0 : tensor<?x?xf32>, %arg1 : tensor<4x?x?x8xf32>) -> tensor<4x?x?x8xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1], [2, 3]] : tensor<?x?xf32> into tensor<?x4x?x8xf32>
  %1 = linalg.generic {
      indexing_maps = [#map0, #map1, #map1],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%0, %arg1 : tensor<?x4x?x8xf32>, tensor<4x?x?x8xf32>)
      outs(%arg1 : tensor<4x?x?x8xf32>) {
    ^bb0(%b0: f32, %b1 : f32, %b2 : f32):
      %2 = arith.addf %b0, %b1 : f32
      linalg.yield %2 : f32
    } -> tensor<4x?x?x8xf32>
  return %1 : tensor<4x?x?x8xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//      CHECK: func @fuse_only_one_reassociation(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<4x?x?x8xf32>
//  CHECK-DAG:   %[[EXPAND_ARG0:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1], [2, 3]{{\]}}
//  CHECK-DAG:   %[[COLLAPSE_ARG0:.+]] = tensor.collapse_shape %[[EXPAND_ARG0]] {{\[}}[0], [1], [2, 3]{{\]}}
//  CHECK-DAG:   %[[COLLAPSE_ARG1_0:.+]] = tensor.collapse_shape %[[ARG1]] {{\[}}[0], [1], [2, 3]{{\]}}
//  CHECK-DAG:   %[[COLLAPSE_ARG1_1:.+]] = tensor.collapse_shape %[[ARG1]] {{\[}}[0], [1], [2, 3]{{\]}}
//      CHECK:   %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP1]]]
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME:       ins(%[[COLLAPSE_ARG0]], %[[COLLAPSE_ARG1_0]] :
// CHECK-SAME:       outs(%[[COLLAPSE_ARG1_1]] :
//      CHECK:   %[[EXPAND_GENERIC:.+]] = tensor.expand_shape %[[GENERIC]] {{\[}}[0], [1], [2, 3]{{\]}}
//      CHECK:   return %[[EXPAND_GENERIC]]

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d1, d0, d2)>
func.func @fold_non_consecutive_dims(%arg0 : tensor<?x?xi32>) -> tensor<?x8x?x4xi32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.expand_shape %arg0 [[0, 1], [2, 3]] : tensor<?x?xi32> into tensor<?x4x?x8xi32>
  %d0 = tensor.dim %0, %c0 : tensor<?x4x?x8xi32>
  %d1 = tensor.dim %0, %c2 : tensor<?x4x?x8xi32>
  %init = tensor.empty(%d1, %d0) : tensor<?x8x?x4xi32>
  %1 = linalg.generic {
      indexing_maps = [#map0, #map1],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%0 : tensor<?x4x?x8xi32>) outs(%init : tensor<?x8x?x4xi32>) {
    ^bb0(%b0 : i32, %b1 : i32):
      %2 = linalg.index 0 : index
      %3 = linalg.index 1 : index
      %4 = linalg.index 2 : index
      %5 = linalg.index 3 : index
      %6 = arith.addi %2, %3 : index
      %7 = arith.addi %6, %4 : index
      %8 = arith.addi %7, %5 : index
      %9 = arith.index_cast %8 : index to i32
      linalg.yield %9: i32
    } -> tensor<?x8x?x4xi32>
  return %1 : tensor<?x8x?x4xi32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d1, d0)>
//      CHECK: func @fold_non_consecutive_dims(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xi32>)
//  CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//  CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
//      CHECK:   %[[INIT:.+]] = tensor.empty
//      CHECK:   %[[COLLAPSE_INIT:.+]] = tensor.collapse_shape %[[INIT]] {{\[}}[0, 1], [2, 3]{{\]}}
//      CHECK:   %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME:       iterator_types = ["parallel", "parallel"]
// CHECK-SAME:       ins(%[[ARG0]] :
// CHECK-SAME:       outs(%[[COLLAPSE_INIT]] :
// CHECK-NEXT:   ^bb{{[0-9]}}
//      CHECK:       %[[ID0:.+]] = linalg.index 0
//  CHECK-DAG:       %[[T0:.+]] = arith.remui %[[ID0]], %[[C4]]
//  CHECK-DAG:       %[[T1:.+]] = arith.divui %[[ID0]], %[[C4]]
//      CHECK:       %[[ID1:.+]] = linalg.index 1
//  CHECK-DAG:       %[[T2:.+]] = arith.remui %[[ID1]], %[[C8]]
//  CHECK-DAG:       %[[T3:.+]] = arith.divui %[[ID1]], %[[C8]]
//  CHECK-DAG:       %[[T4:.+]] = arith.addi %[[T1]], %[[T2]]
//  CHECK-DAG:       %[[T5:.+]] = arith.addi %[[T4]], %[[T0]]
//  CHECK-DAG:       %[[T6:.+]] = arith.addi %[[T5]], %[[T3]]
//  CHECK-DAG:       %[[T7:.+]] = arith.index_cast %[[T6]]
//      CHECK:       linalg.yield %[[T7]]
//      CHECK:   %[[EXPAND_GENERIC:.+]] = tensor.expand_shape %[[GENERIC]] {{\[}}[0, 1], [2, 3]{{\]}}
//      CHECK:   return %[[EXPAND_GENERIC]]

// -----

// None of the folded iteration space dims are contiguous reduction dimensions.
// So no change in the code.
#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> ()>
func.func @no_fold_non_consecutive_reduction_dims(%arg0 : tensor<?x?xi32>) -> tensor<i32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.expand_shape %arg0 [[0, 1], [2, 3]] : tensor<?x?xi32> into tensor<?x4x?x8xi32>
  %init = tensor.empty() : tensor<i32>
  %1 = linalg.generic {
      indexing_maps = [#map0, #map1],
      iterator_types = ["reduction", "reduction", "reduction", "reduction"]}
      ins(%0 : tensor<?x4x?x8xi32>) outs(%init : tensor<i32>) {
    ^bb0(%b0 : i32, %b1 : i32):
      %2 = linalg.index 0 : index
      %3 = linalg.index 1 : index
      %4 = linalg.index 2 : index
      %5 = linalg.index 3 : index
      %6 = arith.addi %2, %3 : index
      %7 = arith.addi %6, %4 : index
      %8 = arith.addi %7, %5 : index
      %9 = arith.index_cast %8 : index to i32
      linalg.yield %9: i32
    } -> tensor<i32>
  return %1 : tensor<i32>
}
//      CHECK: func @no_fold_non_consecutive_reduction_dims(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?xi32>)
//      CHECK:   %[[EXPAND_ARG0:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1], [2, 3]{{\]}}
//      CHECK:   %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:       ins(%[[EXPAND_ARG0]] :
//      CHECK:   return %[[GENERIC]]
