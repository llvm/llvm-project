// RUN: mlir-opt %s -test-linalg-elementwise-fusion-patterns=fuse-with-reshape-by-collapsing -split-input-file | FileCheck %s
// RUN: mlir-opt %s -test-linalg-elementwise-fusion-patterns=fuse-with-reshape-by-collapsing-control -split-input-file | FileCheck %s --check-prefix=CONTROL

// Static problem sizes. Checks all aspects of fusion by collapsing. Rest of the
// tests only check a subset of conditions.
#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d2, d0, d7, d3, d4, d5, d6)>
func.func @fuse_by_collapsing(%arg0 : tensor<2x12x5x336x9xi32>,
    %arg1 : tensor<2x3x4xi32>, %arg2 : tensor<5x6x7x8xi32>) -> (tensor<2x3x4x5x6x7x8x9xi32>, tensor<3x4x2x9x5x6x7x8xi32>) {
  %expand = tensor.expand_shape %arg0 [[0], [1, 2], [3], [4, 5, 6], [7]] output_shape [2, 3, 4, 5, 6, 7, 8, 9] : tensor<2x12x5x336x9xi32> into tensor<2x3x4x5x6x7x8x9xi32>
  %init_0 = tensor.empty() : tensor<2x3x4x5x6x7x8x9xi32>
  %init_1 = tensor.empty() : tensor<3x4x2x9x5x6x7x8xi32>
  %generic:2 = linalg.generic {
    indexing_maps = [#map0, #map1, #map2, #map3, #map4],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]}
    ins(%expand, %arg1, %arg2 : tensor<2x3x4x5x6x7x8x9xi32>, tensor<2x3x4xi32>, tensor<5x6x7x8xi32>)
    outs(%init_0, %init_1 : tensor<2x3x4x5x6x7x8x9xi32>, tensor<3x4x2x9x5x6x7x8xi32>) {
      ^bb0(%b0 : i32, %b1 : i32, %b2 : i32, %b3 : i32, %b4 : i32):
        %t0 = arith.addi %b0, %b1 : i32
        %t1 = arith.addi %t0, %b2 : i32
        linalg.yield %t1, %t1 : i32, i32
    } -> (tensor<2x3x4x5x6x7x8x9xi32>, tensor<3x4x2x9x5x6x7x8xi32>)
  return %generic#0, %generic#1 : tensor<2x3x4x5x6x7x8x9xi32>, tensor<3x4x2x9x5x6x7x8xi32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d0, d4, d2, d3)>
//      CHECK: func @fuse_by_collapsing(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<2x12x5x336x9xi32>
// CHECK-SAME:   %[[ARG1:.+]]: tensor<2x3x4xi32>
// CHECK-SAME:   %[[ARG2:.+]]: tensor<5x6x7x8xi32>
//  CHECK-DAG:   %[[INIT0:.+]] = tensor.empty() : tensor<2x3x4x5x6x7x8x9xi32>
//  CHECK-DAG:   %[[INIT1:.+]] = tensor.empty() : tensor<3x4x2x9x5x6x7x8xi32>
//  CHECK-DAG:   %[[ARG1_RESHAPE:.+]] = tensor.collapse_shape %[[ARG1]] {{\[}}[0], [1, 2]{{\]}}
//  CHECK-DAG:   %[[ARG2_RESHAPE:.+]] = tensor.collapse_shape %[[ARG2]] {{\[}}[0], [1, 2, 3]{{\]}}
//  CHECK-DAG:   %[[INIT0_RESHAPE:.+]] = tensor.collapse_shape %[[INIT0]] {{\[}}[0], [1, 2], [3], [4, 5, 6], [7]{{\]}}
//  CHECK-DAG:   %[[INIT1_RESHAPE:.+]] = tensor.collapse_shape %[[INIT1]] {{\[}}[0, 1], [2], [3], [4], [5, 6, 7]{{\]}}
//      CHECK:   %[[COLLAPSED_OP:.+]]:2 = linalg.generic
// CHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]], #[[MAP0]], #[[MAP3]]]
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1_RESHAPE]], %[[ARG2_RESHAPE]] :
// CHECK-SAME:       outs(%[[INIT0_RESHAPE]], %[[INIT1_RESHAPE]] :
//      CHECK:   %[[RESULT0_RESHAPE:.+]] = tensor.expand_shape %[[COLLAPSED_OP]]#0 {{\[}}[0], [1, 2], [3], [4, 5, 6], [7]{{\]}} output_shape [2, 3, 4, 5, 6, 7, 8, 9]
//      CHECK:   %[[RESULT1_RESHAPE:.+]] = tensor.expand_shape %[[COLLAPSED_OP]]#1 {{\[}}[0, 1], [2], [3], [4], [5, 6, 7]{{\]}} output_shape [3, 4, 2, 9, 5, 6, 7, 8]
//      CHECK:   return %[[RESULT0_RESHAPE]], %[[RESULT1_RESHAPE]]

//      CONTROL: func @fuse_by_collapsing(
// CONTROL-SAME:   %[[ARG0:.+]]: tensor<2x12x5x336x9xi32>
// CONTROL-SAME:   %[[ARG1:.+]]: tensor<2x3x4xi32>
// CONTROL-SAME:   %[[ARG2:.+]]: tensor<5x6x7x8xi32>
//      CONTROL:   %[[EXPAND:.+]] = tensor.expand_shape %[[ARG0]]
//      CONTROL:   %[[GENERIC:.+]]:2 = linalg.generic
// CONTROL-SAME:       ins(%[[EXPAND]],
//      CONTROL:   return %[[GENERIC]]#0, %[[GENERIC]]#1

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>
func.func @fuse_by_collapsing_indexing_op(%arg0 : tensor<2x12x5x336x9xi32>,
    %arg1 : tensor<2x3x4xi32>, %arg2 : tensor<5x6x7x8xi32>) -> tensor<2x3x4x5x6x7x8x9xi32> {
  %expand = tensor.expand_shape %arg0 [[0], [1, 2], [3], [4, 5, 6], [7]] output_shape [2, 3, 4, 5, 6, 7, 8, 9] : tensor<2x12x5x336x9xi32> into tensor<2x3x4x5x6x7x8x9xi32>
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
//       CHECK:     %[[REM_IV1:.+]] = arith.remsi %[[IV1]], %[[C4]]
//       CHECK:     %[[DIV_IV1:.+]] = arith.divsi %[[IV1]], %[[C4]]
//       CHECK:     %[[IV2:.+]] = linalg.index 2
//       CHECK:     %[[IV3:.+]] = linalg.index 3
//       CHECK:     %[[REM1_IV3:.+]] = arith.remsi %[[IV3]], %[[C8]]
//       CHECK:     %[[DIV1_IV3:.+]] = arith.divsi %[[IV3]], %[[C8]]
//       CHECK:     %[[REM2_IV3:.+]] = arith.remsi %[[DIV1_IV3]], %[[C7]]
//       CHECK:     %[[DIV2_IV3:.+]] = arith.divsi %[[DIV1_IV3]], %[[C7]]
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
  %expand = tensor.expand_shape %arg0 [[0], [1, 2], [3], [4, 5, 6], [7]] output_shape [9, 7, 8, 2, 3, 4, 5, 6] : tensor<9x56x2x60x6xi32> into tensor<9x7x8x2x3x4x5x6xi32>
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
//      CHECK:   %[[RESULT_RESHAPE:.+]] = tensor.expand_shape %[[COLLAPSED_OP]] {{\[}}[0], [1, 2, 3], [4], [5, 6], [7]{{\]}} output_shape [2, 3, 4, 5, 6, 7, 8, 9]
//      CHECK:   return %[[RESULT_RESHAPE]]

// -----

// Dynamic case. Only checks things not covered by `fuse_by_collapsing` test above.
#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d7, d5, d6, d0, d1, d2, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d6, d0)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>
func.func @fuse_by_collapsing_dynamic(%arg0 : tensor<?x?x?x?x?xi32>,
    %arg1 : tensor<?x?x?xi32>, %arg2 : tensor<?x?x?x?xi32>, %sz0: index, %sz1: index, %sz2: index, %sz3: index, %sz4: index) -> tensor<?x3x?x5x?x7x?x?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %expand = tensor.expand_shape %arg0 [[0], [1, 2], [3], [4, 5, 6], [7]] output_shape [%sz0, 7, %sz1, %sz2, 3, %sz3, 5, %sz4]
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
//      CHECK: func @fuse_by_collapsing_dynamic
// CHECK-SAME:     (%[[ARG0:.+]]: tensor<?x?x?x?x?xi32>, %[[SZ0:.+]]: index, %[[SZ1:.+]]: index, %[[SZ2:.+]]: index, %[[SZ3:.+]]: index, %[[SZ4:.+]]: index)
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//  CHECK-DAG:   %[[C5:.+]] = arith.constant 5 : index
//      CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[ARG0]]
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[EXPAND]], %[[C2]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[EXPAND]], %[[C5]]
//      CHECK:   linalg.generic
//      CHECK:     %[[IV0:.+]] = linalg.index 1
//      CHECK:     %[[REM1_IV0:.+]] = arith.remsi %[[IV0]], %[[C5]]
//      CHECK:     %[[DIV1_IV0:.+]] = arith.divsi %[[IV0]], %[[C5]]
//      CHECK:     %[[REM2_IV0:.+]] = arith.remsi %[[DIV1_IV0]], %[[D1]]
//      CHECK:     %[[DIV2_IV0:.+]] = arith.divsi %[[DIV1_IV0]], %[[D1]]
//      CHECK:     %[[IV1:.+]] = linalg.index 3
//      CHECK:     %[[REM1_IV1:.+]] = arith.remsi %[[IV1]], %[[D0]]
//      CHECK:     %[[DIV1_IV1:.+]] = arith.divsi %[[IV1]], %[[D0]]

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3)>
func.func @fuse_reductions(%arg0 : tensor<2x?x5xf32>, %arg1 : tensor<2x5xf32>, %sz0: index) -> tensor<2x5xf32> {
  %0 = tensor.expand_shape %arg0 [[0], [1, 2], [3]] output_shape [2, 6, %sz0, 5] : tensor<2x?x5xf32> into tensor<2x6x?x5xf32>
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
// CHECK-SAME:     %[[ARG1:.+]]: tensor<2x5xf32>
// CHECK-SAME:     %[[SZ0:.+]]: index) -> tensor<2x5xf32>
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
  %0 = tensor.expand_shape %arg0 [[0], [1, 2], [3]] output_shape [2, 3, 4, 5] : tensor<2x12x5xf32> into tensor<2x3x4x5xf32>
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
  %0 = tensor.expand_shape %arg0 [[0], [1, 2], [3]] output_shape [2, 3, 4, 5] : tensor<2x12x5xf32> into tensor<2x3x4x5xf32>
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
  %0 = tensor.expand_shape %arg0 [[0], [1, 2], [3]] output_shape [2, 3, 4, 5] : tensor<2x12x5xf32> into tensor<2x3x4x5xf32>
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
  %0 = tensor.expand_shape %arg0 [[0, 1]] output_shape [2, 3] : tensor<6xf32> into tensor<2x3xf32>
  %1 = tensor.expand_shape %arg1 [[0, 1]] output_shape [4, 5] : tensor<20xf32> into tensor<4x5xf32>
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
//      CHECK:   %[[RESHAPE1:.+]] = tensor.expand_shape %[[GENERIC]] {{\[}}[0], [1, 2]{{\]}} output_shape [6, 4, 5]
//      CHECK:   %[[RESHAPE2:.+]] = tensor.expand_shape %[[RESHAPE1]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [2, 3, 4, 5]
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
//      CONTROL:     %[[RESULT:.+]] = tensor.expand_shape %[[GENERIC]] {{\[}}[0], [1], [2, 3]{{\]}} output_shape [2, 3, 4, 5]

// -----

// Corner case that isnt handled currently.
#map = affine_map<(d0) -> (d0)>
func.func @zero_D_test(%arg0: tensor<f32>) -> tensor<1xf32> {
  %0 = tensor.expand_shape %arg0 [] output_shape [1] : tensor<f32> into tensor<1xf32>
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
func.func @fuse_only_one_reassociation(%arg0 : tensor<?x?xf32>, %arg1 : tensor<4x?x?x8xf32>, %sz0: index, %sz1: index) -> tensor<4x?x?x8xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1], [2, 3]] output_shape [%sz0, 4, %sz1, 8] : tensor<?x?xf32> into tensor<?x4x?x8xf32>
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
//      CHECK: func @fuse_only_one_reassociation
// CHECK-SAME:     (%[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<4x?x?x8xf32>, %[[SZ0:.+]]: index, %[[SZ1:.+]]: index)
//  CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
//  CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//  CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//  CHECK-DAG:   %[[EXPAND_ARG0:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1], [2, 3]{{\]}} output_shape [%[[SZ0]], 4, %[[SZ1]], 8]
//  CHECK-DAG:   %[[COLLAPSE_ARG0:.+]] = tensor.collapse_shape %[[EXPAND_ARG0]] {{\[}}[0], [1], [2, 3]{{\]}}
//  CHECK-DAG:   %[[COLLAPSE_ARG1_0:.+]] = tensor.collapse_shape %[[ARG1]] {{\[}}[0], [1], [2, 3]{{\]}}
//  CHECK-DAG:   %[[COLLAPSE_ARG1_1:.+]] = tensor.collapse_shape %[[ARG1]] {{\[}}[0], [1], [2, 3]{{\]}}
//      CHECK:   %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP1]]]
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME:       ins(%[[COLLAPSE_ARG0]], %[[COLLAPSE_ARG1_0]] :
// CHECK-SAME:       outs(%[[COLLAPSE_ARG1_1]] :
//      CHECK:   %[[DIM:.+]] = tensor.dim %[[GENERIC]], %[[C1]] : tensor<4x?x?xf32>
//      CHECK:   %[[DIM_2:.+]] = tensor.dim %[[GENERIC]], %[[C2]] : tensor<4x?x?xf32>
//      CHECK:   %[[VAL_1:.+]] = arith.divsi %[[DIM_2]], %[[C8]] : index
//      CHECK:   %[[EXPANDED_3:.+]] = tensor.expand_shape %[[GENERIC]] {{\[\[}}0], [1], [2, 3]] output_shape [4, %[[DIM]], %[[VAL_1]], 8] : tensor<4x?x?xf32> into tensor<4x?x?x8xf32>
//      CHECK:   return %[[EXPANDED_3]]

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d1, d0, d2)>
func.func @fold_non_consecutive_dims(%arg0 : tensor<?x?xi32>, %sz0: index, %sz1: index) -> tensor<?x8x?x4xi32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.expand_shape %arg0 [[0, 1], [2, 3]] output_shape [%sz0, 4, %sz1, 8] : tensor<?x?xi32> into tensor<?x4x?x8xi32>
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
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xi32>, %[[SZ0:.+]]: index, %[[SZ1:.+]]: index)
//      CHECK:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   %[[C4:.+]] = arith.constant 4 : index
//      CHECK:   %[[C8:.+]] = arith.constant 8 : index
//      CHECK:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK:   %[[C2:.+]] = arith.constant 2 : index
//      CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[ARG0]] {{\[\[}}0, 1], [2, 3]] output_shape [%[[SZ0]], 4, %[[SZ1]], 8] : tensor<?x?xi32> into tensor<?x4x?x8xi32>
//      CHECK:   %[[DIM:.+]] = tensor.dim %[[EXPANDED]], %[[C0]]
//      CHECK:   %[[DIM_0:.+]] = tensor.dim %[[EXPANDED]], %[[C2]]
//      CHECK:   %[[INIT:.+]] = tensor.empty(%[[DIM_0]], %[[DIM]])
//      CHECK:   %[[COLLAPSE_INIT:.+]] = tensor.collapse_shape %[[INIT]] {{\[}}[0, 1], [2, 3]{{\]}}
//      CHECK:   %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME:       iterator_types = ["parallel", "parallel"]
// CHECK-SAME:       ins(%[[ARG0]] :
// CHECK-SAME:       outs(%[[COLLAPSE_INIT]] :
// CHECK-NEXT:   ^bb{{[0-9]}}
//      CHECK:       %[[ID0:.+]] = linalg.index 0
//  CHECK-DAG:       %[[T0:.+]] = arith.remsi %[[ID0]], %[[C4]]
//  CHECK-DAG:       %[[T1:.+]] = arith.divsi %[[ID0]], %[[C4]]
//      CHECK:       %[[ID1:.+]] = linalg.index 1
//  CHECK-DAG:       %[[T2:.+]] = arith.remsi %[[ID1]], %[[C8]]
//  CHECK-DAG:       %[[T3:.+]] = arith.divsi %[[ID1]], %[[C8]]
//  CHECK-DAG:       %[[T4:.+]] = arith.addi %[[T1]], %[[T2]]
//  CHECK-DAG:       %[[T5:.+]] = arith.addi %[[T4]], %[[T0]]
//  CHECK-DAG:       %[[T6:.+]] = arith.addi %[[T5]], %[[T3]]
//  CHECK-DAG:       %[[T7:.+]] = arith.index_cast %[[T6]]
//      CHECK:       linalg.yield %[[T7]]
//      CHECK:   %[[DIM_1:.+]] = tensor.dim %[[GENERIC]], %[[C0]] : tensor<?x?xi32>
//      CHECK:   %[[DIM_2:.+]] = tensor.dim %[[GENERIC]], %[[C1]] : tensor<?x?xi32>
//      CHECK:   %[[VAL_2:.+]] = arith.divsi %[[DIM_1]], %[[C8]] : index
//      CHECK:   %[[VAL_3:.+]] = arith.divsi %[[DIM_2]], %[[C4]] : index
//      CHECK:   %[[EXPANDED_3:.+]] = tensor.expand_shape %[[GENERIC]] {{\[\[}}0, 1], [2, 3]] output_shape [%[[VAL_2]], 8, %[[VAL_3]], 4] : tensor<?x?xi32> into tensor<?x8x?x4xi32>
//      CHECK:   return %[[EXPANDED_3]]

// -----

// None of the folded iteration space dims are contiguous reduction dimensions.
// So no change in the code.
#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> ()>
func.func @no_fold_non_consecutive_reduction_dims(%arg0 : tensor<?x?xi32>, %sz0: index, %sz1: index) -> tensor<i32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.expand_shape %arg0 [[0, 1], [2, 3]] output_shape [%sz0, 4, %sz1, 8] : tensor<?x?xi32> into tensor<?x4x?x8xi32>
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
// CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?xi32>, %[[SZ0:.+]]: index, %[[SZ1:.+]]: index)
//      CHECK:   %[[EXPAND_ARG0:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1], [2, 3]{{\]}} output_shape [%[[SZ0]], 4, %[[SZ1]], 8]
//      CHECK:   %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:       ins(%[[EXPAND_ARG0]] :
//      CHECK:   return %[[GENERIC]]

// -----

func.func @fuse_by_collapsing_pad(%arg0 : tensor<2x12x5x336x9xi32>) -> tensor<8x3x4x17x6x7x8x14xi32> {
  %expand = tensor.expand_shape %arg0 [[0], [1, 2], [3], [4, 5, 6], [7]] output_shape [2, 3, 4, 5, 6, 7, 8, 9] : tensor<2x12x5x336x9xi32> into tensor<2x3x4x5x6x7x8x9xi32>
  %cst = arith.constant 0 : i32
  %padded_0 = tensor.pad %expand low[1, 0, 0, 8, 0, 0, 0, 3] high[5, 0, 0, 4, 0, 0, 0, 2] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index,
       %arg5: index, %arg6: index, %arg7: index, %arg8: index):
    tensor.yield %cst : i32
  } : tensor<2x3x4x5x6x7x8x9xi32> to tensor<8x3x4x17x6x7x8x14xi32>
  return %padded_0 : tensor<8x3x4x17x6x7x8x14xi32>
}
//      CHECK: func @fuse_by_collapsing_pad(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<2x12x5x336x9xi32>)
//      CHECK:   %[[PAD:.+]] = tensor.pad %[[ARG0]]
// CHECK-SAME:       low[1, 0, 8, 0, 3] high[5, 0, 4, 0, 2]
//      CHECK:       tensor<2x12x5x336x9xi32> to tensor<8x12x17x336x14xi32>
//      CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[PAD]] {{\[}}[0], [1, 2], [3], [4, 5, 6], [7]]
// CHECK-SAME:       output_shape [8, 3, 4, 17, 6, 7, 8, 14] : tensor<8x12x17x336x14xi32> into tensor<8x3x4x17x6x7x8x14xi32>
//      CHECK:   return %[[EXPAND]]

// -----

func.func @no_fuse_by_collapsing_pad(%arg0 : tensor<2x12x5x336x9xi32>) -> tensor<8x5x4x17x6x7x8x14xi32> {
  %expand = tensor.expand_shape %arg0 [[0], [1, 2], [3], [4, 5, 6], [7]] output_shape [2, 3, 4, 5, 6, 7, 8, 9] : tensor<2x12x5x336x9xi32> into tensor<2x3x4x5x6x7x8x9xi32>
  %cst = arith.constant 0 : i32
  %padded_0 = tensor.pad %expand low[1, 2, 0, 8, 0, 0, 0, 3] high[5, 0, 0, 4, 0, 0, 0, 2] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index,
       %arg5: index, %arg6: index, %arg7: index, %arg8: index):
    tensor.yield %cst : i32
  } : tensor<2x3x4x5x6x7x8x9xi32> to tensor<8x5x4x17x6x7x8x14xi32>
  return %padded_0 : tensor<8x5x4x17x6x7x8x14xi32>
}
//      CHECK: func @no_fuse_by_collapsing_pad(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<2x12x5x336x9xi32>)
//      CHECK:   %[[EXPAND_ARG0:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0], [1, 2], [3], [4, 5, 6], [7]]
// CHECK-SAME:       output_shape [2, 3, 4, 5, 6, 7, 8, 9] : tensor<2x12x5x336x9xi32> into tensor<2x3x4x5x6x7x8x9xi32>
//      CHECK:   %[[PAD:.+]] = tensor.pad %[[EXPAND_ARG0]]
// CHECK-SAME:       low[1, 2, 0, 8, 0, 0, 0, 3] high[5, 0, 0, 4, 0, 0, 0, 2]
//      CHECK:       tensor<2x3x4x5x6x7x8x9xi32> to tensor<8x5x4x17x6x7x8x14xi32>
//      CHECK:   return %[[PAD]]

// -----

func.func @fuse_by_collapsing_dynamic_pad(%arg0 : tensor<?x?x?x?xf32>,
    %s0 : index, %s1 : index, %s2 : index, %s3 : index, %s4 : index, %s5 : index,
    %l0 : index, %l1 : index, %h0 : index, %h1 : index) -> tensor<?x?x?x?x?x?xf32> {
  %expand = tensor.expand_shape %arg0 [[0], [1, 2], [3], [4, 5]] output_shape [%s0, %s1, %s2, %s3, %s4, %s5] : tensor<?x?x?x?xf32> into tensor<?x?x?x?x?x?xf32>
  %cst = arith.constant 0.0 : f32
  %padded_0 = tensor.pad %expand low[%l0, 0, 0, %l1, 0, 0] high[%h0, 0, 0, %h1, 0, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index):
    tensor.yield %cst : f32
  } : tensor<?x?x?x?x?x?xf32> to tensor<?x?x?x?x?x?xf32>
  return %padded_0 : tensor<?x?x?x?x?x?xf32>
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>
//      CHECK: func @fuse_by_collapsing_dynamic_pad(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?x?x?xf32>
// CHECK-SAME:   %[[S0:.+]]: index, %[[S1:.+]]: index, %[[S2:.+]]: index, %[[S3:.+]]: index, %[[S4:.+]]: index, %[[S5:.+]]: index, %[[L0:.+]]: index, %[[L1:.+]]: index, %[[H0:.+]]: index, %[[H1:.+]]: index
//      CHECK:   %[[PAD_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[L0]], %[[H0]], %[[S0]]]
//      CHECK:   %[[PAD_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[L1]], %[[H1]], %[[S3]]]
//      CHECK:   %[[PAD:.+]] = tensor.pad %[[ARG0]]
// CHECK-SAME:       low[%[[L0]], 0, %[[L1]], 0] high[%[[H0]], 0, %[[H1]], 0]
//      CHECK:       tensor<?x?x?x?xf32> to tensor<?x?x?x?xf32>
//      CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[PAD]] {{\[}}[0], [1, 2], [3], [4, 5]]
// CHECK-SAME:       output_shape [%[[PAD_SIZE0]], %[[S1]], %[[S2]], %[[PAD_SIZE1]], %[[S4]], %[[S5]]] : tensor<?x?x?x?xf32> into tensor<?x?x?x?x?x?xf32>
//      CHECK:   return %[[EXPAND]]
