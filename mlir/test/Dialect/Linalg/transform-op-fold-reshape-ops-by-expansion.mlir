// RUN: mlir-opt %s \
// RUN: -transform-preload-library='transform-library-paths=%p/td/fold-reshape-ops-by-expansion.mlir' \
// RUN: -transform-interpreter=entry-point=fold_reshape_ops_by_expansion \
// RUN: | FileCheck %s --check-prefixes=CHECK,UNRESTRICTED
// RUN: mlir-opt %s \
// RUN: -transform-preload-library='transform-library-paths=%p/td/fold-reshape-ops-by-expansion.mlir' \
// RUN: -transform-interpreter=entry-point=fold_reshape_ops_by_expansion_unrestricted \
// RUN: | FileCheck %s --check-prefixes=CHECK,UNRESTRICTED
// RUN: mlir-opt %s \
// RUN: -transform-preload-library='transform-library-paths=%p/td/fold-reshape-ops-by-expansion.mlir' \
// RUN: -transform-interpreter=entry-point=fold_reshape_ops_by_expansion_single_use \
// RUN: | FileCheck %s --check-prefixes=CHECK,SINGLE

// CHECK-LABEL: func.func @collapse_single_use
//  CHECK-NEXT:   %[[COLLAPSE_SINGLE_INIT:.*]] = tensor.expand_shape
//  CHECK-NEXT:   %[[COLLAPSE_SINGLE_FUSED:.*]] = linalg.generic
//  CHECK-SAME:     iterator_types = ["parallel", "parallel"]
//  CHECK-SAME:     outs(%[[COLLAPSE_SINGLE_INIT]] : tensor<2x3xf32>)
//       CHECK:   } -> tensor<2x3xf32>
//  CHECK-NEXT:   %[[COLLAPSE_SINGLE_RESULT:.*]] = tensor.collapse_shape %[[COLLAPSE_SINGLE_FUSED]]
//  CHECK-NEXT:   return %[[COLLAPSE_SINGLE_RESULT]]
func.func @collapse_single_use(%arg: tensor<2x3xf32>, %init: tensor<6xf32>) -> tensor<6xf32> {
  %collapsed = tensor.collapse_shape %arg [[0, 1]] : tensor<2x3xf32> into tensor<6xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%collapsed : tensor<6xf32>) outs(%init : tensor<6xf32>) {
  ^bb0(%in: f32, %out: f32):
    %neg = arith.negf %in : f32
    linalg.yield %neg : f32
  } -> tensor<6xf32>
  return %result : tensor<6xf32>
}

//       CHECK-LABEL: func.func @collapse_multi_use
//        CHECK-NEXT:   %[[COLLAPSED:.*]] = tensor.collapse_shape
// UNRESTRICTED-NEXT:   %[[COLLAPSE_MULTI_INIT:.*]] = tensor.expand_shape
// UNRESTRICTED-NEXT:   %[[COLLAPSE_MULTI_FUSED:.*]] = linalg.generic
// UNRESTRICTED-SAME:     iterator_types = ["parallel", "parallel"]
// UNRESTRICTED-SAME:     outs(%[[COLLAPSE_MULTI_INIT]] : tensor<2x3xf32>)
//      UNRESTRICTED:   } -> tensor<2x3xf32>
// UNRESTRICTED-NEXT:   %[[COLLAPSE_MULTI_RESULT:.*]] = tensor.collapse_shape %[[COLLAPSE_MULTI_FUSED]]
// UNRESTRICTED-NEXT:   return %[[COLLAPSED]], %[[COLLAPSE_MULTI_RESULT]]
//       SINGLE-NEXT:   %[[COLLAPSE_MULTI_ORIGINAL:.*]] = linalg.generic
//       SINGLE-SAME:     iterator_types = ["parallel"]
//       SINGLE-SAME:     ins(%[[COLLAPSED]] : tensor<6xf32>)
//            SINGLE:   } -> tensor<6xf32>
//       SINGLE-NEXT:   return %[[COLLAPSED]], %[[COLLAPSE_MULTI_ORIGINAL]]
func.func @collapse_multi_use(%arg: tensor<2x3xf32>, %init: tensor<6xf32>) -> (tensor<6xf32>, tensor<6xf32>) {
  %collapsed = tensor.collapse_shape %arg [[0, 1]] : tensor<2x3xf32> into tensor<6xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%collapsed : tensor<6xf32>) outs(%init : tensor<6xf32>) {
  ^bb0(%in: f32, %out: f32):
    %neg = arith.negf %in : f32
    linalg.yield %neg : f32
  } -> tensor<6xf32>
  return %collapsed, %result : tensor<6xf32>, tensor<6xf32>
}

// CHECK-LABEL: func.func @expand_single_use
//   CHECK-DAG:   %[[EXPAND_SINGLE_INPUT:.*]] = tensor.expand_shape %arg0
//   CHECK-DAG:   %[[EXPAND_SINGLE_INIT:.*]] = tensor.expand_shape %arg1
//  CHECK-NEXT:   %[[EXPAND_SINGLE_FUSED:.*]] = linalg.generic
//  CHECK-SAME:     iterator_types = ["parallel", "parallel"]
//  CHECK-SAME:     ins(%[[EXPAND_SINGLE_INPUT]] : tensor<2x3xf32>)
//  CHECK-SAME:     outs(%[[EXPAND_SINGLE_INIT]] : tensor<2x3xf32>)
//       CHECK:   } -> tensor<2x3xf32>
//  CHECK-NEXT:   return %[[EXPAND_SINGLE_FUSED]]
func.func @expand_single_use(%arg: tensor<6xf32>, %init: tensor<6xf32>) -> tensor<2x3xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%arg : tensor<6xf32>) outs(%init : tensor<6xf32>) {
  ^bb0(%in: f32, %out: f32):
    %neg = arith.negf %in : f32
    linalg.yield %neg : f32
  } -> tensor<6xf32>
  %expanded = tensor.expand_shape %result [[0, 1]] output_shape [2, 3] : tensor<6xf32> into tensor<2x3xf32>
  return %expanded : tensor<2x3xf32>
}

//       CHECK-LABEL: func.func @expand_multi_use
//  UNRESTRICTED-DAG:   %[[EXPAND_MULTI_INPUT:.*]] = tensor.expand_shape %arg0
//  UNRESTRICTED-DAG:   %[[EXPAND_MULTI_INIT:.*]] = tensor.expand_shape %arg1
// UNRESTRICTED-NEXT:   %[[EXPAND_MULTI_FUSED:.*]] = linalg.generic
// UNRESTRICTED-SAME:     iterator_types = ["parallel", "parallel"]
// UNRESTRICTED-SAME:     ins(%[[EXPAND_MULTI_INPUT]] : tensor<2x3xf32>)
// UNRESTRICTED-SAME:     outs(%[[EXPAND_MULTI_INIT]] : tensor<2x3xf32>)
//      UNRESTRICTED:   } -> tensor<2x3xf32>
// UNRESTRICTED-NEXT:   %[[EXPAND_MULTI_COLLAPSED:.*]] = tensor.collapse_shape %[[EXPAND_MULTI_FUSED]]
// UNRESTRICTED-NEXT:   return %[[EXPAND_MULTI_COLLAPSED]], %[[EXPAND_MULTI_FUSED]]
//       SINGLE-NEXT:   %[[EXPAND_MULTI_ORIGINAL:.*]] = linalg.generic
//       SINGLE-SAME:     iterator_types = ["parallel"]
//            SINGLE:   } -> tensor<6xf32>
//       SINGLE-NEXT:   %[[EXPAND_MULTI_RESULT:.*]] = tensor.expand_shape %[[EXPAND_MULTI_ORIGINAL]]
//       SINGLE-NEXT:   return %[[EXPAND_MULTI_ORIGINAL]], %[[EXPAND_MULTI_RESULT]]
func.func @expand_multi_use(%arg: tensor<6xf32>, %init: tensor<6xf32>) -> (tensor<6xf32>, tensor<2x3xf32>) {
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%arg : tensor<6xf32>) outs(%init : tensor<6xf32>) {
  ^bb0(%in: f32, %out: f32):
    %neg = arith.negf %in : f32
    linalg.yield %neg : f32
  } -> tensor<6xf32>
  %expanded = tensor.expand_shape %result [[0, 1]] output_shape [2, 3] : tensor<6xf32> into tensor<2x3xf32>
  return %result, %expanded : tensor<6xf32>, tensor<2x3xf32>
}
