// RUN: mlir-opt -test-linalg-drop-unit-dims --split-input-file %s | FileCheck %s

// Drop only the outermost unit dimension (controlled using a control function)
func.func @drop_outermost_unit_dims(%arg0: tensor<1x1x42xf32>) -> tensor<1x1x42xf32> {
  %0 = tensor.empty() : tensor<1x1x42xf32>
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0 : tensor<1x1x42xf32>) outs(%0 : tensor<1x1x42xf32>) {
      ^bb0(%b0: f32, %b1 : f32):
        %2 = arith.addf %b0, %b1 : f32
        linalg.yield %2 : f32
    } -> tensor<1x1x42xf32>
  return %1 : tensor<1x1x42xf32>
}
// CHECK-LABEL: func @drop_outermost_unit_dims
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<1x1x42xf32>
//       CHECK:   %[[OUTS:.+]] = tensor.empty()
//       CHECK:   %[[ARG0_RESHAPE:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1], [2]{{\]}}
//       CHECK:   %[[OUTS_RESHAPE:.+]] = tensor.collapse_shape %[[OUTS]] {{\[}}[0, 1], [2]{{\]}}
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[ARG0_RESHAPE]] :
//  CHECK-SAME:       outs(%[[OUTS_RESHAPE]] :
//       CHECK:   %[[EXPAND_SHAPE:.+]] = tensor.expand_shape %[[GENERIC]] {{\[}}[0, 1], [2]{{\]}}
//       CHECK:   return %[[EXPAND_SHAPE]]
