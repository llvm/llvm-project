// RUN: mlir-opt -test-linalg-drop-unit-dims --split-input-file %s | FileCheck %s
// RUN: mlir-opt -test-linalg-drop-unit-dims="preserve-encoding" --split-input-file %s | FileCheck %s --check-prefix=PRESERVE

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

// -----

#encoding = #test.tensor_encoding<"encoding">

// Test that tensor encodings are preserved when collapsing unit dimensions
func.func @drop_outermost_unit_dims_with_encoding(%arg0: tensor<1x1x42xf32, #encoding>) -> tensor<1x1x42xf32, #encoding> {
  %0 = tensor.empty() : tensor<1x1x42xf32, #encoding>
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0 : tensor<1x1x42xf32, #encoding>)
    outs(%0 : tensor<1x1x42xf32, #encoding>) {
      ^bb0(%b0: f32, %b1 : f32):
        %2 = arith.addf %b0, %b1 : f32
        linalg.yield %2 : f32
    } -> tensor<1x1x42xf32, #encoding>
  return %1 : tensor<1x1x42xf32, #encoding>
}
// Without preserve-encoding flag, encoded tensors are not collapsed
// CHECK-LABEL: func @drop_outermost_unit_dims_with_encoding
//       CHECK:   linalg.generic
//   CHECK-NOT:   tensor.collapse_shape
//   CHECK-NOT:   tensor.expand_shape

// With preserve-encoding flag, encodings are preserved through collapse/expand
// PRESERVE: affine_map<(d0, d1) -> (d0, d1)>
// PRESERVE-LABEL: func @drop_outermost_unit_dims_with_encoding
//  PRESERVE-SAME:     %[[ARG0:.+]]: tensor<1x1x42xf32, #test.tensor_encoding<"encoding">>
//       PRESERVE:   %[[OUTS:.+]] = tensor.empty() : tensor<1x1x42xf32, #test.tensor_encoding<"encoding">>
//       PRESERVE:   %[[ARG0_RESHAPE:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1], [2]{{\]}}
//  PRESERVE-SAME:       : tensor<1x1x42xf32, #test.tensor_encoding<"encoding">> into tensor<1x42xf32, #test.tensor_encoding<"encoding">>
//       PRESERVE:   %[[OUTS_RESHAPE:.+]] = tensor.collapse_shape %[[OUTS]] {{\[}}[0, 1], [2]{{\]}}
//  PRESERVE-SAME:       : tensor<1x1x42xf32, #test.tensor_encoding<"encoding">> into tensor<1x42xf32, #test.tensor_encoding<"encoding">>
//       PRESERVE:   %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
//  PRESERVE-SAME:       ins(%[[ARG0_RESHAPE]] : tensor<1x42xf32, #test.tensor_encoding<"encoding">>)
//  PRESERVE-SAME:       outs(%[[OUTS_RESHAPE]] : tensor<1x42xf32, #test.tensor_encoding<"encoding">>)
//       PRESERVE:   %[[EXPAND_SHAPE:.+]] = tensor.expand_shape %[[GENERIC]] {{\[}}[0, 1], [2]{{\]}}
//  PRESERVE-SAME:       : tensor<1x42xf32, #test.tensor_encoding<"encoding">> into tensor<1x1x42xf32, #test.tensor_encoding<"encoding">>
//       PRESERVE:   return %[[EXPAND_SHAPE]]
