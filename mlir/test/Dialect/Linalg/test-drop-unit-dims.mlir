// RUN: mlir-opt -test-linalg-drop-unit-dims --split-input-file %s | FileCheck %s
// RUN: mlir-opt -test-linalg-drop-unit-dims="collapse-encoded" --split-input-file %s | FileCheck %s --check-prefix=ENCODED

#encoding = #test.tensor_encoding<"encoding">

// Drop only the outermost unit dimension (controlled using a control function)
// In default mode, only the input with no encoding is collapsed.
// With the `collape-encoded` flag, the encoded input is also collapsed.
func.func @drop_outermost_unit_dims(%arg0: tensor<1x1x42xf32>, %arg1: tensor<1x1x42xf32, #encoding>) -> tensor<1x1x42xf32> {
  %0 = tensor.empty() : tensor<1x1x42xf32>
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<1x1x42xf32>, tensor<1x1x42xf32, #encoding>)
    outs(%0 : tensor<1x1x42xf32>) {
      ^bb0(%b0: f32, %b1: f32, %b2: f32):
        %2 = arith.addf %b0, %b1 : f32
        linalg.yield %2 : f32
    } -> tensor<1x1x42xf32>
  return %1 : tensor<1x1x42xf32>
}

// Without the collapse-encoded flag, only the non-encoded tensor is collapsed
// CHECK-LABEL: func @drop_outermost_unit_dims
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<1x1x42xf32>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<1x1x42xf32, #test.tensor_encoding<"encoding">>
//       CHECK:   %[[OUTS:.+]] = tensor.empty()
//       CHECK:   %[[ARG0_RESHAPE:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1], [2]{{\]}}
//       CHECK:   %[[OUTS_RESHAPE:.+]] = tensor.collapse_shape %[[OUTS]] {{\[}}[0, 1], [2]{{\]}}
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[ARG0_RESHAPE]], %[[ARG1]] :
//  CHECK-SAME:       outs(%[[OUTS_RESHAPE]] :
//       CHECK:   %[[EXPAND_SHAPE:.+]] = tensor.expand_shape %[[GENERIC]] {{\[}}[0, 1], [2]{{\]}}
//       CHECK:   return %[[EXPAND_SHAPE]]

// With the collapse-encoded flag, both tensors are collapsed and encodings are preserved
// ENCODED: affine_map<(d0, d1) -> (d0, d1)>
// ENCODED-LABEL: func @drop_outermost_unit_dims
//  ENCODED-SAME:     %[[ARG0:.+]]: tensor<1x1x42xf32>
//  ENCODED-SAME:     %[[ARG1:.+]]: tensor<1x1x42xf32, #test.tensor_encoding<"encoding">>
//       ENCODED:   %[[OUTS:.+]] = tensor.empty() : tensor<1x1x42xf32>
//       ENCODED:   %[[ARG0_RESHAPE:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1], [2]{{\]}}
//  ENCODED-SAME:       : tensor<1x1x42xf32> into tensor<1x42xf32>
//       ENCODED:   %[[ARG1_RESHAPE:.+]] = tensor.collapse_shape %[[ARG1]] {{\[}}[0, 1], [2]{{\]}}
//  ENCODED-SAME:       : tensor<1x1x42xf32, #test.tensor_encoding<"encoding">> into tensor<1x42xf32, #test.tensor_encoding<"encoding">>
//       ENCODED:   %[[OUTS_RESHAPE:.+]] = tensor.collapse_shape %[[OUTS]] {{\[}}[0, 1], [2]{{\]}}
//  ENCODED-SAME:       : tensor<1x1x42xf32> into tensor<1x42xf32>
//       ENCODED:   %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
//  ENCODED-SAME:       ins(%[[ARG0_RESHAPE]], %[[ARG1_RESHAPE]] : tensor<1x42xf32>, tensor<1x42xf32, #test.tensor_encoding<"encoding">>)
//  ENCODED-SAME:       outs(%[[OUTS_RESHAPE]] : tensor<1x42xf32>)
//       ENCODED:   %[[EXPAND_SHAPE:.+]] = tensor.expand_shape %[[GENERIC]] {{\[}}[0, 1], [2]{{\]}}
//  ENCODED-SAME:       output_shape [1, 1, 42] {test.unit_dims_expanded} : tensor<1x42xf32> into tensor<1x1x42xf32>
//       ENCODED:   return %[[EXPAND_SHAPE]]
