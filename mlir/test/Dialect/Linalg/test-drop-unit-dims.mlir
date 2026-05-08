// RUN: mlir-opt -test-linalg-drop-unit-dims --split-input-file %s | FileCheck %s --check-prefixes=CHECK,NOENCODE
// RUN: mlir-opt -test-linalg-drop-unit-dims=collapse-encoded --split-input-file %s | FileCheck %s --check-prefixes=CHECK,ENCODE

// Drop only the outermost unit dimension (controlled using a control function)
// This test does not use an encoding, therefore behavior in both modes is identical.
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

// Drop outermost unit dimension with operand that has an encoding.
// With the default behavior, the transformation is aborted and operation remains unchanged.
// With the custom behavior, the operand gets collapsed and encoding is preserved in the collapse.

#encoding = #test.tensor_encoding<"encoding">

func.func @drop_unit_dims_encoded_operand(%arg0: tensor<1x1x42xf32>, %arg1: tensor<1x1x42xf32, #encoding>) -> tensor<1x1x42xf32> {
  %0 = tensor.empty() : tensor<1x1x42xf32>
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<1x1x42xf32>, tensor<1x1x42xf32, #encoding>) outs(%0 : tensor<1x1x42xf32>) {
      ^bb0(%in0: f32, %in1 : f32, %out : f32):
        %2 = arith.addf %in0, %in1 : f32
        linalg.yield %2 : f32
    } -> tensor<1x1x42xf32>
  return %1 : tensor<1x1x42xf32>
}

// NOENCODE-LABEL:   @drop_unit_dims_encoded_operand(
// NOENCODE-SAME:      %[[ARG0:.*]]: tensor<1x1x42xf32>,
// NOENCODE-SAME:      %[[ARG1:.*]]: tensor<1x1x42xf32, #test.tensor_encoding<"encoding">>) -> tensor<1x1x42xf32> {
// NOENCODE:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<1x1x42xf32>
// NOENCODE:           %[[GENERIC_0:.*]] = linalg.generic
// NOENCODE-SAME:        iterator_types = ["parallel", "parallel", "parallel"]}
// NOENCODE-SAME:        ins(%[[ARG0]], %[[ARG1]] : tensor<1x1x42xf32>, tensor<1x1x42xf32, #test.tensor_encoding<"encoding">>)
// NOENCODE-SAME:        outs(%[[EMPTY_0]] : tensor<1x1x42xf32>)
// NOENCODE:           return %[[GENERIC_0]] : tensor<1x1x42xf32>

//   ENCODE-LABEL:   @drop_unit_dims_encoded_operand(
//   ENCODE-SAME:      %[[ARG0:.*]]: tensor<1x1x42xf32>,
//   ENCODE-SAME:      %[[ARG1:.*]]: tensor<1x1x42xf32, #test.tensor_encoding<"encoding">>) -> tensor<1x1x42xf32> {
//   ENCODE:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<1x1x42xf32>
//   ENCODE:           %[[COLLAPSE_SHAPE_0:.*]] = tensor.collapse_shape %[[ARG0]] {{\[\[}}0, 1], [2]] : tensor<1x1x42xf32> into tensor<1x42xf32>
//   ENCODE:           %[[COLLAPSE_SHAPE_1:.*]] = tensor.collapse_shape %[[ARG1]] {{\[\[}}0, 1], [2]] : tensor<1x1x42xf32, #test.tensor_encoding<"encoding">> into tensor<1x42xf32, #test.tensor_encoding<"encoding">>
//   ENCODE:           %[[COLLAPSE_SHAPE_2:.*]] = tensor.collapse_shape %[[EMPTY_0]] {{\[\[}}0, 1], [2]] : tensor<1x1x42xf32> into tensor<1x42xf32>
//   ENCODE:           %[[GENERIC_0:.*]] = linalg.generic
//   ENCODE-SAME:        iterator_types = ["parallel", "parallel"]
//   ENCODE-SAME:        ins(%[[COLLAPSE_SHAPE_0]], %[[COLLAPSE_SHAPE_1]] : tensor<1x42xf32>, tensor<1x42xf32, #test.tensor_encoding<"encoding">>)
//   ENCODE-SAME:        outs(%[[COLLAPSE_SHAPE_2]] : tensor<1x42xf32>)
//   ENCODE:           %[[EXPAND_SHAPE_0:.*]] = tensor.expand_shape %[[GENERIC_0]] {{\[\[}}0, 1], [2]] output_shape [1, 1, 42] : tensor<1x42xf32> into tensor<1x1x42xf32>
//   ENCODE:           return %[[EXPAND_SHAPE_0]] : tensor<1x1x42xf32>

// -----

// Drop outermost unit dimension with result that has an encoding.
// With the default behavior, the transformation is aborted and operation remains unchanged.
// With the custom behavior, the result gets expanded and encoding is preserved in the expansion.

#encoding = #test.tensor_encoding<"encoding">

func.func @drop_unit_dims_encoded_result(%arg0: tensor<1x1x42xf32>, %arg1: tensor<1x1x42xf32>) -> tensor<1x1x42xf32, #encoding> {
  %0 = tensor.empty() : tensor<1x1x42xf32, #encoding>
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<1x1x42xf32>, tensor<1x1x42xf32>) outs(%0 : tensor<1x1x42xf32, #encoding>) {
      ^bb0(%in0: f32, %in1 : f32, %out : f32):
        %2 = arith.addf %in0, %in1 : f32
        linalg.yield %2 : f32
    } -> tensor<1x1x42xf32, #encoding>
  return %1 : tensor<1x1x42xf32, #encoding>
}

// NOENCODE-LABEL:   @drop_unit_dims_encoded_result(
// NOENCODE-SAME:      %[[ARG0:.*]]: tensor<1x1x42xf32>,
// NOENCODE-SAME:      %[[ARG1:.*]]: tensor<1x1x42xf32>) -> tensor<1x1x42xf32, #test.tensor_encoding<"encoding">>
// NOENCODE:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<1x1x42xf32, #test.tensor_encoding<"encoding">>
// NOENCODE:           %[[GENERIC_0:.*]] = linalg.generic
// NOENCODE-SAME:        iterator_types = ["parallel", "parallel", "parallel"]
// NOENCODE-SAME:        ins(%[[ARG0]], %[[ARG1]] : tensor<1x1x42xf32>, tensor<1x1x42xf32>)
// NOENCODE-SAME:        outs(%[[EMPTY_0]] : tensor<1x1x42xf32, #test.tensor_encoding<"encoding">>)
// NOENCODE-NOT:       tensor.expand_shape
// NOENCODE:           return %[[GENERIC_0]] : tensor<1x1x42xf32, #test.tensor_encoding<"encoding">>

//   ENCODE-LABEL:   @drop_unit_dims_encoded_result(
//   ENCODE-SAME:      %[[ARG0:.*]]: tensor<1x1x42xf32>,
//   ENCODE-SAME:      %[[ARG1:.*]]: tensor<1x1x42xf32>) -> tensor<1x1x42xf32, #test.tensor_encoding<"encoding">>
//   ENCODE:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<1x1x42xf32, #test.tensor_encoding<"encoding">>
//   ENCODE:           %[[COLLAPSE_SHAPE_0:.*]] = tensor.collapse_shape %[[ARG0]] {{\[\[}}0, 1], [2]] : tensor<1x1x42xf32> into tensor<1x42xf32>
//   ENCODE:           %[[COLLAPSE_SHAPE_1:.*]] = tensor.collapse_shape %[[ARG1]] {{\[\[}}0, 1], [2]] : tensor<1x1x42xf32> into tensor<1x42xf32>
//   ENCODE:           %[[COLLAPSE_SHAPE_2:.*]] = tensor.collapse_shape %[[EMPTY_0]] {{\[\[}}0, 1], [2]] : tensor<1x1x42xf32, #test.tensor_encoding<"encoding">> into tensor<1x42xf32, #test.tensor_encoding<"encoding">>
//   ENCODE:           %[[GENERIC_0:.*]] = linalg.generic
//   ENCODE-SAME:        iterator_types = ["parallel", "parallel"]
//   ENCODE-SAME:        ins(%[[COLLAPSE_SHAPE_0]], %[[COLLAPSE_SHAPE_1]] : tensor<1x42xf32>, tensor<1x42xf32>)
//   ENCODE-SAME:        outs(%[[COLLAPSE_SHAPE_2]] : tensor<1x42xf32, #test.tensor_encoding<"encoding">>)
//   ENCODE:           %[[EXPAND_SHAPE_0:.*]] = tensor.expand_shape %[[GENERIC_0]] {{\[\[}}0, 1], [2]] output_shape [1, 1, 42] : tensor<1x42xf32, #test.tensor_encoding<"encoding">> into tensor<1x1x42xf32, #test.tensor_encoding<"encoding">>
//   ENCODE:           return %[[EXPAND_SHAPE_0]] : tensor<1x1x42xf32, #test.tensor_encoding<"encoding">>

// -----

#encoding1 = #test.tensor_encoding<"encoding1">
#encoding2 = #test.tensor_encoding<"encoding2">

func.func @drop_unit_dims_encoded_operand_and_result(%arg0: tensor<1x1x42xf32>, %arg1: tensor<1x1x42xf32, #encoding1>) -> tensor<1x1x42xf32, #encoding2> {
  %0 = tensor.empty() : tensor<1x1x42xf32, #encoding2>
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<1x1x42xf32>, tensor<1x1x42xf32, #encoding1>) outs(%0 : tensor<1x1x42xf32, #encoding2>) {
      ^bb0(%in0: f32, %in1 : f32, %out : f32):
        %2 = arith.addf %in0, %in1 : f32
        linalg.yield %2 : f32
    } -> tensor<1x1x42xf32, #encoding2>
  return %1 : tensor<1x1x42xf32, #encoding2>
}

// NOENCODE-LABEL:   @drop_unit_dims_encoded_operand_and_result
// NOENCODE-SAME:      %[[ARG0:.*]]: tensor<1x1x42xf32>
// NOENCODE-SAME:      %[[ARG1:.*]]: tensor<1x1x42xf32, #test.tensor_encoding<"encoding1">>
// NOENCODE-SAME:      -> tensor<1x1x42xf32, #test.tensor_encoding<"encoding2">>
// NOENCODE:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<1x1x42xf32, #test.tensor_encoding<"encoding2">>
// NOENCODE:           %[[GENERIC_0:.*]] = linalg.generic
// NOENCODE-SAME:        iterator_types = ["parallel", "parallel", "parallel"]
// NOENCODE-SAME:        ins(%[[ARG0]], %[[ARG1]] : tensor<1x1x42xf32>, tensor<1x1x42xf32, #test.tensor_encoding<"encoding1">>)
// NOENCODE-SAME:        outs(%[[EMPTY_0]] : tensor<1x1x42xf32, #test.tensor_encoding<"encoding2">>)
// NOENCODE-NOT:       tensor.expand_shape
// NOENCODE:           return %[[GENERIC_0]] : tensor<1x1x42xf32, #test.tensor_encoding<"encoding2">>

//   ENCODE-LABEL:   @drop_unit_dims_encoded_operand_and_result
//   ENCODE-SAME:      %[[ARG0:.*]]: tensor<1x1x42xf32>,
//   ENCODE-SAME:      %[[ARG1:.*]]: tensor<1x1x42xf32, #test.tensor_encoding<"encoding1">>
//   ENCODE-SAME:      -> tensor<1x1x42xf32, #test.tensor_encoding<"encoding2">>
//   ENCODE:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<1x1x42xf32, #test.tensor_encoding<"encoding2">>
//   ENCODE:           %[[COLLAPSE_SHAPE_0:.*]] = tensor.collapse_shape %[[ARG0]] {{\[\[}}0, 1], [2]] : tensor<1x1x42xf32> into tensor<1x42xf32>
//   ENCODE:           %[[COLLAPSE_SHAPE_1:.*]] = tensor.collapse_shape %[[ARG1]] {{\[\[}}0, 1], [2]] : tensor<1x1x42xf32, #test.tensor_encoding<"encoding1">>
//   ENCODE-SAME:        into tensor<1x42xf32, #test.tensor_encoding<"encoding1">>
//   ENCODE:           %[[COLLAPSE_SHAPE_2:.*]] = tensor.collapse_shape %[[EMPTY_0]] {{\[\[}}0, 1], [2]] : tensor<1x1x42xf32, #test.tensor_encoding<"encoding2">>
//   ENCODE-SAME:        into tensor<1x42xf32, #test.tensor_encoding<"encoding2">>
//   ENCODE:           %[[GENERIC_0:.*]] = linalg.generic
//   ENCODE-SAME:        iterator_types = ["parallel", "parallel"]
//   ENCODE-SAME:        ins(%[[COLLAPSE_SHAPE_0]], %[[COLLAPSE_SHAPE_1]] : tensor<1x42xf32>, tensor<1x42xf32, #test.tensor_encoding<"encoding1">>)
//   ENCODE-SAME:        outs(%[[COLLAPSE_SHAPE_2]] : tensor<1x42xf32, #test.tensor_encoding<"encoding2">>)
//   ENCODE:           %[[EXPAND_SHAPE_0:.*]] = tensor.expand_shape %[[GENERIC_0]] {{\[\[}}0, 1], [2]] output_shape [1, 1, 42] : tensor<1x42xf32, #test.tensor_encoding<"encoding2">>
//   ENCODE-SAME:        into tensor<1x1x42xf32, #test.tensor_encoding<"encoding2">>
//   ENCODE:           return %[[EXPAND_SHAPE_0]] : tensor<1x1x42xf32, #test.tensor_encoding<"encoding2">>

// -----

#encoding1 = #test.tensor_encoding<"encoding1">
#encoding2 = #test.tensor_encoding<"encoding2">

func.func @drop_unit_dims_two_encoded_operands(%arg0: tensor<1x1x42xf32, #encoding1>, %arg1: tensor<1x1x42xf32, #encoding2>) -> tensor<1x1x42xf32> {
  %0 = tensor.empty() : tensor<1x1x42xf32>
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<1x1x42xf32, #encoding1>, tensor<1x1x42xf32, #encoding2>) outs(%0 : tensor<1x1x42xf32>) {
      ^bb0(%in0: f32, %in1 : f32, %out : f32):
        %2 = arith.addf %in0, %in1 : f32
        linalg.yield %2 : f32
    } -> tensor<1x1x42xf32>
  return %1 : tensor<1x1x42xf32>
}

// NOENCODE-LABEL:   @drop_unit_dims_two_encoded_operands
// NOENCODE-SAME:      %[[ARG0:.*]]: tensor<1x1x42xf32, #test.tensor_encoding<"encoding1">>
// NOENCODE-SAME:      %[[ARG1:.*]]: tensor<1x1x42xf32, #test.tensor_encoding<"encoding2">>) -> tensor<1x1x42xf32>
// NOENCODE:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<1x1x42xf32>
// NOENCODE:           %[[GENERIC_0:.*]] = linalg.generic
// NOENCODE-SAME:        iterator_types = ["parallel", "parallel", "parallel"]}
// NOENCODE-SAME:        ins(%[[ARG0]], %[[ARG1]] : tensor<1x1x42xf32, #test.tensor_encoding<"encoding1">>, tensor<1x1x42xf32, #test.tensor_encoding<"encoding2">>)
// NOENCODE-SAME:        outs(%[[EMPTY_0]] : tensor<1x1x42xf32>)
// NOENCODE:           return %[[GENERIC_0]] : tensor<1x1x42xf32>

//   ENCODE-LABEL:   @drop_unit_dims_two_encoded_operands
//   ENCODE-SAME:      %[[ARG0:.*]]: tensor<1x1x42xf32, #test.tensor_encoding<"encoding1">>
//   ENCODE-SAME:      %[[ARG1:.*]]: tensor<1x1x42xf32, #test.tensor_encoding<"encoding2">>) -> tensor<1x1x42xf32>
//   ENCODE:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<1x1x42xf32>
//   ENCODE:           %[[COLLAPSE_SHAPE_0:.*]] = tensor.collapse_shape %[[ARG0]] {{\[\[}}0, 1], [2]] : tensor<1x1x42xf32, #test.tensor_encoding<"encoding1">>
//   ENCODE-SAME:        into tensor<1x42xf32, #test.tensor_encoding<"encoding1">>
//   ENCODE:           %[[COLLAPSE_SHAPE_1:.*]] = tensor.collapse_shape %[[ARG1]] {{\[\[}}0, 1], [2]] : tensor<1x1x42xf32, #test.tensor_encoding<"encoding2">>
//   ENCODE-SAME:        into tensor<1x42xf32, #test.tensor_encoding<"encoding2">>
//   ENCODE:           %[[COLLAPSE_SHAPE_2:.*]] = tensor.collapse_shape %[[EMPTY_0]] {{\[\[}}0, 1], [2]] : tensor<1x1x42xf32> into tensor<1x42xf32>
//   ENCODE:           %[[GENERIC_0:.*]] = linalg.generic
//   ENCODE-SAME:        iterator_types = ["parallel", "parallel"]
//   ENCODE-SAME:        ins(%[[COLLAPSE_SHAPE_0]], %[[COLLAPSE_SHAPE_1]] : tensor<1x42xf32, #test.tensor_encoding<"encoding1">>, tensor<1x42xf32, #test.tensor_encoding<"encoding2">>)
//   ENCODE-SAME:        outs(%[[COLLAPSE_SHAPE_2]] : tensor<1x42xf32>)
//   ENCODE:           %[[EXPAND_SHAPE_0:.*]] = tensor.expand_shape %[[GENERIC_0]] {{\[\[}}0, 1], [2]] output_shape [1, 1, 42] : tensor<1x42xf32> into tensor<1x1x42xf32>
//   ENCODE:           return %[[EXPAND_SHAPE_0]] : tensor<1x1x42xf32>
