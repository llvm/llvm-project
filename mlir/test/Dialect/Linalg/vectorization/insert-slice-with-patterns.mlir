// RUN: mlir-opt %s -transform-interpreter -split-input-file | FileCheck %s

///----------------------------------------------------------------------------------------
/// Tests for tensor.insert_slice
///----------------------------------------------------------------------------------------

// The pad value for xfer-read is neither needed nor available - use the default (0.0).

// CHECK-LABEL: func @insert_static_slice_default_pad
// CHECK-SAME:      %[[ARG_0:.*]]: tensor<1x2x3xf32>,
// CHECK-SAME:      %[[ARG_1:.*]]: tensor<9x8x7x1x2x3xf32>) -> tensor<9x8x7x1x2x3xf32> {
// CHECK:           %[[PAD:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[READ:.*]] = vector.transfer_read %[[ARG_0]]{{\[}}%[[C0]], %[[C0]], %[[C0]]], %[[PAD]] {in_bounds = [true, true, true]} : tensor<1x2x3xf32>, vector<1x2x3xf32>
// CHECK:           %[[WRITE:.*]] = vector.transfer_write %[[READ]], %[[ARG_1]]{{\[}}%[[C0]], %[[C0]], %[[C0]], %[[C0]], %[[C0]], %[[C0]]] {in_bounds = [true, true, true]} : vector<1x2x3xf32>, tensor<9x8x7x1x2x3xf32>
// CHECK:           return %[[WRITE]] : tensor<9x8x7x1x2x3xf32>
func.func @insert_static_slice_default_pad(%arg0: tensor<1x2x3xf32>, %arg1: tensor<9x8x7x1x2x3xf32>) -> tensor<9x8x7x1x2x3xf32> {
  %res = tensor.insert_slice %arg0 into %arg1[0, 0, 0, 0, 0, 0] [1, 1, 1, 1, 2, 3][1, 1, 1, 1, 1, 1] : tensor<1x2x3xf32> into tensor<9x8x7x1x2x3xf32>
  return %res : tensor<9x8x7x1x2x3xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.insert_slice"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 { vectorize_padding } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Same as above, but there's a pad value available that should be used instead of the default value.

// CHECK-LABEL:   func.func @insert_static_slice_non_zero_pad
// CHECK-SAME:      %[[ARG_0:.*]]: tensor<1x2x3xf32>,
// CHECK-SAME:      %[[PAD:.*]]: f32) -> tensor<9x8x7x1x2x3xf32> {
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<9x8x7x1x2x3xf32>
// CHECK:           %[[BC:.*]] = vector.broadcast %[[PAD]] : f32 to vector<9x8x7x1x2x3xf32>
// CHECK:           %[[WRITE:.*]] = vector.transfer_write %[[BC]], %[[EMPTY]]{{.*}} {in_bounds = [true, true, true, true, true, true]} : vector<9x8x7x1x2x3xf32>, tensor<9x8x7x1x2x3xf32>
// CHECK:           %[[READ:.*]] = vector.transfer_read %[[ARG_0]]{{.*}}, %[[PAD]] {in_bounds = [true, true, true]} : tensor<1x2x3xf32>, vector<1x2x3xf32>
// CHECK:           %[[RES:.*]] = vector.transfer_write %[[READ]], %[[WRITE]]{{.*}} {in_bounds = [true, true, true]} : vector<1x2x3xf32>, tensor<9x8x7x1x2x3xf32>
// CHECK:           return %[[RES]] : tensor<9x8x7x1x2x3xf32>
func.func @insert_static_slice_non_zero_pad(%arg0: tensor<1x2x3xf32>, %pad : f32) -> tensor<9x8x7x1x2x3xf32> {
  %init = tensor.empty() : tensor<9x8x7x1x2x3xf32>
  %fill = linalg.fill ins(%pad : f32) outs(%init : tensor<9x8x7x1x2x3xf32>) -> tensor<9x8x7x1x2x3xf32>
  %res = tensor.insert_slice %arg0 into %fill[0, 0, 0, 0, 0, 0] [1, 1, 1, 1, 2, 3][1, 1, 1, 1, 1, 1] : tensor<1x2x3xf32> into tensor<9x8x7x1x2x3xf32>
  return %res : tensor<9x8x7x1x2x3xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.insert_slice"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Same as above, but the source type has is dynamically shaped. This means
// that the pad value is now required and the vector dim corresponding to the
// dynamic shape has to be inferred from the shape of the destination tensor.

// CHECK-LABEL:   func.func @insert_dynamic_slice_non_zero_pad(
// CHECK-SAME:      %[[ARG_0:.*]]: tensor<1x?x3xf32>,
// CHECK-SAME:      %[[PAD:.*]]: f32,
// CHECK-SAME:      %[[SIZE:.*]]: index) -> tensor<9x8x7x1x2x3xf32> {
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<9x8x7x1x2x3xf32>
// CHECK:           %[[BC:.*]] = vector.broadcast %[[PAD]] : f32 to vector<9x8x7x1x2x3xf32>
// CHECK:           %[[WRITE:.*]] = vector.transfer_write %[[BC]], %[[EMPTY]]{{.*}} {in_bounds = [true, true, true, true, true, true]} : vector<9x8x7x1x2x3xf32>, tensor<9x8x7x1x2x3xf32>
// CHECK:           %[[READ:.*]] = vector.transfer_read %[[ARG_0]]{{.*}}, %[[PAD]] {in_bounds = [true, false, true]} : tensor<1x?x3xf32>, vector<1x2x3xf32>
// CHECK:           %[[RES:.*]] = vector.transfer_write %[[READ]], %[[WRITE]]{{.*}} {in_bounds = [true, true, true]} : vector<1x2x3xf32>, tensor<9x8x7x1x2x3xf32>
// CHECK:           return %[[RES]] : tensor<9x8x7x1x2x3xf32>
func.func @insert_dynamic_slice_non_zero_pad(%arg0: tensor<1x?x3xf32>, %pad : f32, %size: index) -> tensor<9x8x7x1x2x3xf32> {
  %init = tensor.empty() : tensor<9x8x7x1x2x3xf32>
  %fill = linalg.fill ins(%pad : f32) outs(%init : tensor<9x8x7x1x2x3xf32>) -> tensor<9x8x7x1x2x3xf32>
  %res = tensor.insert_slice %arg0 into %fill[0, 0, 0, 0, 0, 0] [1, 1, 1, 1, %size, 3][1, 1, 1, 1, 1, 1] : tensor<1x?x3xf32> into tensor<9x8x7x1x2x3xf32>
  return %res : tensor<9x8x7x1x2x3xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.insert_slice"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
