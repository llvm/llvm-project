// RUN: mlir-opt -transform-interpreter -cse -split-input-file %s | FileCheck %s

// 2D tiling of dynamic 2D pad tensor op.
func.func @dynamic_2d_pad_tensor(%input_tensor: tensor<?x?xf32>,
                         %pad_value: f32) -> tensor<?x?xf32> {
  %0 = tensor.pad %input_tensor low[3, 4] high[5, 3] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad_value : f32
    } : tensor<?x?xf32> to tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %pad = transform.structured.match ops{["tensor.pad"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b, %c = transform.structured.tile_using_for %pad [2, 3]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

//  CHECK-DAG:  #[[MAP0:.+]] = affine_map<()[s0] -> (s0 + 8)>
//  CHECK-DAG:  #[[MAP1:.+]] = affine_map<()[s0] -> (s0 + 7)>
//       CHECK: func @dynamic_2d_pad_tensor(
//  CHECK-SAME:     %[[IN:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[DIM_IN0:.+]] = tensor.dim %[[IN]], %[[C0]]
//   CHECK-DAG:   %[[DIM0:.+]] = affine.apply #[[MAP0]]()[%[[DIM_IN0]]]
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[DIM_IN1:.+]] = tensor.dim %[[IN]], %[[C1]]
//   CHECK-DAG:   %[[DIM1:.+]] = affine.apply #[[MAP1]]()[%[[DIM_IN1]]]
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//       CHECK:   %[[RESULT:[a-zA-Z0-9]+]] = scf.for %[[IV0:[a-zA-Z0-9]+]] = %[[C0]] to %[[DIM0]] step %[[C2]]
//       CHECK:     %[[C3:.+]] = arith.constant 3 : index
//       CHECK:     scf.for {{.*}} = %[[C0]] to %[[DIM1]] step %[[C3]] iter_args(%[[INNER_OUT:.*]] =
//       CHECK:       %[[SWAP_RESULT:.*]] = scf.if
//       CHECK:         tensor.generate
//       CHECK:       else
//       CHECK:         %[[SLICE:.*]] = tensor.extract_slice %[[IN]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1]
//       CHECK:         %[[PAD:.*]] = tensor.pad %[[SLICE]]
//       CHECK:       tensor.insert_slice %[[SWAP_RESULT]] into %[[INNER_OUT]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1]
//       CHECK:   return %[[RESULT]]

// -----

func.func @dynamic_2d_pad_tensor_inner_tiling(%input_tensor: tensor<?x?xf32>,
                         %pad_value: f32) -> tensor<?x?xf32> {
  %0 = tensor.pad %input_tensor low[3, 4] high[5, 3] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad_value : f32
    } : tensor<?x?xf32> to tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %pad = transform.structured.match ops{["tensor.pad"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.structured.tile_using_for %pad [0, 3]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//   CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0] -> (s0 + 8)>
//   CHECK-DAG: #[[MAP1:.*]] = affine_map<()[s0] -> (s0 + 7)>
//       CHECK: func @dynamic_2d_pad_tensor_inner_tiling(
//  CHECK-SAME:     %[[IN:.*]]: tensor<?x?xf32>
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[DIM_IN0:.*]] = tensor.dim %[[IN]], %[[C0]]
//   CHECK-DAG:   %[[DIM0:.*]] = affine.apply #[[MAP0]]()[%[[DIM_IN0]]]
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[DIM_IN1:.*]] = tensor.dim %[[IN]], %[[C1]]
//   CHECK-DAG:   %[[DIM1:.*]] = affine.apply #[[MAP1]]()[%[[DIM_IN1]]]
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//       CHECK:   %[[RESULT:.*]] = scf.for {{.*}} = %[[C0]] to %[[DIM1]] step %[[C3]] iter_args(%[[INNER_OUT:.*]] =
//       CHECK:     %[[SWAP_RESULT:.*]] = scf.if
//       CHECK:       tensor.generate
//       CHECK:     else
//       CHECK:       %[[SLICE:.*]] = tensor.extract_slice %[[IN]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1]
//       CHECK:       %[[PAD:.*]] = tensor.pad %[[SLICE]] low[3, %{{.*}}] high[{{.*}}, {{.*}}]
//       CHECK:     tensor.insert_slice %[[SWAP_RESULT]] into %[[INNER_OUT]][0, {{.*}}] [%[[DIM0]], {{.*}}] [1, 1]
//       CHECK:   return %[[RESULT]]

// -----

func.func @static_pad_tensor(%input_tensor: tensor<7x9xf32>,
                        %pad_value: f32) -> tensor<15x16xf32> {
  %0 = tensor.pad %input_tensor low[3, 4] high[5, 3] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad_value : f32
    }  : tensor<7x9xf32> to tensor<15x16xf32>
  return %0 : tensor<15x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %pad = transform.structured.match ops{["tensor.pad"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b, %c = transform.structured.tile_using_for %pad [2, 3]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-LABEL: func @static_pad_tensor(
//  CHECK-SAME:     %[[IN:.*]]: tensor<7x9xf32>
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C15:.*]] = arith.constant 15 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//       CHECK:   %[[RESULT:.*]] = scf.for {{.*}} = %[[C0]] to %[[C15]] step %[[C2]]
//   CHECK-DAG:     %[[C16:.*]] = arith.constant 16 : index
//   CHECK-DAG:     %[[C3:.*]] = arith.constant 3 : index
//       CHECK:     scf.for {{.*}} = %[[C0]] to %[[C16]] step %[[C3]] iter_args(%[[INNER_OUT:.*]] =
//       CHECK:       %[[SWAP_RESULT:.*]] = scf.if
//       CHECK:         tensor.generate
//       CHECK:       else
//       CHECK:         %[[SLICE:.*]] = tensor.extract_slice %[[IN]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1]
//       CHECK:         %[[PAD:.*]] = tensor.pad %[[SLICE]]
//       CHECK:       tensor.insert_slice %[[SWAP_RESULT]] into %[[INNER_OUT]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1]
//       CHECK:   return %[[RESULT]]

// -----

func.func @static_pad_tensor_inner_tiling(%input_tensor: tensor<7x9xf32>,
                        %pad_value: f32) -> tensor<15x16xf32> {
  %0 = tensor.pad %input_tensor low[3, 4] high[5, 3] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad_value : f32
    }  : tensor<7x9xf32> to tensor<15x16xf32>
  return %0 : tensor<15x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %pad = transform.structured.match ops{["tensor.pad"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.structured.tile_using_for %pad [0, 3]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-LABEL: func @static_pad_tensor_inner_tiling(
//  CHECK-SAME:     %[[IN:.*]]: tensor<7x9xf32>
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : index
//       CHECK:   %[[RESULT:.*]] = scf.for {{.*}} = %[[C0]] to %[[C16]] step %[[C3]] iter_args(%[[INNER_OUT:.*]] =
//       CHECK:     %[[SWAP_RESULT:.*]] = scf.if
//       CHECK:       tensor.generate
//       CHECK:     else
//       CHECK:       %[[SLICE:.*]] = tensor.extract_slice %[[IN]][0, {{.*}}] [7, {{.*}}] [1, 1]
//       CHECK:       %[[PAD:.*]] = tensor.pad %[[SLICE]] low[3, %{{.*}}] high[5, {{.*}}]
//       CHECK:     tensor.insert_slice %[[SWAP_RESULT]] into %[[INNER_OUT]][0, {{.*}}] [15, {{.*}}] [1, 1]
//       CHECK:   return %[[RESULT]]

/// Rest of the tests only check that they dont fail.

// -----

func.func @dynamic_2d_pad_tensor_outer_tiling(%input_tensor: tensor<?x?xf32>,
                         %pad_value: f32) -> tensor<?x?xf32> {
  %0 = tensor.pad %input_tensor low[3, 4] high[5, 3] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad_value : f32
    } : tensor<?x?xf32> to tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %pad = transform.structured.match ops{["tensor.pad"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b, %c = transform.structured.tile_using_for %pad [2, 3]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-LABEL: func @dynamic_2d_pad_tensor_outer_tiling

// -----

func.func @static_pad_tensor_outer_tiling(%input_tensor: tensor<7x9xf32>,
                        %pad_value: f32) -> tensor<15x16xf32> {
  %0 = tensor.pad %input_tensor low[3, 4] high[5, 3] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad_value : f32
    }  : tensor<7x9xf32> to tensor<15x16xf32>
  return %0 : tensor<15x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %pad = transform.structured.match ops{["tensor.pad"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.structured.tile_using_for %pad [0, 3]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-LABEL: func @static_pad_tensor_outer_tiling
