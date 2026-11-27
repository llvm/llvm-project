// RUN: mlir-opt %s -transform-interpreter -split-input-file | FileCheck %s

///----------------------------------------------------------------------------------------
/// Tests for linalg.reduce
///----------------------------------------------------------------------------------------

func.func @vectorize_dynamic_reduction_scalable_1d(%arg0: tensor<?xf32>,
                                                   %arg1: tensor<f32>) -> tensor<f32> {

  %0 = linalg.reduce ins(%arg0 : tensor<?xf32>) outs(%arg1 : tensor<f32>) dimensions = [0]
  (%in: f32, %init: f32) {
    %0 = arith.addf %in, %init : f32
    linalg.yield %0 : f32
  }
  return %0 : tensor<f32>
}

// CHECK-LABEL:  func.func @vectorize_dynamic_reduction_scalable_1d(
// CHECK-SAME:     %[[ARG_0:.*]]: tensor<?xf32>, %[[ARG_1:.*]]: tensor<f32>) -> tensor<f32> {
// CHECK:          %[[C0_IDX:.*]] = arith.constant 0 : index
// CHECK:          %[[DIM_A0_0:.*]] = tensor.dim %[[ARG_0]], %[[C0_IDX]] : tensor<?xf32>
// CHECK:          %[[C0_IDX:.*]] = arith.constant 0 : index
// CHECK:          %[[PV:.*]] = ub.poison : f32
// CHECK:          %[[MASK:.*]] = vector.create_mask %[[DIM_A0_0]] : vector<[4]xi1>
// CHECK:          %[[VEC_RD_0:.*]] = vector.mask %[[MASK]] { vector.transfer_read %[[ARG_0]][%[[C0_IDX]]], %[[PV]] {in_bounds = [true]} : tensor<?xf32>, vector<[4]xf32> } : vector<[4]xi1> -> vector<[4]xf32>
// CHECK:          %[[PV:.*]] = ub.poison : f32
// CHECK:          %[[VEC_RD_1:.*]] = vector.transfer_read %[[ARG_1]][], %[[PV]] : tensor<f32>, vector<f32>
// CHECK:          %[[ACC_f32:.*]] = vector.extract %[[VEC_RD_1]][] : f32 from vector<f32>
// CHECK:          %[[REDUCE:.*]] = vector.mask %[[MASK]] { vector.multi_reduction <add>, %[[VEC_RD_0]], %[[ACC_f32]] [0] : vector<[4]xf32> to f32 } : vector<[4]xi1> -> f32
// CHECK:          %[[VEC_f32:.*]] = vector.broadcast %[[REDUCE]] : f32 to vector<f32>
// CHECK:          %{{.*}} = vector.transfer_write %[[VEC_f32]], %[[ARG_1]][] : vector<f32>, tensor<f32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [[4]] : !transform.any_op
    transform.yield
  }
}

// -----

// The value combined into the accumulator is invariant across the iteration
// space, but it is still combined once per iteration, so it has to be
// broadcast and reduced like any other operand.

func.func @vectorize_reduction_with_invariant_operand(%arg0: tensor<4x8xf32>,
                                                      %arg1: tensor<4xf32>,
                                                      %arg2: f32) -> tensor<4xf32> {
  %0 = linalg.reduce ins(%arg0 : tensor<4x8xf32>) outs(%arg1 : tensor<4xf32>) dimensions = [1]
  (%in: f32, %init: f32) {
    %0 = arith.addf %init, %arg2 : f32
    linalg.yield %0 : f32
  }
  return %0 : tensor<4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL:   @vectorize_reduction_with_invariant_operand(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<4x8xf32>, %[[ARG1:.*]]: tensor<4xf32>, %[[ARG2:.*]]: f32)
// CHECK:           %[[ACC:.*]] = vector.transfer_read %[[ARG1]]{{.*}} : tensor<4xf32>, vector<4xf32>
// CHECK:           %[[BCAST:.*]] = vector.broadcast %[[ARG2]] : f32 to vector<4x8xf32>
// CHECK:           %[[RED:.*]] = vector.multi_reduction <add>, %[[BCAST]], %[[ACC]] [1] : vector<4x8xf32> to vector<4xf32>
// CHECK:           vector.transfer_write %[[RED]], %[[ARG1]]

// -----

func.func @vectorize_dynamic_reduction_with_invariant_operand(%arg0: tensor<?xf32>,
                                                              %arg1: tensor<f32>,
                                                              %arg2: f32) -> tensor<f32> {
  %0 = linalg.reduce ins(%arg0 : tensor<?xf32>) outs(%arg1 : tensor<f32>) dimensions = [0]
  (%in: f32, %init: f32) {
    %0 = arith.addf %init, %arg2 : f32
    linalg.yield %0 : f32
  }
  return %0 : tensor<f32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [8] : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL:   @vectorize_dynamic_reduction_with_invariant_operand(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<?xf32>, %[[ARG1:.*]]: tensor<f32>, %[[ARG2:.*]]: f32)
// CHECK:           %[[DIM:.*]] = tensor.dim %[[ARG0]]
// CHECK:           %[[MASK:.*]] = vector.create_mask %[[DIM]] : vector<8xi1>
// CHECK:           %[[ACC:.*]] = vector.extract %{{.*}}[] : f32 from vector<f32>
// CHECK:           %[[BCAST:.*]] = vector.broadcast %[[ARG2]] : f32 to vector<8xf32>
// CHECK:           %[[RED:.*]] = vector.mask %[[MASK]] { vector.multi_reduction <add>, %[[BCAST]], %[[ACC]] [0] : vector<8xf32> to f32 } : vector<8xi1> -> f32
// CHECK:           vector.broadcast %[[RED]] : f32 to vector<f32>
