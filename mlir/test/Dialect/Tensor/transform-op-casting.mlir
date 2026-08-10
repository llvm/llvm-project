// RUN: mlir-opt %s --transform-interpreter -allow-unregistered-dialect --split-input-file | FileCheck %s

func.func @cast_to_dynamic(%arg0: tensor<10x13xf32>, %arg1: tensor<3x13xf32>) -> tensor<13x13xf32> {
  %0 = tensor.concat dim(0) %arg0, %arg1 : (tensor<10x13xf32>, tensor<3x13xf32>) -> tensor<13x13xf32>
  func.return %0 : tensor<13x13xf32>
}

func.func private @concat_replacement(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %funcs = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %f:2 = transform.split_handle %funcs : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %concat = transform.structured.match ops{["tensor.concat"]} in %f#0 : (!transform.any_op) -> !transform.any_op
    %ins = transform.get_operand %concat[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %concat[all] : (!transform.any_op) -> !transform.any_value
    transform.func.cast_and_call %f#1(%ins) -> %out before %concat {
      transform.type_conversion.tensor.cast_shape_dynamic_dims
    } : (!transform.any_op, !transform.any_value,
         !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.apply_dce to %f#0 : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func.func @cast_to_dynamic
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<10x13xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: tensor<3x13xf32>
//   CHECK-DAG:   %[[CAST0:.+]] = tensor.cast %[[ARG0]] : tensor<10x13xf32> to tensor<?x?xf32>
//   CHECK-DAG:   %[[CAST1:.+]] = tensor.cast %[[ARG1]] : tensor<3x13xf32> to tensor<?x?xf32>
//       CHECK:   %[[CALL:.+]] = call @concat_replacement(%[[CAST0]], %[[CAST1]])
//       CHECK:   %[[CAST_RES:.+]] = tensor.cast %[[CALL]] : tensor<?x?xf32> to tensor<13x13xf32>
//       CHECK:   return %[[CAST_RES]] : tensor<13x13xf32>

// -----

func.func @cast_to_static(%arg0: tensor<?x?xf32>) -> tensor<?xf32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<?x?xf32> into tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

func.func private @collapse_replacement(%arg0: tensor<4x5xf32>) -> tensor<20xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %funcs = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %f:2 = transform.split_handle %funcs : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %concat = transform.structured.match ops{["tensor.collapse_shape"]} in %f#0 : (!transform.any_op) -> !transform.any_op
    %ins = transform.get_operand %concat[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %concat[all] : (!transform.any_op) -> !transform.any_value
    transform.func.cast_and_call %f#1(%ins) -> %out before %concat {
      transform.type_conversion.tensor.cast_shape_dynamic_dims ignore_dynamic_info
    } : (!transform.any_op, !transform.any_value,
         !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.apply_dce to %f#0 : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func.func @cast_to_static
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//   CHECK-DAG:   %[[CAST_IN:.+]] = tensor.cast %[[ARG0]] : tensor<?x?xf32> to tensor<4x5xf32>
//       CHECK:   %[[CALL:.+]] = call @collapse_replacement(%[[CAST_IN]])
//       CHECK:   %[[CAST_RES:.+]] = tensor.cast %[[CALL]] : tensor<20xf32> to tensor<?xf32>
//       CHECK:   return %[[CAST_RES]] : tensor<?xf32>
