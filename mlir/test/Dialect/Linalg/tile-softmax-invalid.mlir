// RUN: mlir-opt %s -transform-interpreter -verify-diagnostics

func.func @do_not_tile_softmax_reduction_dimension(
    %input: tensor<4xf32>) -> tensor<4xf32> {
  %init = tensor.empty() : tensor<4xf32>
  // expected-error @below {{failed to tile operation}}
  // expected-error @below {{failed to generate tiling loops}}
  %result = linalg.softmax dimension(0)
      ins(%input : tensor<4xf32>)
      outs(%init : tensor<4xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %softmax = transform.structured.match
        ops{["linalg.softmax"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_for %softmax
        tile_sizes [2]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
