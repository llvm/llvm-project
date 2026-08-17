// RUN: mlir-opt -split-input-file -transform-interpreter -cse --mlir-print-local-scope %s | FileCheck %s

/// CHECK-LABEL: @gather_single_gather_dim
func.func @gather_single_gather_dim(%arg0: tensor<2x2x2x2xf32>, %arg1: tensor<2x3x1xindex>) -> tensor<2x3x2x2x2xf32> {
    /// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<2x3x2x2x2xf32>
    /// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG1:.*]] : tensor<2x3x1xindex>) outs(%[[EMPTY:.*]] : tensor<2x3x2x2x2xf32>)
    %1 = tensor.gather %arg0[%arg1] gather_dims([1]) : (tensor<2x2x2x2xf32>, tensor<2x3x1xindex>) -> tensor<2x3x2x2x2xf32>
  return %1 : tensor<2x3x2x2x2xf32>
}

/// CHECK-LABEL: @gather_single_gather_dim_no_shrink
func.func @gather_single_gather_dim_no_shrink(%arg0: tensor<2x2x2x2xf32>, %arg1: tensor<2x3x1xindex>) -> tensor<2x3x2x1x2x2xf32> {
    /// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<2x3x2x1x2x2xf32>
    /// CHECK: linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, 0)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG1:.*]] : tensor<2x3x1xindex>) outs(%[[EMPTY1:.*]] : tensor<2x3x2x1x2x2xf32>)
    %1 = tensor.gather %arg0[%arg1] gather_dims([1]) : (tensor<2x2x2x2xf32>, tensor<2x3x1xindex>) -> tensor<2x3x2x1x2x2xf32>
  return %1 : tensor<2x3x2x1x2x2xf32>
}

/// CHECK-LABEL: @gather_multiple_gather_dim
func.func @gather_multiple_gather_dim(%arg0: tensor<2x2x2x2xf32>, %arg1: tensor<2x3x2xindex>) -> tensor<2x3x2x2xf32> {
    // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<2x3x2x2xf32>
    /// CHECK: %[[EXTRACTSLICE1:.*]] = tensor.extract_slice %[[ARG1:.*]][0, 0, 0] [2, 3, 1] [1, 1, 1] : tensor<2x3x2xindex> to tensor<2x3x1xindex>
    /// CHECK: %[[EXTRACTSLICE2:.*]] = tensor.extract_slice %[[ARG1:.*]][0, 0, 1] [2, 3, 1] [1, 1, 1] : tensor<2x3x2xindex> to tensor<2x3x1xindex>
      /// CHECK: linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[EXTRACTSLICE1:.*]], %[[EXTRACTSLICE2:.*]] : tensor<2x3x1xindex>, tensor<2x3x1xindex>) outs(%[[EMPTY:.*]] : tensor<2x3x2x2xf32>)
    %1 = tensor.gather %arg0[%arg1] gather_dims([1, 2]) : (tensor<2x2x2x2xf32>, tensor<2x3x2xindex>) -> tensor<2x3x2x2xf32>
  return %1 : tensor<2x3x2x2xf32>
}

/// CHECK-LABEL: @gather_multiple_gather_dim_no_shrink
func.func @gather_multiple_gather_dim_no_shrink(%arg0: tensor<2x2x2x2xf32>, %arg1: tensor<2x3x2xindex>) -> tensor<2x3x2x1x1x2xf32> {
    %1 = tensor.gather %arg0[%arg1] gather_dims([1, 2]) : (tensor<2x2x2x2xf32>, tensor<2x3x2xindex>) -> tensor<2x3x2x1x1x2xf32>
  return %1 : tensor<2x3x2x1x1x2xf32>
}

/// CHECK-LABEL: @gather_single_gather_dim_dynamic
func.func @gather_single_gather_dim_dynamic(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<2x3x1xindex>) -> tensor<2x3x?x?x?xf32> {
    /// CHECK: %[[DIM1:.*]] = tensor.dim
    /// CHECK: %[[DIM2:.*]] = tensor.dim
    /// CHECK: %[[DIM3:.*]] = tensor.dim
    /// CHECK: %[[EMPTY:.*]] = tensor.empty(%[[DIM1:.*]], %[[DIM2:.*]], %[[DIM3:.*]]) : tensor<2x3x?x?x?xf32>
    /// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG0:.*]] : tensor<2x3x1xindex>) outs(%[[EMPTY:.*]] : tensor<2x3x?x?x?xf32>)
    %1 = tensor.gather %arg0[%arg1] gather_dims([1]) : (tensor<?x?x?x?xf32>, tensor<2x3x1xindex>) -> tensor<2x3x?x?x?xf32>
  return %1 : tensor<2x3x?x?x?xf32>
}

/// CHECK-LABEL: @gather_multiple_gather_dim_no_shrink_dynamic
func.func @gather_multiple_gather_dim_no_shrink_dynamic(%arg0: tensor<2x2x2x2xf32>, %arg1: tensor<?x?x2xindex>) -> tensor<?x?x2x1x1x2xf32> {
    /// CHECK: %[[DIM1:.*]] = tensor.dim
    /// CHECK: %[[DIM2:.*]] = tensor.dim
    /// CHECK: %[[EMPTY:.*]] = tensor.empty(%[[DIM1:.*]], %[[DIM2:.*]]) : tensor<?x?x2x1x1x2xf32>
    /// CHECK: %[[EXTRACTSLICE1:.*]] = tensor.extract_slice %[[ARG1:.*]][0, 0, 0] [%[[DIM1:.*]], %[[DIM2:.*]], 1] [1, 1, 1] : tensor<?x?x2xindex> to tensor<?x?x1xindex>
    /// CHECK: %[[EXTRACTSLICE2:.*]] = tensor.extract_slice %[[ARG1:.*]][0, 0, 1] [%[[DIM1:.*]], %[[DIM2:.*]], 1] [1, 1, 1] : tensor<?x?x2xindex> to tensor<?x?x1xindex>
    /// CHECK: linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, 0)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, 0)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[EXTRACTSLICE1:.*]], %[[EXTRACTSLICE2:.*]] : tensor<?x?x1xindex>, tensor<?x?x1xindex>) outs(%[[EMPTY:.*]] : tensor<?x?x2x1x1x2xf32>)
    %1 = tensor.gather %arg0[%arg1] gather_dims([1, 2]) : (tensor<2x2x2x2xf32>, tensor<?x?x2xindex>) -> tensor<?x?x2x1x1x2xf32>
  return %1 : tensor<?x?x2x1x1x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.tensor.decompose_gather
    } : !transform.op<"func.func">
    transform.yield
  }
}
