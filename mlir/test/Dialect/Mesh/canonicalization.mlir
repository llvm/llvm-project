// RUN: mlir-opt --canonicalize %s | FileCheck %s

mesh.cluster @mesh0(rank = 2, dim_sizes = [2, 4])

// CHECK-LABEL: func @all_reduce_empty_mesh_axes
func.func @all_reduce_empty_mesh_axes(
// CHECK-SAME: %[[ARG:.*]]: tensor<4xf32>
    %arg0 : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT: mesh.all_reduce
  %0 = mesh.all_reduce %arg0 on @mesh0
    mesh_axes = []
    : tensor<4xf32> -> tensor<4xf32>
// CHECK: return %[[ARG]]
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @all_reduce_empty_mesh_axes_different_return_type
func.func @all_reduce_empty_mesh_axes_different_return_type(
    %arg0 : tensor<4xf32>) -> tensor<4xf64> {
// CHECK: mesh.all_reduce
  %0 = mesh.all_reduce %arg0 on @mesh0
// CHECK-NOT: mesh_axes
    mesh_axes = []
    : tensor<4xf32> -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

// CHECK-LABEL: func @all_reduce_default_reduction
func.func @all_reduce_default_reduction(
    %arg0 : tensor<4xf32>) -> tensor<4xf64> {
  %0 = mesh.all_reduce %arg0 on @mesh0
    mesh_axes = [0]
// CHECK-NOT: reduction
    reduction = <sum>
    : tensor<4xf32> -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

// CHECK-LABEL: func @all_to_all_empty_mesh_axes
func.func @all_to_all_empty_mesh_axes(
// CHECK-SAME: %[[ARG:.*]]: tensor<8xf32>
    %arg0 : tensor<8xf32>) -> tensor<8xf32> {
// CHECK-NOT: mesh.all_to_all
  %0 = mesh.all_to_all %arg0 on @mesh0
    mesh_axes = []
    split_axis = 0
    concat_axis = 0
    : tensor<8xf32> -> tensor<8xf32>
// CHECK: return %[[ARG]]
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func @all_gather_empty_mesh_axes
func.func @all_gather_empty_mesh_axes(
// CHECK-SAME: %[[ARG:.*]]: tensor<4xf32>
    %arg0 : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT: mesh.all_gather
  %0 = mesh.all_gather %arg0 on @mesh0
    mesh_axes = []
    gather_axis = 0
    : tensor<4xf32> -> tensor<4xf32>
// CHECK: return %[[ARG]]
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @reduce_scatter_empty_mesh_axes
func.func @reduce_scatter_empty_mesh_axes(
// CHECK-SAME: %[[ARG:.*]]: tensor<4xf32>
    %arg0 : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT: mesh.reduce_scatter
  %0 = mesh.reduce_scatter %arg0 on @mesh0
    mesh_axes = []
    scatter_axis = 0
    : tensor<4xf32> -> tensor<4xf32>
// CHECK: return %[[ARG]]
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @reduce_scatter_empty_mesh_axes_different_return_type
func.func @reduce_scatter_empty_mesh_axes_different_return_type(
    %arg0 : tensor<4xf32>) -> tensor<4xf64> {
// CHECK: mesh.reduce_scatter
  %0 = mesh.reduce_scatter %arg0 on @mesh0
// CHECK-NOT: mesh_axes
    mesh_axes = []
    scatter_axis = 0
    : tensor<4xf32> -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

// CHECK-LABEL: func @reduce_scatter_default_reduction
func.func @reduce_scatter_default_reduction(
    %arg0 : tensor<4xf32>) -> tensor<2xf64> {
  %0 = mesh.reduce_scatter %arg0 on @mesh0
    mesh_axes = [0]
// CHECK-NOT: reduction
    reduction = <sum>
    scatter_axis = 0
    : tensor<4xf32> -> tensor<2xf64>
  return %0 : tensor<2xf64>
}
