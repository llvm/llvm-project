// RUN: mlir-opt --canonicalize %s | FileCheck %s

mesh.cluster @mesh0(rank = 2, dim_sizes = [2, 4])


// CHECK-LABEL: func @all_reduce_empty_mesh_axes_and_default_reduction
func.func @all_reduce_empty_mesh_axes_and_default_reduction(
    %arg0 : tensor<4xf32>) -> tensor<4xf64> {
  %0 = mesh.all_reduce %arg0 {
    mesh = @mesh0,
// CHECK-NOT: mesh_axes
    mesh_axes = array<i16>,
// CHECK-NOT: reduction
    reduction = #mesh.partial<sum>
    } : tensor<4xf32> -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

// CHECK-LABEL: func @all_gather_empty_mesh_axes
func.func @all_gather_empty_mesh_axes(
    %arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = mesh.all_gather %arg0 {
    gather_axis = 0 : index,
    mesh = @mesh0,
// CHECK-NOT: mesh_axes
    mesh_axes = array<i16>
    } : tensor<4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @reduce_scatter_empty_mesh_axes_and_default_reduction
func.func @reduce_scatter_empty_mesh_axes_and_default_reduction(
    %arg0 : tensor<4xf32>) -> tensor<4xf64> {
  %0 = mesh.reduce_scatter %arg0 {
    mesh = @mesh0,
// CHECK-NOT: mesh_axes
    mesh_axes = array<i16>,
// CHECK-NOT: reduction
    reduction = #mesh.partial<sum>,
    scatter_axis = 0
    } : tensor<4xf32> -> tensor<4xf64>
  return %0 : tensor<4xf64>
}