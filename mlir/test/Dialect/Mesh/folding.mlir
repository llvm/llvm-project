// RUN: mlir-opt -test-mesh-simplifications %s | FileCheck %s

mesh.cluster @mesh0(rank = 3, dim_sizes = 4x?x2)
mesh.cluster @mesh1(rank = 2, dim_sizes = 2x3)

// CHECK-LABEL: func.func @cluster_shape_op_folding
func.func @cluster_shape_op_folding() -> (index, index) {
  // CHECK: %[[AXIS_2_SIZE:.*]] = arith.constant 2 : index
  // CHECK: %[[AXIS_1_SIZE:.*]] = mesh.cluster_shape @mesh0 axes = [1] : index
  %0:2 = mesh.cluster_shape @mesh0 axes = [2, 1] : index, index
  // CHECK: return %[[AXIS_2_SIZE]], %[[AXIS_1_SIZE]]
  return %0#0, %0#1 : index, index
}

// CHECK-LABEL: func.func @cluster_shape_op_folding_all_axes_static_mesh
func.func @cluster_shape_op_folding_all_axes_static_mesh() -> (index, index) {
  // CHECK: %[[AXIS_0_SIZE:.*]] = arith.constant 2 : index
  // CHECK: %[[AXIS_1_SIZE:.*]] = arith.constant 3 : index
  %0:2 = mesh.cluster_shape @mesh1 : index, index
  // CHECK: return %[[AXIS_0_SIZE]], %[[AXIS_1_SIZE]]
  return %0#0, %0#1 : index, index
}
