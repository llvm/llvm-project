// RUN: mlir-opt --split-input-file --test-mesh-all-slice-op-lowering --test-mesh-simplifications --cse %s | FileCheck %s

mesh.mesh @mesh_1d(shape = ?)

// CHECK-LABEL: func.func @all_slice_op_lowering_of_dynamic_1d_tensor_on_dynamic_1d_mesh
func.func @all_slice_op_lowering_of_dynamic_1d_tensor_on_dynamic_1d_mesh(
  // CHECK: %[[ARG:.*]]: tensor<?xf16>
  %arg0: tensor<?xf16>
// CHECK-SAME: -> tensor<?xf16> {
) -> tensor<?xf16> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[PROC_IDX:.*]] = mesh.process_multi_index on @mesh_1d axes = [0] : index
  // CHECK-DAG: %[[MESH_SIZE:.*]] = mesh.mesh_shape @mesh_1d axes = [0] : index
  // CHECK: %[[TENSOR_AXIS_SIZE:.*]] = tensor.dim %[[ARG]], %c0 : tensor<?xf16>
  // CHECK: %[[AXIS_SIZE_CHECK_REMINDER:.*]] = arith.remui %[[TENSOR_AXIS_SIZE]], %[[MESH_SIZE]] : index
  // CHECK: %[[AXIS_SIZE_CHECK:.*]] = arith.cmpi eq, %[[AXIS_SIZE_CHECK_REMINDER]], %[[C0]] : index
  // CHECK: cf.assert %[[AXIS_SIZE_CHECK]]
  // CHECK: %[[RESULT_AXIS_SIZE:.*]] = arith.divui %[[TENSOR_AXIS_SIZE]], %[[MESH_SIZE]] : index
  // CHECK: %[[SLICE_OFFSET:.*]] = arith.muli %[[PROC_IDX]], %[[RESULT_AXIS_SIZE]] : index
  // CHECK: %[[RESULT:.*]] = tensor.extract_slice %[[ARG]][%[[SLICE_OFFSET]]] [%[[RESULT_AXIS_SIZE]]] [1] : tensor<?xf16> to tensor<?xf16>
  %0 = mesh.all_slice %arg0 on @mesh_1d mesh_axes = [0] slice_axis = 0 : tensor<?xf16> -> tensor<?xf16>
  // CHECK: return %[[RESULT]] : tensor<?xf16>
  return %0 : tensor<?xf16>
}

// -----

mesh.mesh @mesh_1d(shape = 2)

// CHECK-LABEL: func.func @all_slice_op_lowering_of_static_1d_tensor_on_static_1d_mesh
func.func @all_slice_op_lowering_of_static_1d_tensor_on_static_1d_mesh(
  // CHECK: %[[ARG:.*]]: tensor<2xf16>
  %arg0: tensor<2xf16>
// CHECK-SAME: -> tensor<1xf16> {
) -> tensor<1xf16> {
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[PROC_IDX:.*]] = mesh.process_multi_index on @mesh_1d axes = [0] : index
  // CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[ARG]][%[[PROC_IDX]]] [%[[C1]]] [1] : tensor<2xf16> to tensor<?xf16>
  // CHECK: %[[RESULT:.*]] = tensor.cast %[[SLICE]] : tensor<?xf16> to tensor<1xf16>
  %0 = mesh.all_slice %arg0 on @mesh_1d mesh_axes = [0] slice_axis = 0 : tensor<2xf16> -> tensor<1xf16>
  // CHECK: return %[[RESULT]] : tensor<1xf16>
  return %0 : tensor<1xf16>
}

// -----

// CHECK: #map = affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>

mesh.mesh @mesh_4d(shape = ?x?x?x?)

// CHECK-LABEL: func.func @all_slice_op_lowering_of_dynamic_2d_tensor_on_dynamic_4d_mesh
func.func @all_slice_op_lowering_of_dynamic_2d_tensor_on_dynamic_4d_mesh(
  // CHECK: %[[ARG:.*]]: tensor<?x?xf16>
  %arg0 : tensor<?x?xf16>
// CHECK-SAME: -> tensor<?x?xf16> {
) -> tensor<?x?xf16> {
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[IN_GROUP_PROC_MULTI_IDX:.*]]:2 = mesh.process_multi_index on @mesh_4d axes = [3, 1] : index, index
  // CHECK-DAG: %[[PROC_GROUP_SHAPE:.*]]:2 = mesh.mesh_shape @mesh_4d axes = [3, 1] : index, index
  // CHECK: %[[PROC_GROUP_SIZE:.*]] = arith.muli %[[PROC_GROUP_SHAPE]]#0, %[[PROC_GROUP_SHAPE]]#1 : index
  // CHECK: %[[SCATTER_AXIS_SIZE:.*]] = tensor.dim %[[ARG]], %[[C1]] : tensor<?x?xf16>
  // CHECK: %[[AXIS_SIZE_CHECK_REMINDER:.*]] = arith.remui %[[SCATTER_AXIS_SIZE]], %[[PROC_GROUP_SIZE]] : index
  // CHECK: %[[AXIS_SIZE_CHECK:.*]] = arith.cmpi eq, %[[AXIS_SIZE_CHECK_REMINDER]], %[[C0]] : index
  // CHECK: cf.assert %[[AXIS_SIZE_CHECK]]
  // CHECK: %[[RESULT_SCATTER_AXIS_SIZE:.*]] = arith.divui %[[SCATTER_AXIS_SIZE]], %[[PROC_GROUP_SIZE]] : index
  // CHECK: %[[PROC_IN_GROUP_LINEAR_IDX:.*]] = affine.apply #map()[%[[IN_GROUP_PROC_MULTI_IDX]]#0, %[[PROC_GROUP_SHAPE]]#1, %[[IN_GROUP_PROC_MULTI_IDX]]#1]
  // CHECK: %[[AXIS_0_SIZE:.*]] = tensor.dim %[[ARG]], %[[C0]] : tensor<?x?xf16>
  // CHECK: %[[SCATTER_AXIS_OFFSET:.*]] = arith.muli %[[PROC_IN_GROUP_LINEAR_IDX]], %[[RESULT_SCATTER_AXIS_SIZE]] : index
  // CHECK: %[[RESULT:.*]] = tensor.extract_slice %[[ARG]][0, %[[SCATTER_AXIS_OFFSET]]] [%[[AXIS_0_SIZE]], %[[RESULT_SCATTER_AXIS_SIZE]]] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
  %0 = mesh.all_slice %arg0 on @mesh_4d mesh_axes = [3, 1] slice_axis = 1 : tensor<?x?xf16> -> tensor<?x?xf16>
  // CHECK: return %[[RESULT]] : tensor<?x?xf16>
  return %0 : tensor<?x?xf16>
}
