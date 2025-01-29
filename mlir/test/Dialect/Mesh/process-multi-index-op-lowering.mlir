// RUN: mlir-opt -test-mesh-process-multi-index-op-lowering %s | FileCheck %s

mesh.mesh @mesh2d(shape = ?x?)

// CHECK-LABEL: func.func @multi_index_2d_mesh
func.func @multi_index_2d_mesh() -> (index, index) {
  // CHECK: %[[LINEAR_IDX:.*]] = mesh.process_linear_index on @mesh2d : index
  // CHECK: %[[MESH_SHAPE:.*]]:2 = mesh.mesh_shape @mesh2d : index, index
  // CHECK: %[[MULTI_IDX:.*]]:2 = affine.delinearize_index %[[LINEAR_IDX]] into (%[[MESH_SHAPE]]#0, %[[MESH_SHAPE]]#1) : index, index
  %0:2 = mesh.process_multi_index on @mesh2d : index, index
  // CHECK: return %[[MULTI_IDX]]#0, %[[MULTI_IDX]]#1 : index, index
  return %0#0, %0#1 : index, index
}

// CHECK-LABEL: func.func @multi_index_2d_mesh_single_inner_axis
func.func @multi_index_2d_mesh_single_inner_axis() -> index {
  // CHECK: %[[LINEAR_IDX:.*]] = mesh.process_linear_index on @mesh2d : index
  // CHECK: %[[MESH_SHAPE:.*]]:2 = mesh.mesh_shape @mesh2d : index, index
  // CHECK: %[[MULTI_IDX:.*]]:2 = affine.delinearize_index %[[LINEAR_IDX]] into (%[[MESH_SHAPE]]#0, %[[MESH_SHAPE]]#1) : index, index
  %0 = mesh.process_multi_index on @mesh2d axes = [0] : index
  // CHECK: return %[[MULTI_IDX]]#0 : index
  return %0 : index
}
