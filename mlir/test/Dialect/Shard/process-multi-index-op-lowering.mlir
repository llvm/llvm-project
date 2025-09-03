// RUN: mlir-opt -test-grid-process-multi-index-op-lowering %s | FileCheck %s

shard.grid @grid2d(shape = ?x?)

// CHECK-LABEL: func.func @multi_index_2d_grid
func.func @multi_index_2d_grid() -> (index, index) {
  // CHECK: %[[LINEAR_IDX:.*]] = shard.process_linear_index on @grid2d : index
  // CHECK: %[[SHARD_SHAPE:.*]]:2 = shard.grid_shape @grid2d : index, index
  // CHECK: %[[MULTI_IDX:.*]]:2 = affine.delinearize_index %[[LINEAR_IDX]] into (%[[SHARD_SHAPE]]#0, %[[SHARD_SHAPE]]#1) : index, index
  %0:2 = shard.process_multi_index on @grid2d : index, index
  // CHECK: return %[[MULTI_IDX]]#0, %[[MULTI_IDX]]#1 : index, index
  return %0#0, %0#1 : index, index
}

// CHECK-LABEL: func.func @multi_index_2d_grid_single_inner_axis
func.func @multi_index_2d_grid_single_inner_axis() -> index {
  // CHECK: %[[LINEAR_IDX:.*]] = shard.process_linear_index on @grid2d : index
  // CHECK: %[[SHARD_SHAPE:.*]]:2 = shard.grid_shape @grid2d : index, index
  // CHECK: %[[MULTI_IDX:.*]]:2 = affine.delinearize_index %[[LINEAR_IDX]] into (%[[SHARD_SHAPE]]#0, %[[SHARD_SHAPE]]#1) : index, index
  %0 = shard.process_multi_index on @grid2d axes = [0] : index
  // CHECK: return %[[MULTI_IDX]]#0 : index
  return %0 : index
}
