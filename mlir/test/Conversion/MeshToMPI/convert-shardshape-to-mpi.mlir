// RUN: mlir-opt %s --convert-mesh-to-mpi -canonicalize | FileCheck %s

module attributes { mpi.dlti = #dlti.map<"MPI:comm_world_rank" = 24> } {

  // CHECK: mesh.mesh @mesh0
  mesh.mesh @mesh0(shape = 3x4x5)
  
  // Notice: comm_world_rank/linear index 24 is multiindex [1, 0, 4] in @mesh0

  // all shards are equal
  // CHECK-LABEL: func.func @shard_shape_equal() -> (index, index, index) {
  func.func @shard_shape_equal() -> (index, index, index) {
    %sharding = mesh.sharding @mesh0 split_axes = [[0], [1], [2]] : !mesh.sharding
    %0:3 = mesh.process_multi_index on @mesh0 : index, index, index
    %c9 = arith.constant 9 : index
    %c12 = arith.constant 12 : index
    // CHECK: [[vc3:%.*]] = arith.constant 3 : index
    %1:3 = mesh.shard_shape dims = [%c9, %c12, 15] sharding = %sharding device = [%0#0, %0#1, %0#2] : index, index, index
    // CHECK: return [[vc3]], [[vc3]], [[vc3]] : index, index, index
    return %1#0, %1#1, %1#2 : index, index, index
  }

  // last shard in last dim gets an extra element
  // CHECK-LABEL: func.func @shard_shape_odd_1() -> (index, index, index) {
  func.func @shard_shape_odd_1() -> (index, index, index) {
    %sharding = mesh.sharding @mesh0 split_axes = [[0], [1], [2]] : !mesh.sharding
    %0:3 = mesh.process_multi_index on @mesh0 : index, index, index
    %c9 = arith.constant 9 : index
    %c12 = arith.constant 12 : index
    // CHECK-DAG: [[vc3:%.*]] = arith.constant 3 : index
    // CHECK-DAG: [[vc4:%.*]] = arith.constant 4 : index
    %1:3 = mesh.shard_shape dims = [%c9, %c12, 16] sharding = %sharding device = [%0#0, %0#1, %0#2] : index, index, index
    // CHECK: return [[vc3]], [[vc3]], [[vc4]] : index, index, index
    return %1#0, %1#1, %1#2 : index, index, index
  }

  // all except first shard in second dim get an extra element
  // CHECK-LABEL: func.func @shard_shape_odd_2() -> (index, index, index) {
  func.func @shard_shape_odd_2() -> (index, index, index) {
    %sharding = mesh.sharding @mesh0 split_axes = [[0], [1], [2]] : !mesh.sharding
    %0:3 = mesh.process_multi_index on @mesh0 : index, index, index
    %c9 = arith.constant 9 : index
    // CHECK: [[vc3:%.*]] = arith.constant 3 : index
    %1:3 = mesh.shard_shape dims = [%c9, 15, 15] sharding = %sharding device = [%0#0, %0#1, %0#2] : index, index, index
    // CHECK: return [[vc3]], [[vc3]], [[vc3]] : index, index, index
    return %1#0, %1#1, %1#2 : index, index, index
  }

  // all except first shard in first dim get an extra element
  // CHECK-LABEL: func.func @shard_shape_odd_3() -> (index, index, index) {
  func.func @shard_shape_odd_3() -> (index, index, index) {
    %sharding = mesh.sharding @mesh0 split_axes = [[0], [1], [2]] : !mesh.sharding
    %0:3 = mesh.process_multi_index on @mesh0 : index, index, index
    // CHECK-DAG: [[vc3:%.*]] = arith.constant 3 : index
    // CHECK-DAG: [[vc4:%.*]] = arith.constant 4 : index
    %1:3 = mesh.shard_shape dims = [11, 12, 15] sharding = %sharding device = [%0#0, %0#1, %0#2] : index, index, index
    // CHECK: return [[vc4]], [[vc3]], [[vc3]] : index, index, index
    return %1#0, %1#1, %1#2 : index, index, index
  }

  // extract from sharded_dims_offsets
  // CHECK-LABEL: func.func @shard_shape_sharded_dims_offs() -> (index, index, index) {
  func.func @shard_shape_sharded_dims_offs() -> (index, index, index) {
    %sharding = mesh.sharding @mesh0 split_axes = [[0], [1], [2]]
        sharded_dims_offsets = [0, 1, 4, 9, 0, 2, 6, 12, 12, 0, 3, 6, 9, 12, 15]: !mesh.sharding
    %0:3 = mesh.process_multi_index on @mesh0 : index, index, index
    %c9 = arith.constant 9 : index
    %c12 = arith.constant 12 : index
    // CHECK: [[vc3:%.*]] = arith.constant 3 : index
    // CHECK: [[vc2:%.*]] = arith.constant 2 : index
    %1:3 = mesh.shard_shape dims = [%c9, %c12, 15] sharding = %sharding device = [%0#0, %0#1, %0#2] : index, index, index
    // CHECK: return [[vc3]], [[vc2]], [[vc3]] : index, index, index
    return %1#0, %1#1, %1#2 : index, index, index
  }
}