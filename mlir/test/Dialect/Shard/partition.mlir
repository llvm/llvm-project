// RUN: mlir-opt \
// RUN:   --pass-pipeline="builtin.module(func.func(shard-partition,test-single-fold))" \
// RUN:   %s | FileCheck %s

shard.grid @grid_1d(shape = 2)
shard.grid @grid_1d_4(shape = 4)
shard.grid @grid_2d_16(shape = 4x4)
shard.grid @grid_4d(shape = 2x3x4x5)

// CHECK-LABEL: func @return_sharding
func.func @return_sharding(
  // CHECK-SAME: [[ARG:%.*]]: tensor<1xf32>
  %arg0: tensor<2xf32>
// CHECK-SAME: ) -> (tensor<1xf32>, !shard.sharding) {
) -> (tensor<2xf32>, !shard.sharding) {
  %ssharded = shard.sharding @grid_1d split_axes = [[0]] : !shard.sharding
  %sharded = shard.shard %arg0 to %ssharded  : tensor<2xf32>
  // CHECK-NEXT: [[vsharding:%.*]] = shard.sharding @grid_1d split_axes = {{\[\[}}0]] : !shard.sharding
  %r = shard.get_sharding %sharded : tensor<2xf32> -> !shard.sharding
  %sharded_r = shard.shard %sharded to %ssharded annotate_for_users : tensor<2xf32>
  // CHECK-NEXT: return [[ARG]], [[vsharding]] : tensor<1xf32>, !shard.sharding
  return %sharded_r, %r : tensor<2xf32>, !shard.sharding
}

// CHECK-LABEL: func @full_replication
func.func @full_replication(
  // CHECK-SAME: %[[ARG:.*]]: tensor<2xi8>
  %arg0: tensor<2xi8>
// CHECK-SAME: -> tensor<2xi8> {
) -> tensor<2xi8> {
  %s0 = shard.sharding @grid_1d split_axes = [[]] : !shard.sharding
  %0 = shard.shard %arg0 to %s0  : tensor<2xi8>
  %s1 = shard.sharding @grid_1d split_axes = [[]] : !shard.sharding
  %1 = shard.shard %0 to %s1  annotate_for_users : tensor<2xi8>
  // CHECK: return %[[ARG]] : tensor<2xi8>
  return %1 : tensor<2xi8>
}

// CHECK-LABEL: func @sharding_triplet
func.func @sharding_triplet(
  // CHECK-SAME: %[[ARG:.*]]: tensor<1xf32>
  %arg0: tensor<2xf32>
// CHECK-SAME: ) -> tensor<2xf32> {
) -> tensor<2xf32> {
  // CHECK: %[[ALL_GATHER:.*]] = shard.all_gather %[[ARG]] on @grid_1d grid_axes = [0] gather_axis = 0 : tensor<1xf32> -> tensor<2xf32>
  %ssharded = shard.sharding @grid_1d split_axes = [[0]] : !shard.sharding
  %sharded = shard.shard %arg0 to %ssharded  : tensor<2xf32>
  %ssharded_0 = shard.sharding @grid_1d split_axes = [[0]] : !shard.sharding
  %sharded_0 = shard.shard %sharded to %ssharded_0  annotate_for_users : tensor<2xf32>
  %ssharded_1 = shard.sharding @grid_1d split_axes = [[]] : !shard.sharding
  %sharded_1 = shard.shard %sharded_0 to %ssharded_1 annotate_for_users : tensor<2xf32>
  // CHECK: return %[[ALL_GATHER]] : tensor<2xf32>
  return %sharded_1 : tensor<2xf32>
}

// CHECK-LABEL: func.func @unsplit_last_axes_some(
// CHECK-SAME: [[varg0:%.*]]: tensor<6x2xi8>) -> tensor<6x24xi8> {
func.func @unsplit_last_axes_some( %in2: tensor<6x48xi8>) -> tensor<6x48xi8> {
  %sharding0 = shard.sharding @grid_4d split_axes = [[], [0,1,2]] : !shard.sharding
  %sharding1 = shard.shard %in2 to %sharding0 : tensor<6x48xi8>
  %sharding2 = shard.sharding @grid_4d split_axes = [[], [0]] : !shard.sharding
  %sharding3 = shard.shard %sharding1 to %sharding2 annotate_for_users : tensor<6x48xi8>
  // CHECK: [[vall_gather:%.*]] = shard.all_gather [[varg0]] on @grid_4d grid_axes = [1, 2] gather_axis = 1 : tensor<6x2xi8> -> tensor<6x24xi8>
  // CHECK: return [[vall_gather]] : tensor<6x24xi8>
  return %sharding3 : tensor<6x48xi8>
}

// CHECK-LABEL: func.func @unsplit_last_axes_all(
// CHECK-SAME: [[varg0:%.*]]: tensor<2x48xi8>) -> tensor<48x48xi8> {
func.func @unsplit_last_axes_all(%in2: tensor<48x48xi8>) -> tensor<48x48xi8> {
  %sharding0 = shard.sharding @grid_4d split_axes = [[0,1,2]] : !shard.sharding
  %sharding1 = shard.shard %in2 to %sharding0 : tensor<48x48xi8>
  %sharding2 = shard.sharding @grid_4d split_axes = [[]] : !shard.sharding
  %sharding3 = shard.shard %sharding1 to %sharding2 annotate_for_users : tensor<48x48xi8>
  // CHECK: [[vall_gather:%.*]] = shard.all_gather [[varg0]] on @grid_4d grid_axes = [0, 1, 2] gather_axis = 0 : tensor<2x48xi8> -> tensor<48x48xi8>
  // CHECK: return [[vall_gather]] : tensor<48x48xi8>
  return %sharding3 : tensor<48x48xi8>
}

// CHECK-LABEL: func.func @unsplit_all_dims(
// CHECK-SAME: [[varg0:%.*]]: tensor<3x2x4x5xi8>) -> tensor<6x10x16x15xi8> {
func.func @unsplit_all_dims(%arg: tensor<6x10x16x15xi8>) -> tensor<6x10x16x15xi8> {
  %sharding1 = shard.sharding @grid_4d split_axes = [[0], [3], [2], [1]] : !shard.sharding
  %arg_sharded = shard.shard %arg to %sharding1 : tensor<6x10x16x15xi8>
  %sharding2 = shard.sharding @grid_4d split_axes = [[], []] : !shard.sharding
  %res_sharded = shard.shard %arg_sharded to %sharding2 annotate_for_users : tensor<6x10x16x15xi8>
  // CHECK: [[vall_gather:%.*]] = shard.all_gather [[varg0]] on @grid_4d grid_axes = [0] gather_axis = 0 : tensor<3x2x4x5xi8> -> tensor<6x2x4x5xi8>
  // CHECK: [[vall_gather_0:%.*]] = shard.all_gather [[vall_gather]] on @grid_4d grid_axes = [3] gather_axis = 1 : tensor<6x2x4x5xi8> -> tensor<6x10x4x5xi8>
  // CHECK: [[vall_gather_1:%.*]] = shard.all_gather [[vall_gather_0]] on @grid_4d grid_axes = [2] gather_axis = 2 : tensor<6x10x4x5xi8> -> tensor<6x10x16x5xi8>
  // CHECK: [[vall_gather_2:%.*]] = shard.all_gather [[vall_gather_1]] on @grid_4d grid_axes = [1] gather_axis = 3 : tensor<6x10x16x5xi8> -> tensor<6x10x16x15xi8>
  // CHECK: return [[vall_gather_2]] : tensor<6x10x16x15xi8>
  return %res_sharded : tensor<6x10x16x15xi8>
}

// CHECK-LABEL: func.func @unsplit_some_dims(
// CHECK-SAME: [[varg0:%.*]]: tensor<6x2x4x15xi8>) -> tensor<6x10x16x15xi8> {
func.func @unsplit_some_dims(%arg: tensor<6x10x16x15xi8>) -> tensor<6x10x16x15xi8> {
  %sharding1 = shard.sharding @grid_4d split_axes = [[], [3], [2], []] : !shard.sharding
  %arg_sharded = shard.shard %arg to %sharding1 : tensor<6x10x16x15xi8>
  %sharding2 = shard.sharding @grid_4d split_axes = [[]] : !shard.sharding
  %res_sharded = shard.shard %arg_sharded to %sharding2 annotate_for_users : tensor<6x10x16x15xi8>
  // CHECK: [[vall_gather:%.*]] = shard.all_gather [[varg0]] on @grid_4d grid_axes = [3] gather_axis = 1 : tensor<6x2x4x15xi8> -> tensor<6x10x4x15xi8>
  // CHECK: [[vall_gather_0:%.*]] = shard.all_gather [[vall_gather]] on @grid_4d grid_axes = [2] gather_axis = 2 : tensor<6x10x4x15xi8> -> tensor<6x10x16x15xi8>
  // CHECK: return [[vall_gather_0]] : tensor<6x10x16x15xi8>
  return %res_sharded : tensor<6x10x16x15xi8>
}

// CHECK-LABEL: func @move_split_axis
func.func @move_split_axis(
  // CHECK-SAME: %[[ARG:.*]]: tensor<1x2xi8>
  %arg0: tensor<2x2xi8>
// CHECK-SAME: -> tensor<2x1xi8> {
) -> tensor<2x2xi8> {
  // CHECK: %[[ALL_TO_ALL:.*]] = shard.all_to_all %[[ARG]] on @grid_1d
  // CHECK-SAME: grid_axes = [0] split_axis = 1 concat_axis = 0 : tensor<1x2xi8> -> tensor<2x1xi8>
  %s0 = shard.sharding @grid_1d split_axes = [[0]] : !shard.sharding
  %0 = shard.shard %arg0 to %s0  : tensor<2x2xi8>
  %s1 = shard.sharding @grid_1d split_axes = [[], [0]] : !shard.sharding
  %1 = shard.shard %0 to %s1  annotate_for_users : tensor<2x2xi8>
  // CHECK: return %[[ALL_TO_ALL]] : tensor<2x1xi8>
  return %1 : tensor<2x2xi8>
}

// CHECK-LABEL: func.func @unsplit_and_split(
// CHECK-SAME: [[varg0:%.*]]: tensor<3x10x10x15xi8>) -> tensor<6x10x2x15xi8> {
func.func @unsplit_and_split(%arg: tensor<6x10x120x15xi8>) -> tensor<6x10x120x15xi8> {
  %sharding1 = shard.sharding @grid_4d split_axes = [[0], [], [1,2]] : !shard.sharding
  %arg_sharded = shard.shard %arg to %sharding1 : tensor<6x10x120x15xi8>
  %sharding2 = shard.sharding @grid_4d split_axes = [[], [], [1,2,3]] : !shard.sharding
  %res_sharded = shard.shard %arg_sharded to %sharding2 annotate_for_users : tensor<6x10x120x15xi8>
  // CHECK: [[vall_gather:%.*]] = shard.all_gather [[varg0]] on @grid_4d grid_axes = [0] gather_axis = 0 : tensor<3x10x10x15xi8> -> tensor<6x10x10x15xi8>
  // CHECK: [[vall_slice:%.*]] = shard.all_slice [[vall_gather]] on @grid_4d grid_axes = [3] slice_axis = 2 : tensor<6x10x10x15xi8> -> tensor<6x10x2x15xi8>
  // CHECK: return [[vall_slice]] : tensor<6x10x2x15xi8>
  return %res_sharded : tensor<6x10x120x15xi8>
}

// CHECK-LABEL: func.func @move_and_split(
// CHECK-SAME: [[varg0:%.*]]: tensor<3x10x10x15xi8>) -> tensor<6x5x2x15xi8> {
func.func @move_and_split(%arg: tensor<6x10x120x15xi8>) -> tensor<6x10x120x15xi8> {
  %sharding1 = shard.sharding @grid_4d split_axes = [[0], [], [1,2]] : !shard.sharding
  %arg_sharded = shard.shard %arg to %sharding1 : tensor<6x10x120x15xi8>
  %sharding2 = shard.sharding @grid_4d split_axes = [[], [0], [1,2,3]] : !shard.sharding
  %res_sharded = shard.shard %arg_sharded to %sharding2 annotate_for_users : tensor<6x10x120x15xi8>
  // CHECK: [[vall_to_all:%.*]] = shard.all_to_all [[varg0]] on @grid_4d grid_axes = [0] split_axis = 1 concat_axis = 0 : tensor<3x10x10x15xi8> -> tensor<6x5x10x15xi8>
  // CHECK: [[vall_slice:%.*]] = shard.all_slice [[vall_to_all]] on @grid_4d grid_axes = [3] slice_axis = 2 : tensor<6x5x10x15xi8> -> tensor<6x5x2x15xi8>
  // CHECK: return [[vall_slice]] : tensor<6x5x2x15xi8>
  return %res_sharded : tensor<6x10x120x15xi8>
}

// CHECK-LABEL: func.func @move_and_unsplit(
// CHECK-SAME: [[varg0:%.*]]: tensor<3x10x10x15xi8>) -> tensor<6x5x40x15xi8> {
func.func @move_and_unsplit(%arg: tensor<6x10x120x15xi8>) -> tensor<6x10x120x15xi8> {
  %sharding1 = shard.sharding @grid_4d split_axes = [[0], [], [1,2]] : !shard.sharding
  %arg_sharded = shard.shard %arg to %sharding1 : tensor<6x10x120x15xi8>
  %sharding2 = shard.sharding @grid_4d split_axes = [[], [0], [1]] : !shard.sharding
  %res_sharded = shard.shard %arg_sharded to %sharding2 annotate_for_users : tensor<6x10x120x15xi8>
  // CHECK: [[vall_to_all:%.*]] = shard.all_to_all [[varg0]] on @grid_4d grid_axes = [0] split_axis = 1 concat_axis = 0 : tensor<3x10x10x15xi8> -> tensor<6x5x10x15xi8>
  // CHECK: [[vall_gather:%.*]] = shard.all_gather [[vall_to_all]] on @grid_4d grid_axes = [2] gather_axis = 2 : tensor<6x5x10x15xi8> -> tensor<6x5x40x15xi8>
  // CHECK: return [[vall_gather]] : tensor<6x5x40x15xi8>
  return %res_sharded : tensor<6x10x120x15xi8>
}

// CHECK-LABEL: func.func @unsplit_move_split(
// CHECK-SAME: [[varg0:%.*]]: tensor<3x5x120x3xi8>) -> tensor<6x20x30x1xi8>
func.func @unsplit_move_split(%arg: tensor<6x20x120x15xi8>) -> tensor<6x20x120x15xi8> {
  %sharding1 = shard.sharding @grid_4d split_axes = [[0], [2], [], [3]] : !shard.sharding
  %arg_sharded = shard.shard %arg to %sharding1 : tensor<6x20x120x15xi8>
  %sharding2 = shard.sharding @grid_4d split_axes = [[], [], [2], [3, 1]] : !shard.sharding
  %res_sharded = shard.shard %arg_sharded to %sharding2 annotate_for_users : tensor<6x20x120x15xi8>
  // CHECK: [[vall_gather:%.*]] = shard.all_gather [[varg0]] on @grid_4d grid_axes = [0] gather_axis = 0 : tensor<3x5x120x3xi8> -> tensor<6x5x120x3xi8>
  // CHECK: [[vall_to_all:%.*]] = shard.all_to_all [[vall_gather]] on @grid_4d grid_axes = [2] split_axis = 2 concat_axis = 1 : tensor<6x5x120x3xi8> -> tensor<6x20x30x3xi8>
  // CHECK: [[vall_slice:%.*]] = shard.all_slice [[vall_to_all]] on @grid_4d grid_axes = [1] slice_axis = 3 : tensor<6x20x30x3xi8> -> tensor<6x20x30x1xi8>
  // CHECK: return [[vall_slice]] : tensor<6x20x30x1xi8>
  return %res_sharded : tensor<6x20x120x15xi8>
}

// CHECK-LABEL: func @non_tensor_value
func.func @non_tensor_value(
  // CHECK-SAME: %[[ARG:.*]]: i8
  %arg0: i8
// CHECK-SAME: -> i8 {
) -> i8 {
  // CHECK: %[[RES:.*]] = arith.addi %[[ARG]], %[[ARG]] : i8
  %0 = arith.addi %arg0, %arg0 : i8
  // CHECK: return %[[RES]] : i8
  return %0 : i8
}

// CHECK-LABEL: func @unary_elementwise
func.func @unary_elementwise(
  // CHECK-SAME: %[[ARG:.*]]: tensor<1xi8>
  %arg0: tensor<2xi8>
// CHECK-SAME: -> tensor<1xi8> {
) -> tensor<2xi8> {
  %s0 = shard.sharding @grid_1d split_axes = [[0]] : !shard.sharding
  %0 = shard.shard %arg0 to %s0  : tensor<2xi8>
  %s1 = shard.sharding @grid_1d split_axes = [[0]] : !shard.sharding
  %1 = shard.shard %0 to %s1  annotate_for_users : tensor<2xi8>
  // CHECK: %[[RES:.*]] = tosa.abs %[[ARG]] : (tensor<1xi8>) -> tensor<1xi8>
  %2 = tosa.abs %1 : (tensor<2xi8>) -> tensor<2xi8>
  %s3 = shard.sharding @grid_1d split_axes = [[0]] : !shard.sharding
  %3 = shard.shard %2 to %s3  : tensor<2xi8>
  %s4 = shard.sharding @grid_1d split_axes = [[0]] : !shard.sharding
  %4 = shard.shard %3 to %s4  annotate_for_users : tensor<2xi8>
  // CHECK: return %[[RES]] : tensor<1xi8>
  return %4 : tensor<2xi8>
}

// full replication -> shard axis -> abs -> shard axis -> full replication
// CHECK-LABEL: func @unary_elementwise_with_resharding
func.func @unary_elementwise_with_resharding(
  // CHECK-SAME: %[[ARG:.*]]: tensor<2xi8>
  %arg0: tensor<2xi8>
// CHECK-SAME: -> tensor<2xi8> {
) -> tensor<2xi8> {
  // CHECK: %[[SLICE:.*]] = shard.all_slice %[[ARG]] on @grid_1d grid_axes = [0] slice_axis = 0
  // CHECK-SAME: tensor<2xi8> -> tensor<1xi8>
  %s0 = shard.sharding @grid_1d split_axes = [[]] : !shard.sharding
  %0 = shard.shard %arg0 to %s0  : tensor<2xi8>
  %s1 = shard.sharding @grid_1d split_axes = [[0]] : !shard.sharding
  %1 = shard.shard %0 to %s1  annotate_for_users : tensor<2xi8>
  // CHECK: %[[ABS:.*]] = tosa.abs %[[SLICE]] : (tensor<1xi8>) -> tensor<1xi8>
  %2 = tosa.abs %1 : (tensor<2xi8>) -> tensor<2xi8>
  // CHECK: %[[RES:.*]] = shard.all_gather %[[ABS]] on @grid_1d
  // CHECK-SAME: grid_axes = [0] gather_axis = 0 : tensor<1xi8> -> tensor<2xi8>
  %s3 = shard.sharding @grid_1d split_axes = [[0]] : !shard.sharding
  %3 = shard.shard %2 to %s3  : tensor<2xi8>
  %s4 = shard.sharding @grid_1d split_axes = [[]] : !shard.sharding
  %4 = shard.shard %3 to %s4  annotate_for_users : tensor<2xi8>
  // CHECK: return %[[RES]] : tensor<2xi8>
  return %4 : tensor<2xi8>
}

// CHECK-LABEL: func @binary_elementwise
func.func @binary_elementwise(
  // CHECK-SAME: %[[ARG0:.*]]: tensor<1xi8>,
  %arg0: tensor<2xi8>,
  // CHECK-SAME: %[[ARG1:.*]]: tensor<1xi8>
  %arg1: tensor<2xi8>
// CHECK-SAME: -> tensor<1xi8> {
) -> tensor<2xi8> {
  %sarg0_sharded = shard.sharding @grid_1d split_axes = [[0]] : !shard.sharding
  %arg0_sharded = shard.shard %arg0 to %sarg0_sharded  : tensor<2xi8>
  %sop_arg0 = shard.sharding @grid_1d split_axes = [[0]] : !shard.sharding
  %op_arg0 = shard.shard %arg0_sharded to %sop_arg0  annotate_for_users : tensor<2xi8>
  %sarg1_sharded = shard.sharding @grid_1d split_axes = [[0]] : !shard.sharding
  %arg1_sharded = shard.shard %arg1 to %sarg1_sharded  : tensor<2xi8>
  %sop_arg1 = shard.sharding @grid_1d split_axes = [[0]] : !shard.sharding
  %op_arg1 = shard.shard %arg1_sharded to %sop_arg1  annotate_for_users : tensor<2xi8>
  // CHECK: %[[RES:.*]] = tosa.add %[[ARG0]], %[[ARG1]] : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
  %op_res = tosa.add %op_arg0, %op_arg1 : (tensor<2xi8>, tensor<2xi8>) -> tensor<2xi8>
  %sop_res_sharded = shard.sharding @grid_1d split_axes = [[0]] : !shard.sharding
  %op_res_sharded = shard.shard %op_res to %sop_res_sharded  : tensor<2xi8>
  %sres = shard.sharding @grid_1d split_axes = [[0]] : !shard.sharding
  %res = shard.shard %op_res_sharded to %sres  annotate_for_users : tensor<2xi8>
  // CHECK: return %[[RES]] : tensor<1xi8>
  return %res : tensor<2xi8>
}

// reshard
// abs
// reshard
// abs
// reshard
// CHECK-LABEL: func @multiple_chained_ops
func.func @multiple_chained_ops(
  // CHECK-SAME: %[[ARG:.*]]: tensor<2xi8>
  %arg0: tensor<2xi8>
// CHECK-SAME: -> tensor<1xi8> {
) -> tensor<2xi8> {
  // CHECK: %[[RESHARD1:.*]] = shard.all_slice %[[ARG]] on @grid_1d grid_axes = [0] slice_axis = 0
  // CHECK-SAME: tensor<2xi8> -> tensor<1xi8>
  %s0 = shard.sharding @grid_1d split_axes = [[]] : !shard.sharding
  %0 = shard.shard %arg0 to %s0  : tensor<2xi8>
  %s1 = shard.sharding @grid_1d split_axes = [[0]] : !shard.sharding
  %1 = shard.shard %0 to %s1  annotate_for_users : tensor<2xi8>
  // CHECK: %[[ABS1:.*]] = tosa.abs %[[RESHARD1]] : (tensor<1xi8>) -> tensor<1xi8>
  %2 = tosa.abs %1 : (tensor<2xi8>) -> tensor<2xi8>
  // CHECK: %[[RESHARD2:.*]] = shard.all_gather %[[ABS1]] on @grid_1d
  // CHECK-SAME: grid_axes = [0] gather_axis = 0 : tensor<1xi8> -> tensor<2xi8>
  %s3 = shard.sharding @grid_1d split_axes = [[0]] : !shard.sharding
  %3 = shard.shard %2 to %s3  : tensor<2xi8>
  %s4 = shard.sharding @grid_1d split_axes = [[]] : !shard.sharding
  %4 = shard.shard %3 to %s4  annotate_for_users : tensor<2xi8>
  // CHECK: %[[ABS2:.*]] = tosa.abs %[[RESHARD2]] : (tensor<2xi8>) -> tensor<2xi8>
  %5 = tosa.abs %4 : (tensor<2xi8>) -> tensor<2xi8>
  // CHECK: %[[RESHARD3:.*]] = shard.all_slice %[[ABS2]] on @grid_1d grid_axes = [0] slice_axis = 0 :
  // CHECK-SAME: tensor<2xi8> -> tensor<1xi8>
  %s6 = shard.sharding @grid_1d split_axes = [[]] : !shard.sharding
  %6 = shard.shard %5 to %s6  : tensor<2xi8>
  %s7 = shard.sharding @grid_1d split_axes = [[0]] : !shard.sharding
  %7 = shard.shard %6 to %s7  annotate_for_users : tensor<2xi8>
  // CHECK: return %[[RESHARD3]] : tensor<1xi8>
  return %7 : tensor<2xi8>
}

// CHECK-LABEL: func @incomplete_sharding
func.func @incomplete_sharding(
  // CHECK-SAME: %[[ARG:.*]]: tensor<4x16xf32>
  %arg0: tensor<8x16xf32>
// CHECK-SAME: -> tensor<4x16xf32> {
) -> tensor<8x16xf32> {
  %s0 = shard.sharding @grid_1d split_axes = [[0]] : !shard.sharding
  %0 = shard.shard %arg0 to %s0  annotate_for_users : tensor<8x16xf32>
  // CHECK: %[[RES:.*]] = tosa.sigmoid %[[ARG]] : (tensor<4x16xf32>) -> tensor<4x16xf32>
  %1 = tosa.sigmoid %0 : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %s2 = shard.sharding @grid_1d split_axes = [[0]] : !shard.sharding
  %2 = shard.shard %1 to %s2 : tensor<8x16xf32>
  %3 = shard.shard %2 to %s2 annotate_for_users : tensor<8x16xf32>
  // CHECK: return %[[RES]] : tensor<4x16xf32>
  return %3 : tensor<8x16xf32>
}

// CHECK-LABEL: func @ew_chain_with_halo
func.func @ew_chain_with_halo(
  // CHECK-SAME: %[[IN1:[A-Za-z0-9_]+]]: tensor<5x16xf32>
  %arg0: tensor<8x16xf32>,
  // CHECK-SAME: %[[IN2:[A-Za-z0-9_]+]]: tensor<1xf32>
  %arg1: tensor<1xf32>,
  // CHECK-SAME: %[[IN3:[A-Za-z0-9_]+]]: tensor<1xf32>
  %arg2: tensor<1xf32>)
  // CHECK-SAME: -> tensor<5x16xf32>
   -> tensor<8x16xf32> {
  %ssharded = shard.sharding @grid_1d_4 split_axes = [[0]] halo_sizes = [2, 1] : !shard.sharding
  %sharded = shard.shard %arg0 to %ssharded  annotate_for_users : tensor<8x16xf32>
  // CHECK: %[[TMP1:.*]] = tosa.tanh %[[IN1]] : (tensor<5x16xf32>) -> tensor<5x16xf32>
  %0 = tosa.tanh %sharded : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %ssharded_0 = shard.sharding @grid_1d_4 split_axes = [[0]] halo_sizes = [2, 1] : !shard.sharding
  %sharded_0 = shard.shard %0 to %ssharded_0  : tensor<8x16xf32>
  %ssharded_1 = shard.sharding @grid_1d_4 split_axes = [[0]] halo_sizes = [2, 1] : !shard.sharding
  %sharded_1 = shard.shard %sharded_0 to %ssharded_1  annotate_for_users : tensor<8x16xf32>
  // CHECK-NEXT: %[[TMP2:.*]] = tosa.abs %[[TMP1]] : (tensor<5x16xf32>) -> tensor<5x16xf32>
  %1 = tosa.abs %sharded_1 : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %ssharded_2 = shard.sharding @grid_1d_4 split_axes = [[0]] halo_sizes = [2, 1] : !shard.sharding
  %sharded_2 = shard.shard %1 to %ssharded_2  : tensor<8x16xf32>
  %ssharded_4 = shard.sharding @grid_1d_4 split_axes = [[0]] halo_sizes = [2, 1] : !shard.sharding
  %sharded_4 = shard.shard %sharded_2 to %ssharded_4  annotate_for_users : tensor<8x16xf32>
  // CHECK-NEXT: %[[TMP3:.*]] = tosa.negate %[[TMP2]], %[[IN2]], %[[IN3]] : (tensor<5x16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x16xf32>
  %sharding_1 = shard.sharding @grid_1d_4 split_axes = [[]] : !shard.sharding
  %zero_point_1 = shard.shard %arg1 to %sharding_1 annotate_for_users : tensor<1xf32>
  %zero_point_2 = shard.shard %arg2 to %sharding_1 annotate_for_users : tensor<1xf32>
  %2 = tosa.negate %sharded_4, %zero_point_1, %zero_point_2 : (tensor<8x16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<8x16xf32>
  %ssharded_5 = shard.sharding @grid_1d_4 split_axes = [[0]] halo_sizes = [2, 1] : !shard.sharding
  %sharded_5 = shard.shard %2 to %ssharded_5  : tensor<8x16xf32>
  %ssharded_6 = shard.sharding @grid_1d_4 split_axes = [[0]] halo_sizes = [2, 1] : !shard.sharding
  %sharded_6 = shard.shard %sharded_5 to %ssharded_6  annotate_for_users : tensor<8x16xf32>
  // CHECK-NEXT: return %[[TMP3]] : tensor<5x16xf32>
  return %sharded_6 : tensor<8x16xf32>
}

// CHECK-LABEL: func @test_shard_update_halo
// CHECK-SAME: %[[IN1:[A-Za-z0-9_]+]]: tensor<300x1200xi64>
func.func @test_shard_update_halo(%arg0: tensor<1200x1200xi64>) -> tensor<1200x1200xi64> {
  %sharding = shard.sharding @grid_1d_4 split_axes = [[0]] : !shard.sharding
  // CHECK: %[[T:.*]] = tensor.empty() : tensor<304x1200xi64>
  // CHECK: %[[inserted_slice:.*]] = tensor.insert_slice %[[IN1]] into %[[T]][2, 0] [300, 1200] [1, 1] : tensor<300x1200xi64> into tensor<304x1200xi64>
  // CHECK: %[[UH:.*]] = shard.update_halo %[[inserted_slice]] on @grid_1d_4 split_axes = {{\[\[0]]}} halo_sizes = [2, 2] : tensor<304x1200xi64>
  %sharded = shard.shard %arg0 to %sharding : tensor<1200x1200xi64>
  %sharding_0 = shard.sharding @grid_1d_4 split_axes = [[0]] halo_sizes = [2, 2] : !shard.sharding
  %sharded_1 = shard.shard %sharded to %sharding_0 : tensor<1200x1200xi64>
  %sharded_3 = shard.shard %sharded_1 to %sharding_0 annotate_for_users : tensor<1200x1200xi64>
  // CHECK: return %[[UH]] : tensor<304x1200xi64>
  return %sharded_3 : tensor<1200x1200xi64>
}

shard.grid @grid4x4(shape = 4x4)
// CHECK-LABEL: func @test_shard_update_halo2d
// CHECK-SAME: %[[IN1:[A-Za-z0-9_]+]]: tensor<300x300xi64>
func.func @test_shard_update_halo2d(%arg0: tensor<1200x1200xi64>) -> tensor<1200x1200xi64> {
  %sharding = shard.sharding @grid4x4 split_axes = [[0], [1]] : !shard.sharding
  // CHECK: %[[T:.*]] = tensor.empty() : tensor<303x307xi64>
  // CHECK: %[[inserted_slice:.*]] = tensor.insert_slice %[[IN1]] into %[[T]][1, 3] [300, 300] [1, 1] : tensor<300x300xi64> into tensor<303x307xi64>
  // CHECK: %[[UH:.*]] = shard.update_halo %[[inserted_slice]] on @grid4x4 split_axes = {{\[\[}}0], [1]] halo_sizes = [1, 2, 3, 4] : tensor<303x307xi64>
  %sharded = shard.shard %arg0 to %sharding : tensor<1200x1200xi64>
  %sharding_0 = shard.sharding @grid4x4 split_axes = [[0], [1]] halo_sizes = [1, 2, 3, 4] : !shard.sharding
  %sharded_1 = shard.shard %sharded to %sharding_0 : tensor<1200x1200xi64>
  %sharded_3 = shard.shard %sharded_1 to %sharding_0 annotate_for_users : tensor<1200x1200xi64>
  // CHECK: return %[[UH]] : tensor<303x307xi64>
  return %sharded_3 : tensor<1200x1200xi64>
}

shard.grid @grid(shape = 2)
// CHECK-LABEL: func.func @test_reduce_0d(
// CHECK-SAME: %[[ARG0:[A-Za-z0-9_]+]]: tensor<3x6xi32>
func.func @test_reduce_0d(%arg0: tensor<6x6xi32>) -> (tensor<i32>) {
  %sharding = shard.sharding @grid split_axes = [[0]] : !shard.sharding
  %sharded = shard.shard %arg0 to %sharding annotate_for_users : tensor<6x6xi32>
  %4 = tensor.empty() : tensor<i32>
  %sharding_out = shard.sharding @grid split_axes = [[]] : !shard.sharding
  %sharded_out = shard.shard %4 to %sharding_out : tensor<i32>
  %sharded_in = shard.shard %sharded to %sharding annotate_for_users : tensor<6x6xi32>
  // CHECK: %[[reduced:.*]] = linalg.reduce ins(%arg0 : tensor<3x6xi32>)
  %reduced = linalg.reduce ins(%sharded_in : tensor<6x6xi32>) outs(%sharded_out : tensor<i32>) dimensions = [0, 1] 
    (%in: i32, %init: i32) {
      %6 = arith.addi %in, %init : i32
      linalg.yield %6 : i32
    }
  // CHECK: %[[all_reduce:.*]] = shard.all_reduce %[[reduced]] on @grid grid_axes = [0] : tensor<i32> -> tensor<i32>
  %sharded_red = shard.shard %reduced to %sharding_out : tensor<i32>
  %sharded_ret = shard.shard %sharded_red to %sharding_out annotate_for_users : tensor<i32>
  // CHECK: return %[[all_reduce]] : tensor<i32>
  return %sharded_ret : tensor<i32>
}

// CHECK-LABEL: func.func @test_reduce_1d(
// CHECK-SAME: %[[ARG0:[A-Za-z0-9_]+]]: tensor<3x6xi32>
func.func @test_reduce_1d(%arg0: tensor<6x6xi32>) -> (tensor<6xi32>) {
  %sharding = shard.sharding @grid split_axes = [[0]] : !shard.sharding
  %sharded = shard.shard %arg0 to %sharding annotate_for_users : tensor<6x6xi32>
  %4 = tensor.empty() : tensor<6xi32>
  %sharded_4 = shard.shard %4 to %sharding : tensor<6xi32>
  %sharded_out = shard.shard %sharded_4 to %sharding annotate_for_users : tensor<6xi32>
  %sharded_in = shard.shard %sharded to %sharding annotate_for_users : tensor<6x6xi32>
  // CHECK: %[[reduced:.*]] = linalg.reduce ins(%arg0 : tensor<3x6xi32>)
  %reduced = linalg.reduce ins(%sharded_in : tensor<6x6xi32>) outs(%sharded_out : tensor<6xi32>) dimensions = [1] 
    (%in: i32, %init: i32) {
      %6 = arith.addi %in, %init : i32
      linalg.yield %6 : i32
    }
  // CHECK-NOT: shard.all_reduce
  %sharded_red = shard.shard %reduced to %sharding : tensor<6xi32>
  %sharded_ret = shard.shard %sharded_red to %sharding annotate_for_users : tensor<6xi32>
  // CHECK: return %[[reduced]] : tensor<3xi32>
  return %sharded_ret : tensor<6xi32>
}

// CHECK-LABEL: func.func @mlp_1d_weight_stationary
// CHECK-SAME: [[varg0:%.*]]: tensor<512x512xf32>, [[varg1:%.*]]: tensor<2048x256xf32>, [[varg2:%.*]]: tensor<256x2048xf32>) -> tensor<512x2048xf32>
func.func @mlp_1d_weight_stationary(%arg0: tensor<512x2048xf32>, %arg1: tensor<2048x1024xf32>, %arg2: tensor<1024x2048xf32>) -> tensor<512x2048xf32> attributes {llvm.emit_c_interface} {
  // CHECK: [[vcst:%.*]] = arith.constant 0.000000e+00 : f32
  %sharding = shard.sharding @grid_1d_4 split_axes = [[], [0]] : !shard.sharding
  %sharding_0 = shard.sharding @grid_1d_4 split_axes = [[0], []] : !shard.sharding
  %sharding_1 = shard.sharding @grid_1d_4 split_axes = [[]] : !shard.sharding
  %sharding_2 = shard.sharding @grid_1d_4 split_axes = [[], [0]] : !shard.sharding
  %sharding_3 = shard.sharding @grid_1d_4 split_axes = [[], [0]] : !shard.sharding
  %sharding_4 = shard.sharding @grid_1d_4 split_axes = [[0], []] : !shard.sharding
  %sharding_5 = shard.sharding @grid_1d_4 split_axes = [[]] : !shard.sharding
  %sharding_annotated = shard.shard %arg0 to %sharding_2 : tensor<512x2048xf32>
  %sharding_annotated_6 = shard.shard %arg1 to %sharding_3 : tensor<2048x1024xf32>
  %sharding_annotated_7 = shard.shard %arg2 to %sharding_4 : tensor<1024x2048xf32>
  // CHECK-DAG: [[v0:%.*]] = tensor.empty() : tensor<512x256xf32>
  %0 = tensor.empty() : tensor<512x1024xf32>
  %sharding_annotated_8 = shard.shard %0 to %sharding : tensor<512x1024xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %sharding_annotated_9 = shard.shard %sharding_annotated_8 to %sharding annotate_for_users : tensor<512x1024xf32>
  // CHECK-DAG: [[v1:%.*]] = linalg.fill ins([[vcst]] : f32) outs([[v0]] : tensor<512x256xf32>) -> tensor<512x256xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%sharding_annotated_9 : tensor<512x1024xf32>) -> tensor<512x1024xf32>
  %sharding_annotated_10 = shard.shard %1 to %sharding : tensor<512x1024xf32>
  // CHECK-DAG: [[vall_gather:%.*]] = shard.all_gather [[varg0]] on @grid_1d_4 grid_axes = [0] gather_axis = 1 : tensor<512x512xf32> -> tensor<512x2048xf32>
  %sharding_annotated_11 = shard.shard %sharding_annotated to %sharding_1 annotate_for_users : tensor<512x2048xf32>
  %sharding_annotated_12 = shard.shard %sharding_annotated_6 to %sharding annotate_for_users : tensor<2048x1024xf32>
  %sharding_annotated_13 = shard.shard %sharding_annotated_10 to %sharding annotate_for_users : tensor<512x1024xf32>
  // CHECK: [[v2:%.*]] = linalg.matmul ins([[vall_gather]], [[varg1]] : tensor<512x2048xf32>, tensor<2048x256xf32>) outs([[v1]] : tensor<512x256xf32>) -> tensor<512x256xf32>
  %2 = linalg.matmul ins(%sharding_annotated_11, %sharding_annotated_12 : tensor<512x2048xf32>, tensor<2048x1024xf32>) outs(%sharding_annotated_13 : tensor<512x1024xf32>) -> tensor<512x1024xf32>
  %sharding_annotated_14 = shard.shard %2 to %sharding : tensor<512x1024xf32>
  %sharding_annotated_15 = shard.shard %sharding_annotated_14 to %sharding annotate_for_users : tensor<512x1024xf32>
  // CHECK: [[v3:%.*]] = tosa.sigmoid [[v2]] : (tensor<512x256xf32>) -> tensor<512x256xf32>
  %3 = tosa.sigmoid %sharding_annotated_15 : (tensor<512x1024xf32>) -> tensor<512x1024xf32>
  %sharding_annotated_16 = shard.shard %3 to %sharding : tensor<512x1024xf32>
  // CHECK: [[v9:%.*]] = tensor.empty() : tensor<512x2048xf32>
  %4 = tensor.empty() : tensor<512x2048xf32>
  %sharding_annotated_17 = shard.shard %4 to %sharding_1 : tensor<512x2048xf32>
  %sharding_annotated_18 = shard.shard %sharding_annotated_17 to %sharding_1 annotate_for_users : tensor<512x2048xf32>
  // CHECK: [[v10:%.*]] = linalg.fill ins([[vcst]] : f32) outs([[v9]] : tensor<512x2048xf32>) -> tensor<512x2048xf32>
  %5 = linalg.fill ins(%cst : f32) outs(%sharding_annotated_18 : tensor<512x2048xf32>) -> tensor<512x2048xf32>
  %sharding_annotated_19 = shard.shard %5 to %sharding_1 : tensor<512x2048xf32>
  %sharding_annotated_20 = shard.shard %sharding_annotated_16 to %sharding annotate_for_users : tensor<512x1024xf32>
  %sharding_annotated_21 = shard.shard %sharding_annotated_7 to %sharding_0 annotate_for_users : tensor<1024x2048xf32>
  %sharding_annotated_22 = shard.shard %sharding_annotated_19 to %sharding_1 annotate_for_users : tensor<512x2048xf32>
  // CHECK: [[v7:%.*]] = scf.if
  // CHECK: [[v8:%.*]] = linalg.matmul ins([[v3]], [[varg2]] : tensor<512x256xf32>, tensor<256x2048xf32>) outs([[v7]] : tensor<512x2048xf32>) -> tensor<512x2048xf32>
  %6 = linalg.matmul ins(%sharding_annotated_20, %sharding_annotated_21 : tensor<512x1024xf32>, tensor<1024x2048xf32>) outs(%sharding_annotated_22 : tensor<512x2048xf32>) -> tensor<512x2048xf32>
  %sharding_annotated_23 = shard.shard %6 to %sharding_1 : tensor<512x2048xf32>
  // CHECK: [[vall_reduce:%.*]] = shard.all_reduce [[v8]] on @grid_1d_4 grid_axes = [0] : tensor<512x2048xf32> -> tensor<512x2048xf32>
  %sharding_annotated_24 = shard.shard %sharding_annotated_23 to %sharding_5 annotate_for_users : tensor<512x2048xf32>
  // CHECK: return [[vall_reduce]] : tensor<512x2048xf32>
  return %sharding_annotated_24 : tensor<512x2048xf32>
}
