// RUN: mlir-opt \
// RUN:   --pass-pipeline="builtin.module(func.func(mesh-spmdization,test-single-fold))" \
// RUN:   %s | FileCheck %s

mesh.mesh @mesh_1d(shape = 2)

// CHECK-LABEL: func @return_sharding
func.func @return_sharding(
  // CHECK-SAME: [[ARG:%.*]]: tensor<1xf32>
  %arg0: tensor<2xf32>
// CHECK-SAME: ) -> (tensor<1xf32>, !mesh.sharding) {
) -> (tensor<2xf32>, !mesh.sharding) {
  %ssharding_annotated = mesh.sharding @mesh_1d split_axes = [[0]] : !mesh.sharding
  %sharding_annotated = mesh.shard %arg0 to %ssharding_annotated  : tensor<2xf32>
  // CHECK-NEXT: [[vsharding:%.*]] = mesh.sharding @mesh_1d split_axes = {{\[\[}}0]] : !mesh.sharding
  %r = mesh.get_sharding %sharding_annotated : tensor<2xf32> -> !mesh.sharding
  // CHECK-NEXT: return [[ARG]], [[vsharding]] : tensor<1xf32>, !mesh.sharding
  return %sharding_annotated, %r : tensor<2xf32>, !mesh.sharding
}

// CHECK-LABEL: func @full_replication
func.func @full_replication(
  // CHECK-SAME: %[[ARG:.*]]: tensor<2xi8>
  %arg0: tensor<2xi8>
// CHECK-SAME: -> tensor<2xi8> {
) -> tensor<2xi8> {
  %s0 = mesh.sharding @mesh_1d split_axes = [[]] : !mesh.sharding
  %0 = mesh.shard %arg0 to %s0  : tensor<2xi8>
  %s1 = mesh.sharding @mesh_1d split_axes = [[]] : !mesh.sharding
  %1 = mesh.shard %0 to %s1  annotate_for_users : tensor<2xi8>
  // CHECK: return %[[ARG]] : tensor<2xi8>
  return %1 : tensor<2xi8>
}

// CHECK-LABEL: func @sharding_triplet
func.func @sharding_triplet(
  // CHECK-SAME: %[[ARG:.*]]: tensor<1xf32>
  %arg0: tensor<2xf32>
// CHECK-SAME: ) -> tensor<2xf32> {
) -> tensor<2xf32> {
  // CHECK: %[[ALL_GATHER:.*]] = mesh.all_gather %[[ARG]] on @mesh_1d mesh_axes = [0] gather_axis = 0 : tensor<1xf32> -> tensor<2xf32>
  %ssharding_annotated = mesh.sharding @mesh_1d split_axes = [[0]] : !mesh.sharding
  %sharding_annotated = mesh.shard %arg0 to %ssharding_annotated  : tensor<2xf32>
  %ssharding_annotated_0 = mesh.sharding @mesh_1d split_axes = [[0]] : !mesh.sharding
  %sharding_annotated_0 = mesh.shard %sharding_annotated to %ssharding_annotated_0  annotate_for_users : tensor<2xf32>
  %ssharding_annotated_1 = mesh.sharding @mesh_1d split_axes = [[]] : !mesh.sharding
  %sharding_annotated_1 = mesh.shard %sharding_annotated_0 to %ssharding_annotated_1  : tensor<2xf32>
  // CHECK: return %[[ALL_GATHER]] : tensor<2xf32>
  return %sharding_annotated_1 : tensor<2xf32>
}


// CHECK-LABEL: func @move_split_axis
func.func @move_split_axis(
  // CHECK-SAME: %[[ARG:.*]]: tensor<1x2xi8>
  %arg0: tensor<2x2xi8>
// CHECK-SAME: -> tensor<2x1xi8> {
) -> tensor<2x2xi8> {
  // CHECK: %[[ALL_TO_ALL:.*]] = mesh.all_to_all %[[ARG]] on @mesh_1d
  // CHECK-SAME: mesh_axes = [0] split_axis = 1 concat_axis = 0 : tensor<1x2xi8> -> tensor<2x1xi8>
  %s0 = mesh.sharding @mesh_1d split_axes = [[0]] : !mesh.sharding
  %0 = mesh.shard %arg0 to %s0  : tensor<2x2xi8>
  %s1 = mesh.sharding @mesh_1d split_axes = [[], [0]] : !mesh.sharding
  %1 = mesh.shard %0 to %s1  annotate_for_users : tensor<2x2xi8>
  // CHECK: return %[[ALL_TO_ALL]] : tensor<2x1xi8>
  return %1 : tensor<2x2xi8>
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
  %s0 = mesh.sharding @mesh_1d split_axes = [[0]] : !mesh.sharding
  %0 = mesh.shard %arg0 to %s0  : tensor<2xi8>
  %s1 = mesh.sharding @mesh_1d split_axes = [[0]] : !mesh.sharding
  %1 = mesh.shard %0 to %s1  annotate_for_users : tensor<2xi8>
  // CHECK: %[[RES:.*]] = tosa.abs %[[ARG]] : (tensor<1xi8>) -> tensor<1xi8>
  %2 = tosa.abs %1 : (tensor<2xi8>) -> tensor<2xi8>
  %s3 = mesh.sharding @mesh_1d split_axes = [[0]] : !mesh.sharding
  %3 = mesh.shard %2 to %s3  : tensor<2xi8>
  %s4 = mesh.sharding @mesh_1d split_axes = [[0]] : !mesh.sharding
  %4 = mesh.shard %3 to %s4  annotate_for_users : tensor<2xi8>
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
  // CHECK: %[[SLICE:.*]] = mesh.all_slice %[[ARG]] on @mesh_1d mesh_axes = [0] slice_axis = 0
  // CHECK-SAME: tensor<2xi8> -> tensor<1xi8>
  %s0 = mesh.sharding @mesh_1d split_axes = [[]] : !mesh.sharding
  %0 = mesh.shard %arg0 to %s0  : tensor<2xi8>
  %s1 = mesh.sharding @mesh_1d split_axes = [[0]] : !mesh.sharding
  %1 = mesh.shard %0 to %s1  annotate_for_users : tensor<2xi8>
  // CHECK: %[[ABS:.*]] = tosa.abs %[[SLICE]] : (tensor<1xi8>) -> tensor<1xi8>
  %2 = tosa.abs %1 : (tensor<2xi8>) -> tensor<2xi8>
  // CHECK: %[[RES:.*]] = mesh.all_gather %[[ABS]] on @mesh_1d
  // CHECK-SAME: mesh_axes = [0] gather_axis = 0 : tensor<1xi8> -> tensor<2xi8>
  %s3 = mesh.sharding @mesh_1d split_axes = [[0]] : !mesh.sharding
  %3 = mesh.shard %2 to %s3  : tensor<2xi8>
  %s4 = mesh.sharding @mesh_1d split_axes = [[]] : !mesh.sharding
  %4 = mesh.shard %3 to %s4  annotate_for_users : tensor<2xi8>
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
  %sarg0_sharded = mesh.sharding @mesh_1d split_axes = [[0]] : !mesh.sharding
  %arg0_sharded = mesh.shard %arg0 to %sarg0_sharded  : tensor<2xi8>
  %sop_arg0 = mesh.sharding @mesh_1d split_axes = [[0]] : !mesh.sharding
  %op_arg0 = mesh.shard %arg0_sharded to %sop_arg0  annotate_for_users : tensor<2xi8>
  %sarg1_sharded = mesh.sharding @mesh_1d split_axes = [[0]] : !mesh.sharding
  %arg1_sharded = mesh.shard %arg1 to %sarg1_sharded  : tensor<2xi8>
  %sop_arg1 = mesh.sharding @mesh_1d split_axes = [[0]] : !mesh.sharding
  %op_arg1 = mesh.shard %arg1_sharded to %sop_arg1  annotate_for_users : tensor<2xi8>
  // CHECK: %[[RES:.*]] = tosa.add %[[ARG0]], %[[ARG1]] : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
  %op_res = tosa.add %op_arg0, %op_arg1 : (tensor<2xi8>, tensor<2xi8>) -> tensor<2xi8>
  %sop_res_sharded = mesh.sharding @mesh_1d split_axes = [[0]] : !mesh.sharding
  %op_res_sharded = mesh.shard %op_res to %sop_res_sharded  : tensor<2xi8>
  %sres = mesh.sharding @mesh_1d split_axes = [[0]] : !mesh.sharding
  %res = mesh.shard %op_res_sharded to %sres  annotate_for_users : tensor<2xi8>
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
  // CHECK: %[[RESHARD1:.*]] = mesh.all_slice %[[ARG]] on @mesh_1d mesh_axes = [0] slice_axis = 0
  // CHECK-SAME: tensor<2xi8> -> tensor<1xi8>
  %s0 = mesh.sharding @mesh_1d split_axes = [[]] : !mesh.sharding
  %0 = mesh.shard %arg0 to %s0  : tensor<2xi8>
  %s1 = mesh.sharding @mesh_1d split_axes = [[0]] : !mesh.sharding
  %1 = mesh.shard %0 to %s1  annotate_for_users : tensor<2xi8>
  // CHECK: %[[ABS1:.*]] = tosa.abs %[[RESHARD1]] : (tensor<1xi8>) -> tensor<1xi8>
  %2 = tosa.abs %1 : (tensor<2xi8>) -> tensor<2xi8>
  // CHECK: %[[RESHARD2:.*]] = mesh.all_gather %[[ABS1]] on @mesh_1d
  // CHECK-SAME: mesh_axes = [0] gather_axis = 0 : tensor<1xi8> -> tensor<2xi8>
  %s3 = mesh.sharding @mesh_1d split_axes = [[0]] : !mesh.sharding
  %3 = mesh.shard %2 to %s3  : tensor<2xi8>
  %s4 = mesh.sharding @mesh_1d split_axes = [[]] : !mesh.sharding
  %4 = mesh.shard %3 to %s4  annotate_for_users : tensor<2xi8>
  // CHECK: %[[ABS2:.*]] = tosa.abs %[[RESHARD2]] : (tensor<2xi8>) -> tensor<2xi8>
  %5 = tosa.abs %4 : (tensor<2xi8>) -> tensor<2xi8>
  // CHECK: %[[RESHARD3:.*]] = mesh.all_slice %[[ABS2]] on @mesh_1d mesh_axes = [0] slice_axis = 0 :
  // CHECK-SAME: tensor<2xi8> -> tensor<1xi8>
  %s6 = mesh.sharding @mesh_1d split_axes = [[]] : !mesh.sharding
  %6 = mesh.shard %5 to %s6  : tensor<2xi8>
  %s7 = mesh.sharding @mesh_1d split_axes = [[0]] : !mesh.sharding
  %7 = mesh.shard %6 to %s7  annotate_for_users : tensor<2xi8>
  // CHECK: return %[[RESHARD3]] : tensor<1xi8>
  return %7 : tensor<2xi8>
}

// CHECK-LABEL: func @incomplete_sharding
func.func @incomplete_sharding(
  // CHECK-SAME: %[[ARG:.*]]: tensor<4x16xf32>
  %arg0: tensor<8x16xf32>
// CHECK-SAME: -> tensor<4x16xf32> {
) -> tensor<8x16xf32> {
  %s0 = mesh.sharding @mesh_1d split_axes = [[0]] : !mesh.sharding
  %0 = mesh.shard %arg0 to %s0  annotate_for_users : tensor<8x16xf32>
  // CHECK: %[[RES:.*]] = tosa.sigmoid %[[ARG]] : (tensor<4x16xf32>) -> tensor<4x16xf32>
  %1 = tosa.sigmoid %0 : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %s2 = mesh.sharding @mesh_1d split_axes = [[0]] : !mesh.sharding
  %2 = mesh.shard %1 to %s2  : tensor<8x16xf32>
  // CHECK: return %[[RES]] : tensor<4x16xf32>
  return %2 : tensor<8x16xf32>
}

mesh.mesh @mesh_1d_4(shape = 4)

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
  %ssharding_annotated = mesh.sharding @mesh_1d_4 split_axes = [[0]] halo_sizes = [2, 1] : !mesh.sharding
  %sharding_annotated = mesh.shard %arg0 to %ssharding_annotated  annotate_for_users : tensor<8x16xf32>
  // CHECK: %[[TMP1:.*]] = tosa.tanh %[[IN1]] : (tensor<5x16xf32>) -> tensor<5x16xf32>
  %0 = tosa.tanh %sharding_annotated : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %ssharding_annotated_0 = mesh.sharding @mesh_1d_4 split_axes = [[0]] halo_sizes = [2, 1] : !mesh.sharding
  %sharding_annotated_0 = mesh.shard %0 to %ssharding_annotated_0  : tensor<8x16xf32>
  %ssharding_annotated_1 = mesh.sharding @mesh_1d_4 split_axes = [[0]] halo_sizes = [2, 1] : !mesh.sharding
  %sharding_annotated_1 = mesh.shard %sharding_annotated_0 to %ssharding_annotated_1  annotate_for_users : tensor<8x16xf32>
  // CHECK-NEXT: %[[TMP2:.*]] = tosa.abs %[[TMP1]] : (tensor<5x16xf32>) -> tensor<5x16xf32>
  %1 = tosa.abs %sharding_annotated_1 : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %ssharding_annotated_2 = mesh.sharding @mesh_1d_4 split_axes = [[0]] halo_sizes = [2, 1] : !mesh.sharding
  %sharding_annotated_2 = mesh.shard %1 to %ssharding_annotated_2  : tensor<8x16xf32>
  %ssharding_annotated_4 = mesh.sharding @mesh_1d_4 split_axes = [[0]] halo_sizes = [2, 1] : !mesh.sharding
  %sharding_annotated_4 = mesh.shard %sharding_annotated_2 to %ssharding_annotated_4  annotate_for_users : tensor<8x16xf32>
  // CHECK-NEXT: %[[TMP3:.*]] = tosa.negate %[[TMP2]], %[[IN2]], %[[IN3]] : (tensor<5x16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x16xf32>
  %sharding_1 = mesh.sharding @mesh_1d_4 split_axes = [[]] : !mesh.sharding
  %zero_point_1 = mesh.shard %arg1 to %sharding_1 annotate_for_users : tensor<1xf32>
  %zero_point_2 = mesh.shard %arg2 to %sharding_1 annotate_for_users : tensor<1xf32>
  %2 = tosa.negate %sharding_annotated_4, %zero_point_1, %zero_point_2 : (tensor<8x16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<8x16xf32>
  %ssharding_annotated_5 = mesh.sharding @mesh_1d_4 split_axes = [[0]] halo_sizes = [2, 1] : !mesh.sharding
  %sharding_annotated_5 = mesh.shard %2 to %ssharding_annotated_5  : tensor<8x16xf32>
  %ssharding_annotated_6 = mesh.sharding @mesh_1d_4 split_axes = [[0]] halo_sizes = [2, 1] : !mesh.sharding
  %sharding_annotated_6 = mesh.shard %sharding_annotated_5 to %ssharding_annotated_6  annotate_for_users : tensor<8x16xf32>
  // CHECK-NEXT: return %[[TMP3]] : tensor<5x16xf32>
  return %sharding_annotated_6 : tensor<8x16xf32>
}

// CHECK-LABEL: func @test_shard_update_halo
// CHECK-SAME: %[[IN1:[A-Za-z0-9_]+]]: tensor<300x1200xi64>
func.func @test_shard_update_halo(%arg0: tensor<1200x1200xi64>) -> tensor<1200x1200xi64> {
  %sharding = mesh.sharding @mesh_1d_4 split_axes = [[0]] : !mesh.sharding
  // CHECK: %[[T:.*]] = tensor.empty() : tensor<304x1200xi64>
  // CHECK: %[[inserted_slice:.*]] = tensor.insert_slice %[[IN1]] into %[[T]][2, 0] [300, 1200] [1, 1] : tensor<300x1200xi64> into tensor<304x1200xi64>
  // CHECK: %[[UH:.*]] = mesh.update_halo %[[inserted_slice]] on @mesh_1d_4 split_axes = {{\[\[0]]}} halo_sizes = [2, 2] : tensor<304x1200xi64>
  %sharding_annotated = mesh.shard %arg0 to %sharding : tensor<1200x1200xi64>
  %sharding_0 = mesh.sharding @mesh_1d_4 split_axes = [[0]] halo_sizes = [2, 2] : !mesh.sharding
  %sharding_annotated_1 = mesh.shard %sharding_annotated to %sharding_0 : tensor<1200x1200xi64>
  %sharding_annotated_3 = mesh.shard %sharding_annotated_1 to %sharding_0 annotate_for_users : tensor<1200x1200xi64>
  // CHECK: return %[[UH]] : tensor<304x1200xi64>
  return %sharding_annotated_3 : tensor<1200x1200xi64>
}

mesh.mesh @mesh4x4(shape = 4x4)
// CHECK-LABEL: func @test_shard_update_halo2d
// CHECK-SAME: %[[IN1:[A-Za-z0-9_]+]]: tensor<300x300xi64>
func.func @test_shard_update_halo2d(%arg0: tensor<1200x1200xi64>) -> tensor<1200x1200xi64> {
  %sharding = mesh.sharding @mesh4x4 split_axes = [[0], [1]] : !mesh.sharding
  // CHECK: %[[T:.*]] = tensor.empty() : tensor<303x307xi64>
  // CHECK: %[[inserted_slice:.*]] = tensor.insert_slice %[[IN1]] into %[[T]][1, 3] [300, 300] [1, 1] : tensor<300x300xi64> into tensor<303x307xi64>
  // CHECK: %[[UH:.*]] = mesh.update_halo %[[inserted_slice]] on @mesh4x4 split_axes = {{\[\[}}0], [1]] halo_sizes = [1, 2, 3, 4] : tensor<303x307xi64>
  %sharding_annotated = mesh.shard %arg0 to %sharding : tensor<1200x1200xi64>
  %sharding_0 = mesh.sharding @mesh4x4 split_axes = [[0], [1]] halo_sizes = [1, 2, 3, 4] : !mesh.sharding
  %sharding_annotated_1 = mesh.shard %sharding_annotated to %sharding_0 : tensor<1200x1200xi64>
  %sharding_annotated_3 = mesh.shard %sharding_annotated_1 to %sharding_0 annotate_for_users : tensor<1200x1200xi64>
  // CHECK: return %[[UH]] : tensor<303x307xi64>
  return %sharding_annotated_3 : tensor<1200x1200xi64>
}

mesh.mesh @mesh(shape = 2)
// CHECK-LABEL: func.func @test_reduce_0d(
// CHECK-SAME: %[[ARG0:[A-Za-z0-9_]+]]: tensor<3x6xi32>
func.func @test_reduce_0d(%arg0: tensor<6x6xi32>) -> (tensor<i32>) {
  %sharding = mesh.sharding @mesh split_axes = [[0]] : !mesh.sharding
  %sharded = mesh.shard %arg0 to %sharding annotate_for_users : tensor<6x6xi32>
  %4 = tensor.empty() : tensor<i32>
  %sharding_out = mesh.sharding @mesh split_axes = [[]] : !mesh.sharding
  %sharded_out = mesh.shard %4 to %sharding_out : tensor<i32>
  %sharded_in = mesh.shard %sharded to %sharding annotate_for_users : tensor<6x6xi32>
  // CHECK: %[[reduced:.*]] = linalg.reduce ins(%arg0 : tensor<3x6xi32>)
  %reduced = linalg.reduce ins(%sharded_in : tensor<6x6xi32>) outs(%sharded_out : tensor<i32>) dimensions = [0, 1] 
    (%in: i32, %init: i32) {
      %6 = arith.addi %in, %init : i32
      linalg.yield %6 : i32
    }
  // CHECK: %[[all_reduce:.*]] = mesh.all_reduce %[[reduced]] on @mesh mesh_axes = [0] : tensor<i32> -> tensor<i32>
  %sharded_red = mesh.shard %reduced to %sharding_out : tensor<i32>
  %sharded_ret = mesh.shard %sharded_red to %sharding_out annotate_for_users : tensor<i32>
  // CHECK: return %[[all_reduce]] : tensor<i32>
  return %sharded_ret : tensor<i32>
}

// CHECK-LABEL: func.func @test_reduce_1d(
// CHECK-SAME: %[[ARG0:[A-Za-z0-9_]+]]: tensor<3x6xi32>
func.func @test_reduce_1d(%arg0: tensor<6x6xi32>) -> (tensor<6xi32>) {
  %sharding = mesh.sharding @mesh split_axes = [[0]] : !mesh.sharding
  %sharded = mesh.shard %arg0 to %sharding annotate_for_users : tensor<6x6xi32>
  %4 = tensor.empty() : tensor<6xi32>
  %sharded_out = mesh.shard %4 to %sharding : tensor<6xi32>
  %sharded_in = mesh.shard %sharded to %sharding annotate_for_users : tensor<6x6xi32>
  // CHECK: %[[reduced:.*]] = linalg.reduce ins(%arg0 : tensor<3x6xi32>)
  %reduced = linalg.reduce ins(%sharded_in : tensor<6x6xi32>) outs(%sharded_out : tensor<6xi32>) dimensions = [1] 
    (%in: i32, %init: i32) {
      %6 = arith.addi %in, %init : i32
      linalg.yield %6 : i32
    }
  // CHECK-NOT: mesh.all_reduce
  %sharded_red = mesh.shard %reduced to %sharding : tensor<6xi32>
  %sharded_ret = mesh.shard %sharded_red to %sharding annotate_for_users : tensor<6xi32>
  // CHECK: return %[[reduced]] : tensor<3xi32>
  return %sharded_ret : tensor<6xi32>
}
