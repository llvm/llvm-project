// RUN: mlir-opt --mesh-spmdization --test-constant-fold %s | FileCheck %s

mesh.mesh @mesh_1d(shape = 2)

// CHECK-LABEL: func @full_replication
func.func @full_replication(
  // CHECK-SAME: %[[ARG:.*]]: tensor<2xi8>
  %arg0: tensor<2xi8>
// CHECK-SAME: -> tensor<2xi8> {
) -> tensor<2xi8> {
  %0 = mesh.shard %arg0 to <@mesh_1d, [[]]> : tensor<2xi8>
  %1 = mesh.shard %0 to <@mesh_1d, [[]]> annotate_for_users: tensor<2xi8>
  // CHECK: return %[[ARG]] : tensor<2xi8>
  return %1 : tensor<2xi8>
}

// CHECK-LABEL: func @move_split_axis
func.func @move_split_axis(
  // CHECK-SAME: %[[ARG:.*]]: tensor<1x2xi8>
  %arg0: tensor<2x2xi8>
// CHECK-SAME: -> tensor<2x1xi8> {
) -> tensor<2x2xi8> {
  // CHECK: %[[ALL_TO_ALL:.*]] = mesh.all_to_all %[[ARG]] on @mesh_1d
  // CHECK-SAME: mesh_axes = [0] split_axis = 1 concat_axis = 0 : tensor<1x2xi8> -> tensor<2x1xi8>
  %0 = mesh.shard %arg0 to <@mesh_1d, [[0]]> : tensor<2x2xi8>
  %1 = mesh.shard %0 to <@mesh_1d, [[], [0]]> annotate_for_users: tensor<2x2xi8>
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
  %0 = mesh.shard %arg0 to <@mesh_1d, [[0]]> : tensor<2xi8>
  %1 = mesh.shard %0 to <@mesh_1d, [[0]]> annotate_for_users: tensor<2xi8>
  // CHECK: %[[RES:.*]] = tosa.abs %[[ARG]] : (tensor<1xi8>) -> tensor<1xi8>
  %2 = tosa.abs %1 : (tensor<2xi8>) -> tensor<2xi8>
  %3 = mesh.shard %2 to <@mesh_1d, [[0]]> : tensor<2xi8>
  %4 = mesh.shard %3 to <@mesh_1d, [[0]]> annotate_for_users: tensor<2xi8>
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
  // We don't care about the whole resharding IR, just that it happens.
  // CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[ARG]][%{{.*}}] [1] [1]
  // CHECK-SAME: tensor<2xi8> to tensor<1xi8>
  %0 = mesh.shard %arg0 to <@mesh_1d, [[]]> : tensor<2xi8>
  %1 = mesh.shard %0 to <@mesh_1d, [[0]]> annotate_for_users: tensor<2xi8>
  // CHECK: %[[ABS:.*]] = tosa.abs %[[SLICE]] : (tensor<1xi8>) -> tensor<1xi8>
  %2 = tosa.abs %1 : (tensor<2xi8>) -> tensor<2xi8>
  // CHECK: %[[RES:.*]] = mesh.all_gather %[[ABS]] on @mesh_1d
  // CHECK-SAME: mesh_axes = [0] gather_axis = 0 : tensor<1xi8> -> tensor<2xi8>
  %3 = mesh.shard %2 to <@mesh_1d, [[0]]> : tensor<2xi8>
  %4 = mesh.shard %3 to <@mesh_1d, [[]]> annotate_for_users: tensor<2xi8>
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
  %arg0_sharded = mesh.shard %arg0 to <@mesh_1d, [[0]]> : tensor<2xi8>
  %op_arg0 = mesh.shard %arg0_sharded to <@mesh_1d, [[0]]> annotate_for_users: tensor<2xi8>
  %arg1_sharded = mesh.shard %arg1 to <@mesh_1d, [[0]]> : tensor<2xi8>
  %op_arg1 = mesh.shard %arg1_sharded to <@mesh_1d, [[0]]> annotate_for_users: tensor<2xi8>
  // CHECK: %[[RES:.*]] = tosa.add %[[ARG0]], %[[ARG1]] : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
  %op_res = tosa.add %op_arg0, %op_arg1 : (tensor<2xi8>, tensor<2xi8>) -> tensor<2xi8>
  %op_res_sharded = mesh.shard %op_res to <@mesh_1d, [[0]]> : tensor<2xi8>
  %res = mesh.shard %op_res_sharded to <@mesh_1d, [[0]]> annotate_for_users: tensor<2xi8>
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
  // We don't care about the whole resharding IR, just that it happens.
  // CHECK: %[[RESHARD1:.*]] = tensor.extract_slice %[[ARG]][%{{.*}}] [1] [1]
  // CHECK-SAME: tensor<2xi8> to tensor<1xi8>
  %0 = mesh.shard %arg0 to <@mesh_1d, [[]]> : tensor<2xi8>
  %1 = mesh.shard %0 to <@mesh_1d, [[0]]> annotate_for_users: tensor<2xi8>
  // CHECK: %[[ABS1:.*]] = tosa.abs %[[RESHARD1]] : (tensor<1xi8>) -> tensor<1xi8>
  %2 = tosa.abs %1 : (tensor<2xi8>) -> tensor<2xi8>
  // CHECK: %[[RESHARD2:.*]] = mesh.all_gather %[[ABS1]] on @mesh_1d
  // CHECK-SAME: mesh_axes = [0] gather_axis = 0 : tensor<1xi8> -> tensor<2xi8>
  %3 = mesh.shard %2 to <@mesh_1d, [[0]]> : tensor<2xi8>
  %4 = mesh.shard %3 to <@mesh_1d, [[]]> annotate_for_users: tensor<2xi8>
  // CHECK: %[[ABS2:.*]] = tosa.abs %[[RESHARD2]] : (tensor<2xi8>) -> tensor<2xi8>
  %5 = tosa.abs %4 : (tensor<2xi8>) -> tensor<2xi8>
  // CHECK: %[[RESHARD3:.*]] = tensor.extract_slice %[[ABS2]][%{{.*}}] [1] [1]
  // CHECK-SAME: tensor<2xi8> to tensor<1xi8>
  %6 = mesh.shard %5 to <@mesh_1d, [[]]> : tensor<2xi8>
  %7 = mesh.shard %6 to <@mesh_1d, [[0]]> annotate_for_users: tensor<2xi8>
  // CHECK: return %[[RESHARD3]] : tensor<1xi8>
  return %7 : tensor<2xi8>
}
