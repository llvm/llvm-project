// RUN: mlir-opt -allow-unregistered-dialect \
// RUN:   --pass-pipeline="builtin.module(func.func(mesh-spmdization,test-constant-fold))" \
// RUN:   %s | FileCheck %s

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

// CHECK-LABEL: func @sharding_triplet
func.func @sharding_triplet(
  // CHECK-SAME: %[[ARG:.*]]: tensor<1xf32>
  %arg0: tensor<2xf32>
// CHECK-SAME: ) -> tensor<2xf32> {
) -> tensor<2xf32> {
  // CHECK: %[[ALL_GATHER:.*]] = mesh.all_gather %[[ARG]] on @mesh_1d mesh_axes = [0] gather_axis = 0 : tensor<1xf32> -> tensor<2xf32>
  %sharding_annotated = mesh.shard %arg0 to <@mesh_1d, [[0]]> : tensor<2xf32>
  %sharding_annotated_0 = mesh.shard %sharding_annotated to <@mesh_1d, [[0]]> annotate_for_users : tensor<2xf32>
  %sharding_annotated_1 = mesh.shard %sharding_annotated_0 to <@mesh_1d, [[]]> : tensor<2xf32>
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
  // CHECK: %[[SLICE:.*]] = mesh.all_slice %[[ARG]] on @mesh_1d mesh_axes = [0] slice_axis = 0
  // CHECK-SAME: tensor<2xi8> -> tensor<1xi8>
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
  // CHECK: %[[RESHARD1:.*]] = mesh.all_slice %[[ARG]] on @mesh_1d mesh_axes = [0] slice_axis = 0
  // CHECK-SAME: tensor<2xi8> -> tensor<1xi8>
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
  // CHECK: %[[RESHARD3:.*]] = mesh.all_slice %[[ABS2]] on @mesh_1d mesh_axes = [0] slice_axis = 0 : 
  // CHECK-SAME: tensor<2xi8> -> tensor<1xi8>
  %6 = mesh.shard %5 to <@mesh_1d, [[]]> : tensor<2xi8>
  %7 = mesh.shard %6 to <@mesh_1d, [[0]]> annotate_for_users: tensor<2xi8>
  // CHECK: return %[[RESHARD3]] : tensor<1xi8>
  return %7 : tensor<2xi8>
}

// CHECK-LABEL: func @incomplete_sharding
func.func @incomplete_sharding(
  // CHECK-SAME: %[[ARG:.*]]: tensor<4x16xf32>
  %arg0: tensor<8x16xf32>
// CHECK-SAME: -> tensor<4x16xf32> {
) -> tensor<8x16xf32> {
  %0 = mesh.shard %arg0 to <@mesh_1d, [[0]]> annotate_for_users : tensor<8x16xf32>
  // CHECK: %[[RES:.*]] = tosa.sigmoid %[[ARG]] : (tensor<4x16xf32>) -> tensor<4x16xf32>
  %1 = tosa.sigmoid %0 : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %2 = mesh.shard %1 to <@mesh_1d, [[0]]> : tensor<8x16xf32>
  // CHECK: return %[[RES]] : tensor<4x16xf32>
  return %2 : tensor<8x16xf32>
}

mesh.mesh @mesh_1d_4(shape = 4)
// CHECK-LABEL: func @update_halo_constraint
func.func @update_halo_constraint(
  // CHECK-SAME: %[[IN1:[A-Za-z0-9_]+]]: tensor<11x16xi8>
  %in1: tensor<32x16xi8>
  // CHECK-SAME: -> tensor<11x16xi8> {
) -> tensor<32x16xi8> {
  // CHECK: %[[RES:.*]] = mesh.update_halo %[[IN1]] on @mesh_1d_4 halo_sizes = [2, 1] : (tensor<11x16xi8>) -> tensor<11x16xi8>
  %in1_sharded1 = mesh.shard %in1 to <@mesh_1d_4, [[0]] {<halo_sizes = [2, 1]>}> : tensor<32x16xi8>
  %in1_sharded2 = mesh.shard %in1_sharded1 to <@mesh_1d_4, [[0]] {<force = true halo_sizes = [2, 1]>}> annotate_for_users: tensor<32x16xi8>
  // CHECK: return %[[RES]] : tensor<11x16xi8>
  return %in1_sharded2 : tensor<32x16xi8>
}

// CHECK-LABEL: func @ew_chain_with_halo
func.func @ew_chain_with_halo(
  // CHECK-SAME: %[[IN1:[A-Za-z0-9_]+]]: tensor<5x16xf32>
  %arg0: tensor<8x16xf32>)
  // CHECK-SAME: -> tensor<5x16xf32>
   -> tensor<8x16xf32> {
  %sharding_annotated = mesh.shard %arg0 to <@mesh_1d_4, [[0]] {<halo_sizes = [2, 1]>}> annotate_for_users : tensor<8x16xf32>
  // CHECK: %[[TMP1:.*]] = tosa.tanh %[[IN1]] : (tensor<5x16xf32>) -> tensor<5x16xf32>
  %0 = tosa.tanh %sharding_annotated : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %sharding_annotated_0 = mesh.shard %0 to <@mesh_1d_4, [[0]] {<halo_sizes = [2, 1]>}> : tensor<8x16xf32>
  %sharding_annotated_1 = mesh.shard %sharding_annotated_0 to <@mesh_1d_4, [[0]] {<halo_sizes = [2, 1]>}> annotate_for_users : tensor<8x16xf32>
  // CHECK-NEXT: %[[TMP2:.*]] = tosa.abs %[[TMP1]] : (tensor<5x16xf32>) -> tensor<5x16xf32>
  %1 = tosa.abs %sharding_annotated_1 : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %sharding_annotated_2 = mesh.shard %1 to <@mesh_1d_4, [[0]] {<halo_sizes = [2, 1]>}> : tensor<8x16xf32>
  %sharding_annotated_4 = mesh.shard %sharding_annotated_2 to <@mesh_1d_4, [[0]] {<halo_sizes = [2, 1]>}> annotate_for_users : tensor<8x16xf32>
  // CHECK-NEXT: %[[TMP3:.*]] = tosa.negate %[[TMP2]] : (tensor<5x16xf32>) -> tensor<5x16xf32>
  %2 = tosa.negate %sharding_annotated_4 : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %sharding_annotated_5 = mesh.shard %2 to <@mesh_1d_4, [[0]] {<halo_sizes = [2, 1]>}> : tensor<8x16xf32>
  %sharding_annotated_6 = mesh.shard %sharding_annotated_5 to <@mesh_1d_4, [[0]] {<halo_sizes = [2, 1]>}> annotate_for_users : tensor<8x16xf32>
  // CHECK-NEXT: return %[[TMP3]] : tensor<5x16xf32>
  return %sharding_annotated_6 : tensor<8x16xf32>
}

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @stencil_with_halo
func.func @stencil_with_halo() -> () {
  %a = "xxx.empty"() : () -> tensor<32x16xf32>
  %sc1 = mesh.sharding_constraint sharded_dims = [] halo_sizes = [1, 2] : !mesh.constraint
  %sa = mesh.shard %a to <@mesh_1d_4, [[0]]>, !mesh.constraint = %sc1 : tensor<32x16xf32>
  %b = "xxx.empty"() : () -> tensor<8x16xf32>
  %sc2 = mesh.sharding_constraint sharded_dims = [1, 2, 3, 2] halo_sizes = [] : !mesh.constraint
  %sb = mesh.shard %b to <@mesh_1d_4, [[0]]>, !mesh.constraint = %sc2 : tensor<8x16xf32>

  %sai1 = mesh.shard %sa to <@mesh_1d_4, [[0]]>, !mesh.constraint = %sc1 annotate_for_users : tensor<32x16xf32>
  %v1 = "xxx.view"(%sa) {x = 1} : (tensor<32x16xf32>) -> tensor<8x16xf32>
  %sv1 = mesh.shard %v1 to <@mesh_1d_4, [[0]]>, !mesh.constraint = %sc2 : tensor<8x16xf32>

  %sai2 = mesh.shard %sa to <@mesh_1d_4, [[0]]>, !mesh.constraint = %sc1 annotate_for_users : tensor<32x16xf32>
  %v2 = "xxx.view"(%sa) {x = 2} : (tensor<32x16xf32>) -> tensor<8x16xf32>
  %sv2 = mesh.shard %v2 to <@mesh_1d_4, [[0]]>, !mesh.constraint = %sc2 : tensor<8x16xf32>
  
  %v1i = mesh.shard %sv1 to <@mesh_1d_4, [[0]]>, !mesh.constraint = %sc2 annotate_for_users : tensor<8x16xf32>
  %v2i = mesh.shard %sv2 to <@mesh_1d_4, [[0]]>, !mesh.constraint = %sc2 annotate_for_users : tensor<8x16xf32>
  %bo = mesh.shard %sb to <@mesh_1d_4, [[0]]>, !mesh.constraint = %sc2 annotate_for_users : tensor<8x16xf32>
  %r = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%v1i, %v2i : tensor<8x16xf32>, tensor<8x16xf32>) outs(%bo : tensor<8x16xf32>) {
    ^bb0(%in: f32, %in_56: f32, %out: f32):
      %47 = arith.addf %in, %in_56 : f32
      linalg.yield %47 : f32
    } -> tensor<8x16xf32>
  %sr = mesh.shard %r to <@mesh_1d_4, [[0]]>, !mesh.constraint = %sc2 : tensor<8x16xf32>

  %sai3 = mesh.shard %sa to <@mesh_1d_4, [[0]]>, !mesh.constraint = %sc1 annotate_for_users : tensor<32x16xf32>
  %sri = mesh.shard %sr to <@mesh_1d_4, [[0]]>, !mesh.constraint = %sc2 annotate_for_users : tensor<8x16xf32>
  "xxx.insert_slice"(%sai3, %sri) : (tensor<32x16xf32>, tensor<8x16xf32>) -> ()
  %sc3 = mesh.sharding_constraint sharded_dims = [] halo_sizes = [1, 2] force : !mesh.constraint
  %sai4 = mesh.shard %sa to <@mesh_1d_4, [[0]]>, !mesh.constraint = %sc3 : tensor<32x16xf32>

  return
}
// CHECK: %[[V0:.*]] = "xxx.empty"() : () -> tensor<11x16xf32>
// CHECK-NEXT: %[[V1:.*]] = "xxx.empty"() : () -> tensor<?x16xf32>
// CHECK-NEXT: %[[V2:.*]] = "xxx.view"([[V0]]) {x = 1 : i64} : (tensor<11x16xf32>) -> tensor<?x16xf32>
// CHECK-NEXT: %[[V3:.*]] = "xxx.view"([[V0]]) {x = 2 : i64} : (tensor<11x16xf32>) -> tensor<?x16xf32>
// CHECK-NEXT: %[[V4:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins([[V2]], [[V3]] : tensor<?x16xf32>, tensor<?x16xf32>) outs([[V1]] : tensor<?x16xf32>) {
// CHECK: "xxx.insert_slice"([[V0]], [[V4]]) : (tensor<11x16xf32>, tensor<?x16xf32>) -> ()
// CHECK-NEXT: %update_halo = mesh.update_halo [[V0]] on @mesh_1d_4 halo_sizes = [1, 2] : (tensor<11x16xf32>) -> tensor<11x16xf32>
// CHECK-NEXT: return
