// RUN: mlir-opt \
// RUN:  --pass-pipeline="builtin.module(func.func(mesh-spmdization,test-constant-fold))" \
// RUN:  --split-input-file \
// RUN:  %s | FileCheck %s

// CHECK: #[[$MAP_IDENTITY_1D:.*]] = affine_map<(d0) -> (d0)>
#map_identity_1d = affine_map<(d0) -> (d0)>

mesh.mesh @mesh_1d(shape = 2)

// CHECK-LABEL: func @elementwise_static_1d_mesh_static_1d_tensor
func.func @elementwise_static_1d_mesh_static_1d_tensor(
  // CHECK-SAME: %[[IN1:[A-Za-z0-9_]+]]: tensor<1xi8>,
  %in1: tensor<2xi8>,
  // CHECK-SAME: %[[IN2:[A-Za-z0-9_]+]]: tensor<1xi8>,
  %in2: tensor<2xi8>,
  // CHECK-SAME: %[[DPS_OUT:[A-Za-z0-9_]+]]: tensor<1xi8>
  %dps_out: tensor<2xi8>
// CHECK-SAME: -> tensor<1xi8> {
) -> tensor<2xi8> {
  %in1_shared1 = mesh.shard %in1 to <@mesh_1d, [[0]]> : tensor<2xi8>
  %in1_shared2 = mesh.shard %in1_shared1 to <@mesh_1d, [[0]]> annotate_for_users: tensor<2xi8>
  %in2_shared1 = mesh.shard %in2 to <@mesh_1d, [[0]]> : tensor<2xi8>
  %in2_shared2 = mesh.shard %in2_shared1 to <@mesh_1d, [[0]]> annotate_for_users: tensor<2xi8>
  %dps_out_shared1 = mesh.shard %dps_out to <@mesh_1d, [[0]]> : tensor<2xi8>
  %dps_out_shared2 = mesh.shard %dps_out_shared1 to <@mesh_1d, [[0]]> annotate_for_users: tensor<2xi8>
  // CHECK: %[[RES:.*]] = linalg.generic {
  // CHECK-SAME: indexing_maps = [#[[$MAP_IDENTITY_1D]], #[[$MAP_IDENTITY_1D]], #[[$MAP_IDENTITY_1D]]],
  // CHECK-SAME: iterator_types = ["parallel"]}
  // CHECK-SAME: ins(%[[IN1]], %[[IN2]] : tensor<1xi8>, tensor<1xi8>)
  // CHECK-SAME: outs(%[[DPS_OUT]] : tensor<1xi8>) {
  %res = linalg.generic {
      indexing_maps = [#map_identity_1d, #map_identity_1d, #map_identity_1d],
      iterator_types = ["parallel"]
    } ins(%in1_shared2, %in2_shared2 : tensor<2xi8>, tensor<2xi8>)
      outs(%dps_out_shared2 : tensor<2xi8>) {
    ^bb0(%in1_scalar: i8, %in2_scalar: i8, %out: i8):
      %res_scalar = arith.muli %in1_scalar, %in2_scalar : i8
      linalg.yield %res_scalar : i8
    } -> tensor<2xi8>
  %res_shared1 = mesh.shard %res to <@mesh_1d, [[0]]> : tensor<2xi8>
  %res_shared2 = mesh.shard %res_shared1 to <@mesh_1d, [[0]]> annotate_for_users: tensor<2xi8>
  // CHECK: return %[[RES]] : tensor<1xi8>
  return %res_shared2 : tensor<2xi8>
}

// -----

mesh.mesh @mesh_1d(shape = 4)

// CHECK-LABEL: func @matmul_1d_mesh_static_tensors_parallel_iterator_sharding
func.func @matmul_1d_mesh_static_tensors_parallel_iterator_sharding(
  // CHECK-SAME: %[[IN1:[A-Za-z0-9_]+]]: tensor<1x3xi8>,
  %in1: tensor<4x3xi8>,
// CHECK-SAME: %[[IN2:[A-Za-z0-9_]+]]: tensor<3x8xi8>,
  %in2: tensor<3x8xi8>,
// CHECK-SAME: %[[DPS_OUT:[A-Za-z0-9_]+]]: tensor<1x8xi8>
  %dps_out: tensor<4x8xi8>
// CHECK-SAME: -> tensor<1x8xi8> {
) -> tensor<4x8xi8> {
  %in1_shared1 = mesh.shard %in1 to <@mesh_1d, [[0]]> : tensor<4x3xi8>
  %in1_shared2 = mesh.shard %in1_shared1 to <@mesh_1d, [[0]]> annotate_for_users: tensor<4x3xi8>
  %in2_shared1 = mesh.shard %in2 to <@mesh_1d, [[]]> : tensor<3x8xi8>
  %in2_shared2 = mesh.shard %in2_shared1 to <@mesh_1d, [[]]> annotate_for_users: tensor<3x8xi8>
  %dps_out_shared1 = mesh.shard %dps_out to <@mesh_1d, [[0]]> : tensor<4x8xi8>
  %dps_out_shared2 = mesh.shard %dps_out_shared1 to <@mesh_1d, [[0]]> annotate_for_users: tensor<4x8xi8>
  // CHECK: %[[RES:.*]] = linalg.matmul
  // CHECK-SAME: ins(%[[IN1]], %[[IN2]] : tensor<1x3xi8>, tensor<3x8xi8>)
  // CHECK-SAME: outs(%[[DPS_OUT]] : tensor<1x8xi8>)
  // CHECK-SAME: -> tensor<1x8xi8>
  %res = linalg.matmul ins(%in1_shared2, %in2_shared2 : tensor<4x3xi8>, tensor<3x8xi8>)
      outs(%dps_out_shared2 : tensor<4x8xi8>) -> tensor<4x8xi8>
  %res_shared1 = mesh.shard %res to <@mesh_1d, [[0]]> : tensor<4x8xi8>
  %res_shared2 = mesh.shard %res_shared1 to <@mesh_1d, [[0]]> annotate_for_users: tensor<4x8xi8>
  // CHECK: return %[[RES]] : tensor<1x8xi8>
  return %res_shared2 : tensor<4x8xi8>
}

// -----

mesh.mesh @mesh_1d(shape = 3)

// CHECK-LABEL: func @matmul_1d_mesh_static_tensors_reduction_iterator_sharding
func.func @matmul_1d_mesh_static_tensors_reduction_iterator_sharding(
  // CHECK-SAME: %[[IN1:[A-Za-z0-9_]+]]: tensor<4x2xi8>,
  %in1: tensor<4x6xi8>,
// CHECK-SAME: %[[IN2:[A-Za-z0-9_]+]]: tensor<2x8xi8>,
  %in2: tensor<6x8xi8>,
// CHECK-SAME: %[[DPS_OUT:[A-Za-z0-9_]+]]: tensor<4x8xi8>
  %dps_out: tensor<4x8xi8>
// CHECK-SAME: -> tensor<4x8xi8> {
) -> tensor<4x8xi8> {
  %in1_shared1 = mesh.shard %in1 to <@mesh_1d, [[], [0]]> : tensor<4x6xi8>
  %in1_shared2 = mesh.shard %in1_shared1 to <@mesh_1d, [[], [0]]> annotate_for_users: tensor<4x6xi8>
  %in2_shared1 = mesh.shard %in2 to <@mesh_1d, [[0]]> : tensor<6x8xi8>
  %in2_shared2 = mesh.shard %in2_shared1 to <@mesh_1d, [[0]]> annotate_for_users: tensor<6x8xi8>
  %dps_out_shared1 = mesh.shard %dps_out to <@mesh_1d, [[]]> : tensor<4x8xi8>
  %dps_out_shared2 = mesh.shard %dps_out_shared1 to <@mesh_1d, [[]]> annotate_for_users: tensor<4x8xi8>
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C0_I8:.*]] = arith.constant 0 : i8
  // CHECK-DAG:  %[[PROCESS_IDX:.*]] = mesh.process_multi_index on @mesh_1d axes = [0] : index
  // CHECK-DAG:  %[[MESH_SIZE:.*]] = mesh.mesh_shape @mesh_1d axes = [0] : index
  // CHECK:      %[[DPS_INIT_OPERAND_CONDITION:.*]] = arith.cmpi eq, %[[PROCESS_IDX]], %[[C0]] : index
  // CHECK:      %[[DPS_INIT_OPERAND:.*]] = scf.if %[[DPS_INIT_OPERAND_CONDITION]] -> (tensor<4x8xi8>) {
  // CHECK:        scf.yield %[[DPS_OUT]] : tensor<4x8xi8>
  // CHECK:      } else {
  // CHECK-DAG:    %[[EMPTY_TENSOR:.*]] = tensor.empty() : tensor<4x8xi8>
  // CHECK:        %[[NEUTRAL_ELEMENT_FILLED_TENSOR:.*]] = linalg.fill ins(%[[C0_I8]] : i8)
  // CHECK-SAME:       outs(%[[EMPTY_TENSOR]] : tensor<4x8xi8>) -> tensor<4x8xi8>
  // CHECK:        scf.yield %[[NEUTRAL_ELEMENT_FILLED_TENSOR]] : tensor<4x8xi8>
  // CHECK:      }
  // CHECK:      %[[SHARDED_MATMUL:.*]] = linalg.matmul ins(%[[IN1]], %[[IN2]] : tensor<4x2xi8>, tensor<2x8xi8>)
  // CHECK-SAME:     outs(%[[DPS_INIT_OPERAND]] : tensor<4x8xi8>) -> tensor<4x8xi8>
  // CHECK:      %[[ALL_REDUCED:.*]] = mesh.all_reduce %[[SHARDED_MATMUL]] on @mesh_1d mesh_axes = [0] : tensor<4x8xi8> -> tensor<4x8xi8>
  %res = linalg.matmul ins(%in1_shared2, %in2_shared2 : tensor<4x6xi8>, tensor<6x8xi8>)
      outs(%dps_out_shared2 : tensor<4x8xi8>) -> tensor<4x8xi8>
  %res_shared1 = mesh.shard %res to <@mesh_1d, [[]]> : tensor<4x8xi8>
  %res_shared2 = mesh.shard %res_shared1 to <@mesh_1d, [[]]> annotate_for_users: tensor<4x8xi8>
  // CHECK:      return %[[ALL_REDUCED]] : tensor<4x8xi8>
  return %res_shared2 : tensor<4x8xi8>
}

// -----

mesh.mesh @mesh_1d(shape = 3)

// CHECK-LABEL: func @matmul_1d_mesh_static_tensors_reduction_iterator_sharding_with_partial_result
func.func @matmul_1d_mesh_static_tensors_reduction_iterator_sharding_with_partial_result(
  // CHECK-SAME: %[[IN1:[A-Za-z0-9_]+]]: tensor<4x2xi8>,
  %in1: tensor<4x6xi8>,
// CHECK-SAME: %[[IN2:[A-Za-z0-9_]+]]: tensor<2x8xi8>,
  %in2: tensor<6x8xi8>,
// CHECK-SAME: %[[DPS_OUT:[A-Za-z0-9_]+]]: tensor<4x8xi8>
  %dps_out: tensor<4x8xi8>
// CHECK-SAME: -> tensor<4x8xi8> {
) -> tensor<4x8xi8> {
  %in1_shared1 = mesh.shard %in1 to <@mesh_1d, [[], [0]]> : tensor<4x6xi8>
  %in1_shared2 = mesh.shard %in1_shared1 to <@mesh_1d, [[], [0]]> annotate_for_users: tensor<4x6xi8>
  %in2_shared1 = mesh.shard %in2 to <@mesh_1d, [[0]]> : tensor<6x8xi8>
  %in2_shared2 = mesh.shard %in2_shared1 to <@mesh_1d, [[0]]> annotate_for_users: tensor<6x8xi8>
  %dps_out_shared1 = mesh.shard %dps_out to <@mesh_1d, [[]]> : tensor<4x8xi8>
  %dps_out_shared2 = mesh.shard %dps_out_shared1 to <@mesh_1d, [[]]> annotate_for_users: tensor<4x8xi8>
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C0_I8:.*]] = arith.constant 0 : i8
  // CHECK-DAG:  %[[PROCESS_IDX:.*]] = mesh.process_multi_index on @mesh_1d axes = [0] : index
  // CHECK-DAG:  %[[MESH_SIZE:.*]] = mesh.mesh_shape @mesh_1d axes = [0] : index
  // CHECK:      %[[DPS_INIT_OPERAND_CONDITION:.*]] = arith.cmpi eq, %[[PROCESS_IDX]], %[[C0]] : index
  // CHECK:      %[[DPS_INIT_OPERAND:.*]] = scf.if %[[DPS_INIT_OPERAND_CONDITION]] -> (tensor<4x8xi8>) {
  // CHECK:        scf.yield %[[DPS_OUT]] : tensor<4x8xi8>
  // CHECK:      } else {
  // CHECK-DAG:    %[[EMPTY_TENSOR:.*]] = tensor.empty() : tensor<4x8xi8>
  // CHECK:        %[[NEUTRAL_ELEMENT_FILLED_TENSOR:.*]] = linalg.fill ins(%[[C0_I8]] : i8)
  // CHECK-SAME:       outs(%[[EMPTY_TENSOR]] : tensor<4x8xi8>) -> tensor<4x8xi8>
  // CHECK:        scf.yield %[[NEUTRAL_ELEMENT_FILLED_TENSOR]] : tensor<4x8xi8>
  // CHECK:      }
  // CHECK:      %[[SHARDED_MATMUL:.*]] = linalg.matmul ins(%[[IN1]], %[[IN2]] : tensor<4x2xi8>, tensor<2x8xi8>)
  // CHECK-SAME:     outs(%[[DPS_INIT_OPERAND]] : tensor<4x8xi8>) -> tensor<4x8xi8>
  %res = linalg.matmul ins(%in1_shared2, %in2_shared2 : tensor<4x6xi8>, tensor<6x8xi8>)
      outs(%dps_out_shared2 : tensor<4x8xi8>) -> tensor<4x8xi8>
  %res_shared1 = mesh.shard %res to <@mesh_1d, [[]], partial = sum[0]> : tensor<4x8xi8>
  %res_shared2 = mesh.shard %res_shared1 to <@mesh_1d, [[]], partial = sum[0]> annotate_for_users: tensor<4x8xi8>
  // CHECK:      return %[[SHARDED_MATMUL]] : tensor<4x8xi8>
  return %res_shared2 : tensor<4x8xi8>
}
