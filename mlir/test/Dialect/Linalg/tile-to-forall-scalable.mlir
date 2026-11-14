// RUN: mlir-opt %s --transform-interpreter --test-transform-dialect-erase-schedule --cse --split-input-file | FileCheck %s

// Each target needs its own scalable tile size in the target's scope. Neither
// extent is divisible by the scalable tile size, so the last tile is bounded.
// CHECK-DAG: #[[$CEIL130:.+]] = affine_map<()[s0] -> (130 ceildiv s0)>
// CHECK-DAG: #[[$OFFSET:.+]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK-DAG: #[[$MIN130:.+]] = affine_map<(d0)[s0] -> (-d0 + 130, s0)>
// CHECK-DAG: #[[$CEIL17:.+]] = affine_map<()[s0] -> (17 ceildiv s0)>
// CHECK-LABEL: func.func @scalable_tail
// CHECK-SAME: %[[INPUT:.*]]: tensor<130x32xf32>, %[[INIT:.*]]: tensor<130x32xf32>
// CHECK: %[[VS0:.*]] = vector.vscale
// CHECK: %[[C8:.*]] = arith.constant 8 : index
// CHECK: %[[TILE0:.*]] = arith.muli %[[C8]], %[[VS0]] : index
// CHECK: %[[UB0:.*]] = affine.apply #[[$CEIL130]]()[%[[TILE0]]]
// CHECK: scf.forall (%[[I:.*]], %[[J:.*]]) in (%[[UB0]], 32) shared_outs(%[[OUT:.*]] = %[[INIT]])
// CHECK: %[[OFF:.*]] = affine.apply #[[$OFFSET]](%[[I]])[%[[TILE0]]]
// CHECK: %[[LEN:.*]] = affine.min #[[$MIN130]](%[[OFF]])[%[[TILE0]]]
// CHECK: tensor.extract_slice %[[INPUT]][%[[OFF]], %[[J]]] [%[[LEN]], 1] [1, 1]
// CHECK: tensor.extract_slice %[[OUT]][%[[OFF]], %[[J]]] [%[[LEN]], 1] [1, 1]
// CHECK: linalg.copy
// CHECK: tensor.parallel_insert_slice %{{.*}} into %[[OUT]][%[[OFF]], %[[J]]] [%[[LEN]], 1] [1, 1]
func.func @scalable_tail(%input: tensor<130x32xf32>, %init: tensor<130x32xf32>) -> tensor<130x32xf32> {
  %copy = linalg.copy ins(%input : tensor<130x32xf32>) outs(%init : tensor<130x32xf32>) -> tensor<130x32xf32>
  return %copy : tensor<130x32xf32>
}

// CHECK-LABEL: func.func @second_target
// CHECK: %[[VS1:.*]] = vector.vscale
// CHECK: %[[C8_1:.*]] = arith.constant 8 : index
// CHECK: %[[TILE1:.*]] = arith.muli %[[C8_1]], %[[VS1]] : index
// CHECK: %[[UB1:.*]] = affine.apply #[[$CEIL17]]()[%[[TILE1]]]
// CHECK: scf.forall (%[[I1:.*]], %{{.*}}) in (%[[UB1]], 32)
// CHECK: affine.apply #[[$OFFSET]](%[[I1]])[%[[TILE1]]]
// CHECK: linalg.copy
func.func @second_target(%input: tensor<17x32xf32>, %init: tensor<17x32xf32>) -> tensor<17x32xf32> {
  %copy = linalg.copy ins(%input : tensor<17x32xf32>) outs(%init : tensor<17x32xf32>) -> tensor<17x32xf32>
  return %copy : tensor<17x32xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %copy = transform.structured.match ops{["linalg.copy"]} in %root : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_forall %copy tile_sizes [[8], 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Both dimensions reuse one vscale value.
// CHECK-LABEL: func.func @two_scalable_dimensions
// CHECK: %[[VS:.*]] = vector.vscale
// CHECK: %[[TILE0:.*]] = arith.muli %{{.*}}, %[[VS]]
// CHECK: %[[TILE1:.*]] = arith.muli %{{.*}}, %[[VS]]
// CHECK-NOT: vector.vscale
// CHECK: %[[UB0:.*]] = affine.apply {{.*}}()[%[[TILE0]]]
// CHECK: %[[UB1:.*]] = affine.apply {{.*}}()[%[[TILE1]]]
// CHECK: scf.forall (%[[I:.*]], %[[J:.*]]) in (%[[UB0]], %[[UB1]])
// CHECK: %[[OFF0:.*]] = affine.apply {{.*}}(%[[I]])[%[[TILE0]]]
// CHECK: %[[OFF1:.*]] = affine.apply {{.*}}(%[[J]])[%[[TILE1]]]
// CHECK: %[[LEN0:.*]] = affine.min {{.*}}(%[[OFF0]])[%[[TILE0]]]
// CHECK: %[[LEN1:.*]] = affine.min {{.*}}(%[[OFF1]])[%[[TILE1]]]
// CHECK: tensor.extract_slice %{{.*}}[%[[OFF0]], %[[OFF1]]] [%[[LEN0]], %[[LEN1]]] [1, 1]
func.func @two_scalable_dimensions(%input: tensor<130x33xf32>, %init: tensor<130x33xf32>) -> tensor<130x33xf32> {
  %copy = linalg.copy ins(%input : tensor<130x33xf32>) outs(%init : tensor<130x33xf32>) -> tensor<130x33xf32>
  return %copy : tensor<130x33xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %copy = transform.structured.match ops{["linalg.copy"]} in %root : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_forall %copy tile_sizes [[8], [4]] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Scalable sizes may be mixed with ordinary dynamic sizes.
// CHECK-LABEL: func.func @mixed_dynamic
// CHECK: %[[DYNAMIC:.*]] = call @tile_size
// CHECK: %[[VS:.*]] = vector.vscale
// CHECK: %[[TILE:.*]] = arith.muli %{{.*}}, %[[VS]]
// CHECK: %[[UB0:.*]] = affine.apply {{.*}}()[%[[TILE]]]
// CHECK: %[[UB1:.*]] = affine.apply {{.*}}()[%[[DYNAMIC]]]
// CHECK: scf.forall (%[[I:.*]], %[[J:.*]]) in (%[[UB0]], %[[UB1]])
// CHECK: %[[OFF0:.*]] = affine.apply {{.*}}(%[[I]])[%[[TILE]]]
// CHECK: %[[OFF1:.*]] = affine.apply {{.*}}(%[[J]])[%[[DYNAMIC]]]
// CHECK: %[[LEN0:.*]] = affine.min {{.*}}(%[[OFF0]])[%[[TILE]]]
// CHECK: %[[LEN1:.*]] = affine.min {{.*}}(%[[OFF1]])[%[[DYNAMIC]]]
// CHECK: tensor.extract_slice %{{.*}}[%[[OFF0]], %[[OFF1]]] [%[[LEN0]], %[[LEN1]]] [1, 1]
func.func @mixed_dynamic(%input: tensor<130x33xf32>, %init: tensor<130x33xf32>) -> tensor<130x33xf32> {
  %size = func.call @tile_size() : () -> index
  %copy = linalg.copy ins(%input : tensor<130x33xf32>) outs(%init : tensor<130x33xf32>) -> tensor<130x33xf32>
  return %copy : tensor<130x33xf32>
}
func.func private @tile_size() -> index

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %size = transform.structured.match ops{["func.call"]} in %root : (!transform.any_op) -> !transform.any_op
    %copy = transform.structured.match ops{["linalg.copy"]} in %root : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_forall %copy tile_sizes [[8], %size] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Zero remains untiled even when marked scalable.
// CHECK-LABEL: func.func @scalable_zero
// CHECK-NOT: vector.vscale
// CHECK: scf.forall (%{{.*}}) in (8)
// CHECK: tensor.extract_slice %{{.*}}[0, %{{.*}}] [130, 4] [1, 1]
func.func @scalable_zero(%input: tensor<130x32xf32>, %init: tensor<130x32xf32>) -> tensor<130x32xf32> {
  %copy = linalg.copy ins(%input : tensor<130x32xf32>) outs(%init : tensor<130x32xf32>) -> tensor<130x32xf32>
  return %copy : tensor<130x32xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %copy = transform.structured.match ops{["linalg.copy"]} in %root : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_forall %copy tile_sizes [[0], 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Dynamic extents, buffer semantics and device mapping remain supported.
// CHECK-LABEL: func.func @scalable_memref
// CHECK: memref.dim
// CHECK: vector.vscale
// CHECK: arith.muli
// CHECK: scf.forall (%{{.*}})
// CHECK: memref.subview
// CHECK: linalg.copy
// CHECK: mapping = [#gpu.thread<x>]
func.func @scalable_memref(%input: memref<?x32xf32>, %init: memref<?x32xf32>) {
  linalg.copy ins(%input : memref<?x32xf32>) outs(%init : memref<?x32xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %copy = transform.structured.match ops{["linalg.copy"]} in %root : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_forall %copy tile_sizes [[8], 0] (mapping = [#gpu.thread<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// The pack inner tile and the forall tile size each scale independently.
#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[$PACK_OFFSET:.+]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK-DAG: #[[$PACK_MIN:.+]] = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
// CHECK-DAG: #[[$SOURCE_OFFSET:.+]] = affine_map<(d0)[s0, s1] -> ((d0 * s1) * s0)>
// CHECK-LABEL: func.func @pack_scalable_prod
// CHECK-SAME: %[[SOURCE:.*]]: tensor<1024x32xf32>
// CHECK: %[[EIGHT:.*]] = arith.constant 8 : index
// CHECK: %[[SCALE:.*]] = vector.vscale
// CHECK: %[[SCALED_TILE:.*]] = arith.muli %[[SCALE]], %[[EIGHT]] : index
// CHECK: scf.forall (%[[PI:.*]], %[[PJ:.*]]) in (%{{.*}}, 32) shared_outs(%[[DEST:.*]] = %{{.*}})
// CHECK: %[[POFF:.*]] = affine.apply #[[$PACK_OFFSET]](%[[PI]])[%[[SCALED_TILE]]]
// CHECK: %[[PLEN:.*]] = affine.min #[[$PACK_MIN]](%[[POFF]])[%{{.*}}, %[[SCALED_TILE]]]
// CHECK: %[[SOFF:.*]] = affine.apply #[[$SOURCE_OFFSET]](%[[PI]])[%[[SCALED_TILE]], %[[SCALED_TILE]]]
// CHECK: %[[SLEN:.*]] = affine.apply #[[$PACK_OFFSET]](%[[PLEN]])[%[[SCALED_TILE]]]
// CHECK: tensor.extract_slice %[[SOURCE]][%[[SOFF]], %[[PJ]]] [%[[SLEN]], 1] [1, 1]
// CHECK: %[[PRODUCER:.*]] = linalg.generic
// CHECK: %[[PACK_DEST:.*]] = tensor.extract_slice %[[DEST]][%[[POFF]], %[[PJ]], 0, 0] [%[[PLEN]], 1, %{{.*}}, 1] [1, 1, 1, 1]
// CHECK: %[[PACK:.*]] = linalg.pack %[[PRODUCER]] inner_dims_pos = [0, 1] inner_tiles = [%[[SCALED_TILE]], 1] into %[[PACK_DEST]]
// CHECK: tensor.parallel_insert_slice %[[PACK]] into %[[DEST]][%[[POFF]], %[[PJ]], 0, 0] [%[[PLEN]], 1, %{{.*}}, 1] [1, 1, 1, 1]
func.func @pack_scalable_prod(%input: tensor<1024x32xf32>) -> tensor<?x32x?x1xf32> {
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %vscale, %c8 : index
  %outer_size = affine.apply affine_map<()[s0] -> (1024 ceildiv s0)>()[%c8_vscale]
  %pack_init = tensor.empty(%outer_size, %c8_vscale) : tensor<?x32x?x1xf32>
  %producer_init = tensor.empty() : tensor<1024x32xf32>
  %producer = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%input : tensor<1024x32xf32>) outs(%producer_init : tensor<1024x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %sum = arith.addf %in, %in : f32
    linalg.yield %sum : f32
  } -> tensor<1024x32xf32>
  %pack = linalg.pack %producer inner_dims_pos = [0, 1] inner_tiles = [%c8_vscale, 1] into %pack_init : tensor<1024x32xf32> -> tensor<?x32x?x1xf32>
  return %pack : tensor<?x32x?x1xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %module : (!transform.any_op) -> !transform.any_op
    %pack = transform.structured.match ops{["linalg.pack"]} in %module : (!transform.any_op) -> !transform.any_op
    %tiled_pack, %loops = transform.structured.tile_using_forall %pack tile_sizes [[8], 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %generic into %loops : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
