// RUN: mlir-opt %s --transform-interpreter=entry-point=aligned -split-input-file -canonicalize -cse | FileCheck %s --check-prefixes=ALIGNED
// RUN: mlir-opt %s --transform-interpreter=entry-point=equal -split-input-file -canonicalize -cse | FileCheck %s --check-prefixes=EQUAL
// RUN: mlir-opt %s --transform-interpreter=entry-point=unaligned -split-input-file -canonicalize -cse | FileCheck %s --check-prefixes=UNALIGNED
// RUN: mlir-opt %s -transform-interpreter -split-input-file -canonicalize -cse | FileCheck %s --check-prefixes=CHECK

// Producer fusion of a scalable `linalg.unpack` into a tiled consumer.
// The `inner_tile_alignments` hint is a per-dimension keyword list:
//   - `Equal`:    the loop tile size equals the pack/unpack inner tile size.
//   - `Multiple`: the loop tile size is an integer multiple of the inner tile.
//   - `Unknown`:  the default; nothing is asserted for that dimension.

// The elemwise consumer is tiled three ways from the same input IR: aligned, equal, and unaligned.

// ALIGNED-DAG: #[[$MAP_CEILDIV:.+]] = affine_map<(d0)[s0] -> (d0 ceildiv s0)>
func.func @unpack_elemwise_scalable(%arg0: tensor<4x8x?x?xf32>, %arg1: tensor<?x?xf32>, %arg2 : index, %arg3 : index) -> tensor<?x?xf32> {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
  // 16 * vscale is only consumed as a loop tile size by `@aligned`; `@equal` and
  // `@unaligned` leave it dead and it is folded away.
  %c16_vscale = arith.muli %c16, %vscale : index
  %0 = tensor.empty(%arg2, %arg3) : tensor<?x?xf32>
  %1 = linalg.unpack %arg0 inner_dims_pos = [0, 1]
      inner_tiles = [%c8_vscale, %c4_vscale] into %0
      : tensor<4x8x?x?xf32> -> tensor<?x?xf32>
  %2 = linalg.exp ins(%1: tensor<?x?xf32>)
                       outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  // Aligned: loop tile sizes (16 * vscale, 8 * vscale) are integer multiples of
  // the inner tiles, asserted via `Multiple`. The outer dims become
  // `ceilDiv(loop tile, inner tile)`.
  transform.named_sequence @aligned(%arg1: !transform.any_op {transform.readonly}) {
    %exp = transform.structured.match ops{["linalg.exp"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %mulis = transform.structured.match ops{["arith.muli"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %h8, %h4, %h16 = transform.split_handle %mulis : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %tiled, %loops:2 = transform.structured.fuse %exp tile_sizes [%h16, %h8] interchange [0, 1]
      inner_tile_alignments = [Multiple, Multiple]
      : (!transform.any_op, !transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
  // ALIGNED-LABEL: func.func @unpack_elemwise_scalable
  //   ALIGNED-DAG:    %[[C4:.*]] = arith.constant 4 : index
  //   ALIGNED-DAG:    %[[C8:.*]] = arith.constant 8 : index
  //   ALIGNED-DAG:    %[[C16:.*]] = arith.constant 16 : index
  //   ALIGNED-DAG:    %[[VSCALE:.*]] = vector.vscale
  //   ALIGNED-DAG:    %[[C8_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C8]] : index
  //   ALIGNED-DAG:    %[[C4_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C4]] : index
  //   ALIGNED-DAG:    %[[C16_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C16]] : index
  //       ALIGNED:    %[[RES:.*]] = scf.for {{.*}} step %[[C16_VSCALE]]
  //       ALIGNED:      scf.for {{.*}} step %[[C8_VSCALE]]
  //       ALIGNED:        %[[OUTER_DIM0:.*]] = affine.apply #[[$MAP_CEILDIV]](%{{.*}})[%[[C8_VSCALE]]]
  //       ALIGNED:        %[[OUTER_DIM1:.*]] = affine.apply #[[$MAP_CEILDIV]](%{{.*}})[%[[C4_VSCALE]]]
  //       ALIGNED:        %[[SRC:.*]] = tensor.extract_slice %{{.*}}[%{{.*}}, %{{.*}}, 0, 0] [%[[OUTER_DIM0]], %[[OUTER_DIM1]], %[[C8_VSCALE]], %[[C4_VSCALE]]]
  //       ALIGNED:        %[[UNPACK:.*]] = linalg.unpack %[[SRC]]
  //  ALIGNED-SAME:            tensor<?x?x?x?xf32> -> tensor<?x?xf32>
  //   ALIGNED-NOT:         tensor.extract_slice %[[UNPACK]]
  //       ALIGNED:        linalg.exp ins(%[[UNPACK]]
  //       ALIGNED:      scf.yield
  //       ALIGNED:    scf.yield
  //       ALIGNED:    return %[[RES]]

  // Equal: loop tile sizes (8 * vscale, 4 * vscale) equal the inner tiles,
  // asserted via `Equal`; the fused unpack's outer dims collapse to 1.
  transform.named_sequence @equal(%arg1: !transform.any_op {transform.readonly}) {
    %exp = transform.structured.match ops{["linalg.exp"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %mulis = transform.structured.match ops{["arith.muli"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %h8, %h4, %h16 = transform.split_handle %mulis : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %tiled, %loops:2 = transform.structured.fuse %exp tile_sizes [%h8, %h4] interchange [0, 1]
      inner_tile_alignments = [Equal, Equal]
      : (!transform.any_op, !transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
  //   EQUAL-LABEL: func.func @unpack_elemwise_scalable
  //   EQUAL-DAG:    %[[C4:.*]] = arith.constant 4 : index
  //   EQUAL-DAG:    %[[C8:.*]] = arith.constant 8 : index
  //   EQUAL-DAG:    %[[VSCALE:.*]] = vector.vscale
  //   EQUAL-DAG:    %[[C8_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C8]] : index
  //   EQUAL-DAG:    %[[C4_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C4]] : index
  //       EQUAL:    %[[RES:.*]] = scf.for {{.*}} step %[[C8_VSCALE]]
  //       EQUAL:      scf.for {{.*}} step %[[C4_VSCALE]]
  //       EQUAL:        %[[UNPACK:.*]] = linalg.unpack
  //  EQUAL-SAME:            tensor<1x1x?x?xf32> -> tensor<?x?xf32>
  //   EQUAL-NOT:         tensor.extract_slice %[[UNPACK]]
  //       EQUAL:        linalg.exp ins(%[[UNPACK]]
  //       EQUAL:      scf.yield
  //       EQUAL:    scf.yield
  //       EQUAL:    return %[[RES]]

  // Unaligned: static tile sizes (7, 5) are not aligned to the scalable inner
  // tiles and no hint is passed, so the fused unpack over-computes and a trailing
  // extract_slice recovers the needed slice.
  transform.named_sequence @unaligned(%arg1: !transform.any_op {transform.readonly}) {
    %exp = transform.structured.match ops{["linalg.exp"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tiled, %loops:2 = transform.structured.fuse %exp tile_sizes [7, 5] interchange [0, 1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
  //   UNALIGNED-LABEL: func.func @unpack_elemwise_scalable
  //   UNALIGNED-DAG:    %[[C7:.*]] = arith.constant 7 : index
  //   UNALIGNED-DAG:    %[[C5:.*]] = arith.constant 5 : index
  //       UNALIGNED:    %[[RES:.*]] = scf.for {{.*}} step %[[C7]]
  //       UNALIGNED:      scf.for {{.*}} step %[[C5]]
  //       UNALIGNED:        %[[UNPACK:.*]] = linalg.unpack
  //  UNALIGNED-SAME:            tensor<?x?x?x?xf32> -> tensor<?x?xf32>
  //       UNALIGNED:        %[[EXTRACT:.*]] = tensor.extract_slice %[[UNPACK]]
  //   UNALIGNED-NOT:        linalg.exp ins(%[[UNPACK]]
  //       UNALIGNED:        linalg.exp ins(%[[EXTRACT]]
  //       UNALIGNED:      scf.yield
  //       UNALIGNED:    scf.yield
  //       UNALIGNED:    return %[[RES]]

  // Dummy entry point so the default RUN line resolves here.
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    transform.yield
  }
}

// -----

// Fusing a scalable `linalg.unpack` producer into a containing `scf.forall`. The
// same input IR is fused three ways: aligned, equal, and unaligned.

func.func @fuse_unpack_into_containing(
    %src: tensor<?x?x?x?xf32>, %unpack_empty: tensor<?x?xf32>,
    %out: tensor<?x?xf32>, %ub0: index, %ub1: index) -> tensor<?x?xf32> {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  // Inner tile sizes of the unpack.
  %c8_vscale = arith.muli %c8, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
  // Containing-loop tile sizes: numerically the same as the inner tiles, but
  // deliberately distinct SSA values so the `@aligned` (`Multiple`) path cannot
  // statically fold `ceilDiv(tile, inner)` to 1 and keeps a dynamic outer dim.
  %c8_vscale_step = arith.muli %c8, %vscale : index
  %c4_vscale_step = arith.muli %c4, %vscale : index
  %unpack = linalg.unpack %src inner_dims_pos = [0, 1]
      inner_tiles = [%c8_vscale, %c4_vscale] into %unpack_empty
      : tensor<?x?x?x?xf32> -> tensor<?x?xf32>
  // Loop tile sizes (8 * vscale, 4 * vscale) equal the inner tile sizes of the
  // consumer `linalg.unpack` in value.
  %res = scf.forall (%i, %j) = (0, 0) to (%ub0, %ub1) step (%c8_vscale_step, %c4_vscale_step)
      shared_outs(%o = %out) -> (tensor<?x?xf32>) {
    %slice = tensor.extract_slice %unpack[%i, %j] [%c8_vscale_step, %c4_vscale_step] [1, 1]
        : tensor<?x?xf32> to tensor<?x?xf32>
    %oslice = tensor.extract_slice %o[%i, %j] [%c8_vscale_step, %c4_vscale_step] [1, 1]
        : tensor<?x?xf32> to tensor<?x?xf32>
    %0 = linalg.exp ins(%slice : tensor<?x?xf32>) outs(%oslice : tensor<?x?xf32>) -> tensor<?x?xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %0 into %o[%i, %j] [%c8_vscale_step, %c4_vscale_step] [1, 1]
          : tensor<?x?xf32> into tensor<?x?xf32>
    }
  }
  return %res : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  // Equal tiles: the fused unpack collapses to a 1x1 outer tile.
  transform.named_sequence @equal(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fused, %newc = transform.structured.fuse_into_containing_op %0 into %1
        inner_tile_alignments = [Equal, Equal]
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
  //   EQUAL-LABEL: func.func @fuse_unpack_into_containing
  //       EQUAL:   scf.forall
  //       EQUAL:     %[[UNPACK:.+]] = linalg.unpack
  //  EQUAL-SAME:         : tensor<1x1x?x?xf32> -> tensor<?x?xf32>
  //   EQUAL-NOT:     tensor.extract_slice %[[UNPACK]]
  //       EQUAL:     linalg.exp ins(%[[UNPACK]]

  // No hint: general (unaligned) tiling with a trailing result slice.
  transform.named_sequence @unaligned(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fused, %newc = transform.structured.fuse_into_containing_op %0 into %1
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
  // UNALIGNED-LABEL: func.func @fuse_unpack_into_containing
  //     UNALIGNED:   scf.forall
  //     UNALIGNED:     %[[UNPACK:.+]] = linalg.unpack
  // UNALIGNED-SAME:         : tensor<?x?x?x?xf32> -> tensor<?x?xf32>
  //     UNALIGNED:     %[[EXTRACTED:.+]] = tensor.extract_slice %[[UNPACK]]
  // UNALIGNED-NOT:     linalg.exp ins(%[[UNPACK]]
  //     UNALIGNED:     linalg.exp ins(%[[EXTRACTED]]

  // Aligned: the `Multiple` hints assert the containing loop's tile sizes are
  // multiples of the inner tiles. Although the loop tile size and inner tile sizes
  // are equal but distinct SSA values, the equivalence cannot be inferred in the
  // absence of the `Equal` hint.
  transform.named_sequence @aligned(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fused, %newc = transform.structured.fuse_into_containing_op %0 into %1
        inner_tile_alignments = [Multiple, Multiple]
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
  //   ALIGNED-LABEL: func.func @fuse_unpack_into_containing
  //       ALIGNED:   scf.forall
  // The Multiple hint keeps a dynamic ceilDiv outer dim (step and inner tile are
  // distinct SSA values, so it does not fold to 1) but needs no result slice.
  //       ALIGNED:     %[[OUTER:.+]] = affine.apply {{.*}}[%{{.*}}, %{{.*}}]
  //   ALIGNED-NOT:     linalg.unpack {{.*}} : tensor<1x1x?x?xf32> -> tensor<?x?xf32>
  //       ALIGNED:     %[[UNPACK:.+]] = linalg.unpack
  //  ALIGNED-SAME:         : tensor<?x?x?x?xf32> -> tensor<?x?xf32>
  //   ALIGNED-NOT:     tensor.extract_slice %[[UNPACK]]
  //       ALIGNED:     linalg.exp ins(%[[UNPACK]]

  // Dummy entry point so the default RUN line resolves here.
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    transform.yield
  }
}

// -----

// Producer fusion with a transposing consumer, `inner_tile_alignments` is
// a caller-asserted hint indexed in the unpack's dest-dim order, so the
// caller must arrange it for the transpose. Source dim 0 is `Multiple`
// so  -> outer size is `ceilDiv(loop tile,8 * vscale)` (dynamic); 
// source dim 1 is `Equal` -> outer size collapses to 1.

// CHECK-DAG: #[[$MAP_CEILDIV:.+]] = affine_map<(d0)[s0] -> (d0 ceildiv s0)>
func.func @unpack_transposed_consumer_scalable(%arg0: tensor<2x4x?x?xf32>, %out: tensor<?x?xf32>, %d0: index, %d1: index) -> tensor<?x?xf32> {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
  %t0 = arith.muli %c4, %vscale : index
  %t1 = arith.muli %c16, %vscale : index
  %0 = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %unpack = linalg.unpack %arg0 inner_dims_pos = [0, 1]
      inner_tiles = [%c8_vscale, %c4_vscale] into %0
      : tensor<2x4x?x?xf32> -> tensor<?x?xf32>
  // Consumer reads %unpack transposed.
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                                        affine_map<(d0, d1) -> (d0, d1)>],
                       iterator_types = ["parallel", "parallel"]}
       ins(%unpack : tensor<?x?xf32>) outs(%out : tensor<?x?xf32>) {
    ^bb0(%in: f32, %o: f32):
      linalg.yield %in : f32
  } -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @unpack_transposed_consumer_scalable
//  CHECK-DAG:    %[[C4:.*]] = arith.constant 4 : index
//  CHECK-DAG:    %[[C8:.*]] = arith.constant 8 : index
//  CHECK-DAG:    %[[C16:.*]] = arith.constant 16 : index
//  CHECK-DAG:    %[[VSCALE:.*]] = vector.vscale
//  CHECK-DAG:    %[[C4_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C4]] : index
//  CHECK-DAG:    %[[C8_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C8]] : index
//  CHECK-DAG:    %[[C16_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C16]] : index
//      CHECK:    %[[RES:.*]] = scf.for {{.*}} step %[[C4_VSCALE]]
//      CHECK:      scf.for {{.*}} step %[[C16_VSCALE]]
//      CHECK:        %[[OUTER_DIM0:.*]] = affine.apply #[[$MAP_CEILDIV]](%{{.*}})[%[[C8_VSCALE]]]
//      CHECK:        %[[SRC:.*]] = tensor.extract_slice %{{.*}}[%{{.*}}, %{{.*}}, 0, 0] [%[[OUTER_DIM0]], 1, %{{.*}}, %{{.*}}] [1, 1, 1, 1]
// CHECK-SAME:            tensor<2x4x?x?xf32> to tensor<?x1x?x?xf32>
//      CHECK:        %[[UNPACK:.*]] = linalg.unpack %[[SRC]]
//  CHECK-NOT:         tensor.extract_slice %[[UNPACK]]
//      CHECK:        linalg.generic
//      CHECK:      scf.yield
//      CHECK:    scf.yield
//      CHECK:    return %[[RES]]

module attributes {transform.with_named_sequence} {
  // Hint indexed in the unpack's dest-dim order: dim0 Multiple, dim1 Equal.
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %gen = transform.structured.match ops{["linalg.generic"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %mulis = transform.structured.match ops{["arith.muli"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %i0, %i1, %h0, %h1 = transform.split_handle %mulis
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
                                  !transform.any_op, !transform.any_op)
    %tiled, %loops:2 = transform.structured.fuse %gen tile_sizes [%h0, %h1]
        inner_tile_alignments = [Multiple, Equal]
        : (!transform.any_op, !transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }

  // Dummy entry points so the aligned/equal/unaligned RUN lines resolve here.
  transform.named_sequence @aligned(%arg1: !transform.any_op {transform.readonly}) {
    transform.yield
  }
  transform.named_sequence @equal(%arg1: !transform.any_op {transform.readonly}) {
    transform.yield
  }
  transform.named_sequence @unaligned(%arg1: !transform.any_op {transform.readonly}) {
    transform.yield
  }
}

// -----

// The hint also reaches the block-argument fusion path, when the producer is the
// `scf.forall` init (used through the block argument rather than via a direct
// extract use), `inner_tile_alignments` still yields the aligned tiling.

func.func @fuse_unpack_through_block_arg(
    %src: tensor<?x?x?x?xf32>, %unpack_empty: tensor<?x?xf32>,
    %ub0: index, %ub1: index) -> tensor<?x?xf32> {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
  %unpack = linalg.unpack %src inner_dims_pos = [0, 1]
      inner_tiles = [%c8_vscale, %c4_vscale] into %unpack_empty
      : tensor<?x?x?x?xf32> -> tensor<?x?xf32>
  %res = scf.forall (%i, %j) = (0, 0) to (%ub0, %ub1) step (%c8_vscale, %c4_vscale)
      shared_outs(%o = %unpack) -> (tensor<?x?xf32>) {
    %slice = tensor.extract_slice %o[%i, %j] [%c8_vscale, %c4_vscale] [1, 1]
        : tensor<?x?xf32> to tensor<?x?xf32>
    %0 = linalg.exp ins(%slice : tensor<?x?xf32>) outs(%slice : tensor<?x?xf32>) -> tensor<?x?xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %0 into %o[%i, %j] [%c8_vscale, %c4_vscale] [1, 1]
          : tensor<?x?xf32> into tensor<?x?xf32>
    }
  }
  return %res : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @fuse_unpack_through_block_arg
//       CHECK:   scf.forall
//       CHECK:     %[[UNPACK:.+]] = linalg.unpack
//  CHECK-SAME:         : tensor<1x1x?x?xf32> -> tensor<?x?xf32>
//   CHECK-NOT:     tensor.extract_slice %[[UNPACK]]
//       CHECK:     linalg.exp ins(%[[UNPACK]]

module attributes {transform.with_named_sequence} {
  // The `Equal` hints assert the containing loop's tile sizes equal the unpack
  // inner tile sizes (8 * vscale, 4 * vscale).
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fused, %newc = transform.structured.fuse_into_containing_op %0 into %1
        inner_tile_alignments = [Equal, Equal]
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }

  // Dummy entry points so the aligned/equal/unaligned RUN lines resolve here.
  transform.named_sequence @aligned(%arg1: !transform.any_op {transform.readonly}) {
    transform.yield
  }
  transform.named_sequence @equal(%arg1: !transform.any_op {transform.readonly}) {
    transform.yield
  }
  transform.named_sequence @unaligned(%arg1: !transform.any_op {transform.readonly}) {
    transform.yield
  }
}
