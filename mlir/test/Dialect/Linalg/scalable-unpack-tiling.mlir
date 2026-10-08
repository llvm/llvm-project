// RUN: mlir-opt %s --transform-interpreter=entry-point=equal -canonicalize -cse | FileCheck %s --check-prefixes=ALL,EQUAL
// RUN: mlir-opt %s --transform-interpreter=entry-point=aligned -canonicalize -cse | FileCheck %s --check-prefixes=ALL,ALIGNED
// RUN: mlir-opt %s --transform-interpreter=entry-point=unaligned -canonicalize -cse | FileCheck %s --check-prefixes=ALL,UNALIGNED

// Pure tiling of a scalable `linalg.unpack` (no producer/consumer fusion). The
// same input IR is tiled three ways, each selected by a different transform
// entry point (see the named sequences and RUN lines):
//   - `@equal`     tiles with [2, 4] * vscale, equal to the inner tiles, and
//                  asserts that via `[Equal, Equal]`. Perfect tiling: the tiled
//                  unpack's outer dims collapse to 1 and no remainder is needed.
//   - `@aligned`   tiles with [4, 8] * vscale, integer multiples of the inner
//                  tiles, asserted via `[Multiple, Multiple]`. The outer dims
//                  become `ceilDiv(loop tile, inner tile)` and no remainder is
//                  needed.
//   - `@unaligned` tiles with static [7, 5], not aligned to the scalable inner
//                  tiles and with no hint, so the tiled unpack keeps a trailing
//                  remainder `tensor.extract_slice`.
//
// The `inner_tile_alignments` hint is a per-dimension keyword list:
//   - `Equal`:    the loop tile size equals the pack/unpack inner tile size.
//   - `Multiple`: the loop tile size is an integer multiple of the inner tile.
//   - `Unknown`:  the default; nothing is asserted for that dimension.

//   ALIGNED-DAG: #[[$MAP_CEILDIV:.+]] = affine_map<(d0)[s0] -> (d0 ceildiv s0)>
// ALL-LABEL: func.func @CKkc_to_KC_scalable

func.func @CKkc_to_KC_scalable(%source: tensor<32x4x?x?xf32>, %dest: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %vscale = vector.vscale
  %c2_vscale = arith.muli %c2, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
  %0 = linalg.unpack %source outer_dims_perm = [1, 0] inner_dims_pos = [0, 1]
      inner_tiles = [%c2_vscale, %c4_vscale] into %dest
      : tensor<32x4x?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  // Perfect tiling: the scalable tile sizes ([2] = 2 * vscale, [4] = 4 * vscale)
  // equal the scalable inner tiles, asserted via `Equal`.
  transform.named_sequence @equal(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [[2], [4]]
        inner_tile_alignments = [Equal, Equal]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
  // --- @equal: perfect tiling, outer dims collapse to 1. ---
  //   EQUAL-DAG:    %[[C2:.*]] = arith.constant 2 : index
  //   EQUAL-DAG:    %[[C4:.*]] = arith.constant 4 : index
  //   EQUAL-DAG:    %[[VSCALE:.*]] = vector.vscale
  //   EQUAL-DAG:    %[[C2_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C2]] : index
  //   EQUAL-DAG:    %[[C4_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C4]] : index
  //       EQUAL:    %[[RES:.*]] = scf.for {{.*}} step %[[C2_VSCALE]]
  //       EQUAL:      scf.for {{.*}} step %[[C4_VSCALE]]
  //       EQUAL:        %[[UNPACK:.*]] = linalg.unpack
  //  EQUAL-SAME:            tensor<1x1x?x?xf32> -> tensor<?x?xf32>
  // For the perfect tiling case, the outer dims equal 1 and the unpack can be
  // consumed directly - no extra extract_slice on the result is needed.
  //   EQUAL-NOT:         tensor.extract_slice %[[UNPACK]]
  //       EQUAL:        tensor.insert_slice %[[UNPACK]]
  //       EQUAL:      scf.yield
  //       EQUAL:    scf.yield
  //       EQUAL:    return %[[RES]]

  // Aligned tiling: the scalable tile sizes ([4] = 4 * vscale, [8] = 8 * vscale)
  // are integer multiples of the scalable inner tiles (2 * vscale, 4 * vscale),
  // asserted via `Multiple`.
  transform.named_sequence @aligned(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [[4], [8]]
        inner_tile_alignments = [Multiple, Multiple]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
  // --- @aligned: multiple tiling, outer dims are ceilDiv(loop tile, inner tile). ---
  // ALIGNED-DAG:    %[[C2:.*]] = arith.constant 2 : index
  // ALIGNED-DAG:    %[[C4:.*]] = arith.constant 4 : index
  // ALIGNED-DAG:    %[[VSCALE:.*]] = vector.vscale
  // ALIGNED-DAG:    %[[C2_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C2]] : index
  // ALIGNED-DAG:    %[[C4_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C4]] : index
  // ALIGNED-DAG:    %[[C8_VSCALE:.*]] = arith.muli %[[VSCALE]], %{{.*}} : index
  //     ALIGNED:    %[[RES:.*]] = scf.for {{.*}} step %[[C4_VSCALE]]
  //     ALIGNED:      scf.for {{.*}} step %[[C8_VSCALE]]
  // The outer dims are ceilDiv(loop tile, inner tile); the transpose (outer_dims_perm
  // = [1, 0]) swaps them back into source-dim order in the extracted slice.
  //     ALIGNED:        %[[CEIL_C2:.*]] = affine.apply #[[$MAP_CEILDIV]](%{{.*}})[%[[C2_VSCALE]]]
  //     ALIGNED:        %[[CEIL_C4:.*]] = affine.apply #[[$MAP_CEILDIV]](%{{.*}})[%[[C4_VSCALE]]]
  //     ALIGNED:        %[[SRC:.*]] = tensor.extract_slice %{{.*}}[%{{.*}}, %{{.*}}, 0, 0] [%[[CEIL_C4]], %[[CEIL_C2]], %[[C2_VSCALE]], %[[C4_VSCALE]]]
  //     ALIGNED:        %[[UNPACK:.*]] = linalg.unpack %[[SRC]]
  // ALIGNED-SAME:            tensor<?x?x?x?xf32> -> tensor<?x?xf32>
  // For the aligned tiling case, the unpack can also be consumed directly.
  // ALIGNED-NOT:         tensor.extract_slice %[[UNPACK]]
  //     ALIGNED:        tensor.insert_slice %[[UNPACK]]
  //     ALIGNED:      scf.yield
  //     ALIGNED:    scf.yield
  //     ALIGNED:    return %[[RES]]

  // Unaligned tiling: static tile sizes (7, 5) are not aligned to the scalable
  // inner tiles and no hint is passed, so the tiled unpack keeps a remainder.
  transform.named_sequence @unaligned(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [7, 5]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
  // --- @unaligned: static tiling, keeps a trailing remainder slice. ---
  // UNALIGNED-DAG:    %[[C7:.*]] = arith.constant 7 : index
  // UNALIGNED-DAG:    %[[C5:.*]] = arith.constant 5 : index
  //     UNALIGNED:    %[[RES:.*]] = scf.for {{.*}} step %[[C7]]
  //     UNALIGNED:      scf.for {{.*}} step %[[C5]]
  //     UNALIGNED:        %[[UNPACK:.*]] = linalg.unpack
  // UNALIGNED-SAME:            tensor<?x?x?x?xf32> -> tensor<?x?xf32>
  // For the unaligned tiling case the unpack over-computes, so a trailing
  // extract_slice recovers the correct slice from the result.
  //     UNALIGNED:        %[[EXTRACT:.*]] = tensor.extract_slice %[[UNPACK]]
  //     UNALIGNED:        tensor.insert_slice %[[EXTRACT]]
  //     UNALIGNED:      scf.yield
  //     UNALIGNED:    scf.yield
  //     UNALIGNED:    return %[[RES]]
}
