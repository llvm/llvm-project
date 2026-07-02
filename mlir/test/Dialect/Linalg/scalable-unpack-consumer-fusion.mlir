// RUN: mlir-opt %s -transform-interpreter -split-input-file -canonicalize -cse | FileCheck %s --check-prefixes=CHECK
// RUN: not mlir-opt %s --transform-interpreter=entry-point=no_hint -split-input-file 2>&1 | FileCheck %s --check-prefixes=NO_HINT
// RUN: not mlir-opt %s --transform-interpreter=entry-point=multiple -split-input-file 2>&1 | FileCheck %s --check-prefixes=MULTIPLE

// Consumer fusion of a scalable `linalg.unpack` into an `scf.for` loop.
// The `inner_tile_alignments` hint is a per-dimension keyword list:
//   - `Equal`:    the loop tile size equals the pack/unpack inner tile size.
//   - `Multiple`: the loop tile size is an integer multiple of the inner tile.
//   - `Unknown`:  the default; nothing is asserted for that dimension.

// The loop tile size on the tiled (inner-tile) dimension is 8 * vscale, the same
// scalable value as the unpack inner tile. The SAME input IR is fused three ways:
// with the `Equal` hint, with the `Multiple` hint, and without a hint.

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @fuse_scalable_unpack_consumer(
    %arg0: tensor<32x?xf32>, %arg1: tensor<32x?xf32>,
    %arg2: tensor<32x?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %dim1 = tensor.dim %arg2, %c1 : tensor<32x?xf32>

  // Loop tile size is equal to the inner tile size of the consumer
  // `linalg.unpack` (8 * vscale).
  %0 = scf.for %iv = %c0 to %dim1 step %c8_vscale iter_args(%out = %arg2) -> (tensor<32x?xf32>) {
    %sz = affine.min affine_map<(d0)[s0, s1] -> (s1 - d0, s0)>(%iv)[%c8_vscale, %dim1]
    %ext_a = tensor.extract_slice %arg0[0, %iv] [32, %sz] [1, 1]
        : tensor<32x?xf32> to tensor<32x?xf32>
    %ext_b = tensor.extract_slice %arg1[0, %iv] [32, %sz] [1, 1]
        : tensor<32x?xf32> to tensor<32x?xf32>
    %ext_out = tensor.extract_slice %out[0, %iv] [32, %sz] [1, 1]
        : tensor<32x?xf32> to tensor<32x?xf32>
    %computed = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%ext_a, %ext_b : tensor<32x?xf32>, tensor<32x?xf32>)
        outs(%ext_out : tensor<32x?xf32>) {
      ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
        %mul = arith.mulf %in0, %in1 : f32
        linalg.yield %mul : f32
    } -> tensor<32x?xf32>
    %inserted = tensor.insert_slice %computed into %out[0, %iv] [32, %sz] [1, 1]
        : tensor<32x?xf32> into tensor<32x?xf32>
    scf.yield %inserted : tensor<32x?xf32>
  }

  %output = tensor.empty(%dim1) : tensor<?xf32>
  %unpack = linalg.unpack %0 outer_dims_perm = [0]
      inner_dims_pos = [0] inner_tiles = [%c8_vscale]
      into %output : tensor<32x?xf32> -> tensor<?xf32>
  return %unpack : tensor<?xf32>
}

module attributes {transform.with_named_sequence} {
  // Intended fusion: the loop tile equals the inner tile, asserted via `Equal`.
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %unpack = transform.structured.match ops{["linalg.unpack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_consumer %unpack into (%loop) inner_tile_alignments = [Equal]
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
  //   CHECK-LABEL: func.func @fuse_scalable_unpack_consumer
  //    CHECK-SAME:     %[[ARG0:.+]]: tensor<32x?xf32>, %[[ARG1:.+]]: tensor<32x?xf32>, %[[ARG2:.+]]: tensor<32x?xf32>
  //         CHECK:    %[[VSCALE:.*]] = vector.vscale
  //         CHECK:    %[[C8_VSCALE:.*]] = arith.muli %[[VSCALE]], %{{.*}} : index
  //         CHECK:    %[[RES:.*]]:2 = scf.for {{.*}} step %[[C8_VSCALE]]
  //    CHECK-SAME:        iter_args(%{{.*}} = %[[ARG2]], %{{.*}} = %{{.*}})
  //         CHECK:      %[[GENERIC:.*]] = linalg.generic
  //         CHECK:      %[[UNPACK:.*]] = linalg.unpack %[[GENERIC]]
  //    CHECK-SAME:          inner_tiles = [%[[C8_VSCALE]]]
  //         CHECK:      scf.yield {{.*}}, %{{.*}} :
  //         CHECK:    return %[[RES]]#1

  // No hint: fusion fails (scalable loop tile vs scalable inner tile, undecidable).
  transform.named_sequence @no_hint(%arg1: !transform.any_op {transform.readonly}) {
    %unpack = transform.structured.match ops{["linalg.unpack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_consumer %unpack into (%loop)
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
  // NO_HINT: 'linalg.unpack' op failed to fuse consumer of slice

  // `Multiple` hint: insufficient for an unpack consumer's inner dim, fusion fails.
  transform.named_sequence @multiple(%arg1: !transform.any_op {transform.readonly}) {
    %unpack = transform.structured.match ops{["linalg.unpack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_consumer %unpack into (%loop) inner_tile_alignments = [Multiple]
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
  // MULTIPLE: 'linalg.unpack' op failed to fuse consumer of slice
}

// -----

// A 3-op dispatch tiled from the mmt4d root, fusing its consumers, with an
// inner-tile alignment hint driving the scalable unpack fusion:
//
//   linalg.mmt4d              (root, produces a packed [M, N, M0, N0] layout)
//   -> linalg.generic         (transposing bias add: [M, N, M0, N0] -> [N, M, N0, M0])
//   -> linalg.unpack          ([N, M, N0, M0] -> [N*N0, M*M0])
//
// The mmt4d is tiled along its scalable inner dim N0 (iteration dim 4) by
// 8 * vscale; the generic and unpack are fused as consumers. The SAME input IR
// is fused two ways: with the equal hint and without.

#id4 = affine_map<(m, n, m0, n0) -> (m, n, m0, n0)>
#tr4 = affine_map<(m, n, m0, n0) -> (n, m, n0, m0)>

// CHECK: #[[$TR:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>
func.func @mmt4d_transpose_unpack(
    %lhs: tensor<2x2x4x2xf32>, %rhs: tensor<2x2x?x2xf32>,
    %acc: tensor<2x2x4x?xf32>, %bias: tensor<2x2x?x4xf32>,
    %trinit: tensor<2x2x?x4xf32>, %out: tensor<?x8xf32>) -> tensor<?x8xf32> {
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index

  // 1. mmt4d root: lhs[M,K,M0,K0] x rhs[N,K,N0,K0] -> out[M,N,M0,N0].
  %mm = linalg.mmt4d ins(%lhs, %rhs : tensor<2x2x4x2xf32>, tensor<2x2x?x2xf32>)
      outs(%acc : tensor<2x2x4x?xf32>) -> tensor<2x2x4x?xf32>

  // 2. transposing bias add: [M,N,M0,N0] -> [N,M,N0,M0].
  %tr = linalg.generic {
      indexing_maps = [#id4, #tr4, #tr4],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%mm, %bias : tensor<2x2x4x?xf32>, tensor<2x2x?x4xf32>)
      outs(%trinit : tensor<2x2x?x4xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %s = arith.addf %a, %b : f32
      linalg.yield %s : f32
  } -> tensor<2x2x?x4xf32>

  // 3. unpack: [N,M,N0,M0] -> [N*N0, M*M0] = [?, 8].
  %unpack = linalg.unpack %tr outer_dims_perm = [0, 1] inner_dims_pos = [0, 1]
      inner_tiles = [%c8_vscale, 4] into %out
      : tensor<2x2x?x4xf32> -> tensor<?x8xf32>
  return %unpack : tensor<?x8xf32>
}

module attributes {transform.with_named_sequence} {
  // Intended fusion: tile the mmt4d root, fuse the transposing generic and the
  // unpack, asserting `Equal` on the unpack's tiled (transposed) inner dim.
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %mmt4d = transform.structured.match ops{["linalg.mmt4d"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    // Tile the mmt4d root along its scalable inner dim N0 (iteration dim 4).
    %tiled, %loop = transform.structured.tile_using_for %mmt4d tile_sizes [0, 0, 0, 0, [8], 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %gen = transform.structured.match ops{["linalg.generic"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %fg, %loop2 = transform.test.fuse_consumer %gen into (%loop)
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %unp = transform.structured.match ops{["linalg.unpack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %fu, %loop3 = transform.test.fuse_consumer %unp into (%loop2) inner_tile_alignments = [Equal, Unknown]
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
  //   CHECK-LABEL: func.func @mmt4d_transpose_unpack
  //    CHECK-SAME:     %[[LHS:.+]]: tensor<2x2x4x2xf32>, %[[RHS:.+]]: tensor<2x2x?x2xf32>, %[[ACC:.+]]: tensor<2x2x4x?xf32>, %[[BIAS:.+]]: tensor<2x2x?x4xf32>, %[[TRINIT:.+]]: tensor<2x2x?x4xf32>, %[[OUT:.+]]: tensor<?x8xf32>
  //         CHECK:    %[[VSCALE:.*]] = vector.vscale
  //         CHECK:    %[[C8_VSCALE:.*]] = arith.muli %[[VSCALE]], %{{.*}} : index
  //         CHECK:    %[[RES:.*]]:3 = scf.for {{.*}} step %[[C8_VSCALE]]
  //    CHECK-SAME:        iter_args(%{{.*}} = %[[ACC]], %{{.*}} = %[[TRINIT]], %{{.*}} = %[[OUT]])
  //         CHECK:      %[[MM:.*]] = linalg.mmt4d
  //         CHECK:      %[[GENERIC:.*]] = linalg.generic
  //    CHECK-SAME:          indexing_maps = [#{{.+}}, #[[$TR]], #[[$TR]]]
  //    CHECK-SAME:          ins(%[[MM]],
  //         CHECK:      %[[UNPACK:.*]] = linalg.unpack %[[GENERIC]]
  //    CHECK-SAME:          outer_dims_perm = [0, 1] inner_dims_pos = [0, 1]
  //    CHECK-SAME:          inner_tiles = [%[[C8_VSCALE]], 4]
  //         CHECK:      scf.yield {{.*}}, {{.*}}, %{{.*}} :
  //         CHECK:    return %[[RES]]#2

  // No hint on the unpack: unpack fusion fails.
  transform.named_sequence @no_hint(%arg1: !transform.any_op {transform.readonly}) {
    %mmt4d = transform.structured.match ops{["linalg.mmt4d"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_for %mmt4d tile_sizes [0, 0, 0, 0, [8], 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %gen = transform.structured.match ops{["linalg.generic"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %fg, %loop2 = transform.test.fuse_consumer %gen into (%loop)
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %unp = transform.structured.match ops{["linalg.unpack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %fu, %loop3 = transform.test.fuse_consumer %unp into (%loop2)
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
  // NO_HINT: 'linalg.unpack' op failed to fuse consumer of slice

  // Dummy entry point so the `multiple` RUN line resolves here.
  transform.named_sequence @multiple(%arg1: !transform.any_op {transform.readonly}) {
    transform.yield
  }
}
