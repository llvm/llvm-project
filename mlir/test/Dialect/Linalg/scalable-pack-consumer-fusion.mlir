// RUN: mlir-opt %s -transform-interpreter -canonicalize -cse -split-input-file --verify-diagnostics | FileCheck %s

// Consumer fusion - linalg.pack with scalable inner tiles. Producer step (8*vscale)
// equals the pack inner tile size(8*vscale) on the tiled source dimension, so the
// outer dim of the fused pack tile is statically 1. This information is passed as
// an inner tile alignment hint `Equal`.

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @fuse_scalable_pack_consumer_equal
// CHECK-SAME:      %[[ARG0:.+]]: tensor<256x128xf32>, %[[ARG1:.+]]: tensor<256x128xf32>, %[[ARG2:.+]]: tensor<256x128xf32>, %[[DEST:.+]]: tensor<?x?x?x?xf32>
func.func @fuse_scalable_pack_consumer_equal(
    %arg0: tensor<256x128xf32>, %arg1: tensor<256x128xf32>,
    %arg2: tensor<256x128xf32>, %dest: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c256 = arith.constant 256 : index
  %vscale = vector.vscale
  %c4_vscale = arith.muli %c4, %vscale : index
  %c8_vscale = arith.muli %c8, %vscale : index

  // Loop tile size is equal to the inner tile size of the consumer `linalg.pack` (8 * vscale).
  %0 = scf.for %iv = %c0 to %c256 step %c8_vscale iter_args(%out = %arg2) -> (tensor<256x128xf32>) {
    %sz = affine.min affine_map<(d0)[s0] -> (-d0 + 256, s0)>(%iv)[%c8_vscale]
    %ext_out = tensor.extract_slice %out[%iv, 0] [%sz, 128] [1, 1]
        : tensor<256x128xf32> to tensor<?x128xf32>
    %ext_a = tensor.extract_slice %arg0[%iv, 0] [%sz, 128] [1, 1]
        : tensor<256x128xf32> to tensor<?x128xf32>
    %ext_b = tensor.extract_slice %arg1[%iv, 0] [%sz, 128] [1, 1]
        : tensor<256x128xf32> to tensor<?x128xf32>
    %computed = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%ext_a, %ext_b : tensor<?x128xf32>, tensor<?x128xf32>)
        outs(%ext_out : tensor<?x128xf32>) {
      ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
        %mul = arith.mulf %in0, %in1 : f32
        linalg.yield %mul : f32
    } -> tensor<?x128xf32>
    %inserted = tensor.insert_slice %computed into %out[%iv, 0] [%sz, 128] [1, 1]
        : tensor<?x128xf32> into tensor<256x128xf32>
    scf.yield %inserted : tensor<256x128xf32>
  }

  %pack = linalg.pack %0 outer_dims_perm = [0, 1]
      inner_dims_pos = [0, 1] inner_tiles = [%c8_vscale, %c4_vscale]
      into %dest : tensor<256x128xf32> -> tensor<?x?x?x?xf32>
  return %pack : tensor<?x?x?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %pack = transform.structured.match ops{["linalg.pack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    // The `Equal` hint is passed to hint the equality between the loop tile size 8 * vscale
    // and the inner tile size 8 * vscale.
    %a, %b = transform.test.fuse_consumer %pack into (%loop) inner_tile_alignments = [Equal, Unknown]
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
  //      CHECK:    %[[C8:.*]] = arith.constant 8 : index
  //      CHECK:    %[[VSCALE:.*]] = vector.vscale
  //      CHECK:    %[[C8_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C8]] : index
  //      CHECK:    %[[RES:.*]]:2 = scf.for {{.*}} step %[[C8_VSCALE]]
  // CHECK-SAME:        iter_args(%{{.*}} = %[[ARG2]], %{{.*}} = %[[DEST]])
  //      CHECK:      %[[GENERIC:.*]] = linalg.generic
  //      CHECK:      %[[PACK:.*]] = linalg.pack %[[GENERIC]]
  // CHECK-SAME:          inner_tiles = [%[[C8_VSCALE]], %{{.*}}]
  // CHECK-SAME:          -> tensor<1x?x?x?xf32>
  //      CHECK:      scf.yield {{.*}}, %{{.*}} :
  //      CHECK:    return %[[RES]]#1
}

// -----

// Consumer fusion with a static producer step (64) and a scalable pack inner
// tile (8*vscale), hinted `Multiple`. Fusion honors the hint and takes the aligned 
// (non-equal) path and the outer dim of the fused pack tile is dynamic (`64 ceildiv 8*vscale`).

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK:       #[[$MAP_CEILDIV:.+]] = affine_map<()[s0] -> (64 ceildiv s0)>
// CHECK-LABEL: func.func @fuse_scalable_pack_consumer_aligned
// CHECK-SAME:      %[[ARG0:.+]]: tensor<256x128xf32>, %[[ARG1:.+]]: tensor<256x128xf32>, %[[ARG2:.+]]: tensor<256x128xf32>, %[[DEST:.+]]: tensor<?x?x?x?xf32>
func.func @fuse_scalable_pack_consumer_aligned(
    %arg0: tensor<256x128xf32>, %arg1: tensor<256x128xf32>,
    %arg2: tensor<256x128xf32>, %dest: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index
  %vscale = vector.vscale
  %c4_vscale = arith.muli %c4, %vscale : index
  %c8_vscale = arith.muli %c8, %vscale : index

  %0 = scf.for %iv = %c0 to %c256 step %c64 iter_args(%out = %arg2) -> (tensor<256x128xf32>) {
    %ext_out = tensor.extract_slice %out[%iv, 0] [64, 128] [1, 1]
        : tensor<256x128xf32> to tensor<64x128xf32>
    %ext_a = tensor.extract_slice %arg0[%iv, 0] [64, 128] [1, 1]
        : tensor<256x128xf32> to tensor<64x128xf32>
    %ext_b = tensor.extract_slice %arg1[%iv, 0] [64, 128] [1, 1]
        : tensor<256x128xf32> to tensor<64x128xf32>
    %computed = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%ext_a, %ext_b : tensor<64x128xf32>, tensor<64x128xf32>)
        outs(%ext_out : tensor<64x128xf32>) {
      ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
        %mul = arith.mulf %in0, %in1 : f32
        linalg.yield %mul : f32
    } -> tensor<64x128xf32>
    %inserted = tensor.insert_slice %computed into %out[%iv, 0] [64, 128] [1, 1]
        : tensor<64x128xf32> into tensor<256x128xf32>
    scf.yield %inserted : tensor<256x128xf32>
  }

  %pack = linalg.pack %0 outer_dims_perm = [0, 1]
      inner_dims_pos = [0, 1] inner_tiles = [%c8_vscale, %c4_vscale]
      into %dest : tensor<256x128xf32> -> tensor<?x?x?x?xf32>
  return %pack : tensor<?x?x?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %pack = transform.structured.match ops{["linalg.pack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    // The `Multiple` hint is passed to hint the alignment between the loop tile size 64
    // and the inner tile size 8 * vscale.
    %a, %b = transform.test.fuse_consumer %pack into (%loop) inner_tile_alignments = [Multiple, Unknown]
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
  //  CHECK-DAG:    %[[C4:.*]] = arith.constant 4 : index
  //  CHECK-DAG:    %[[C8:.*]] = arith.constant 8 : index
  //  CHECK-DAG:    %[[C64:.*]] = arith.constant 64 : index
  //  CHECK-DAG:    %[[VSCALE:.*]] = vector.vscale
  //  CHECK-DAG:    %[[C8_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C8]] : index
  //  CHECK-DAG:    %[[C4_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C4]] : index
  //      CHECK:    %[[RES:.*]]:2 = scf.for {{.*}} step %[[C64]]
  // CHECK-SAME:        iter_args(%{{.*}} = %[[ARG2]], %{{.*}} = %[[DEST]])
  //      CHECK:      %[[GENERIC:.*]] = linalg.generic
  //      CHECK:      %[[OUTER:.*]] = affine.apply #[[$MAP_CEILDIV]]()[%[[C8_VSCALE]]]
  //      CHECK:      %[[PACK_DEST:.*]] = tensor.extract_slice %{{.*}}[%{{.*}}, 0, 0, 0] [%[[OUTER]], %{{.*}}, %{{.*}}, %{{.*}}] [1, 1, 1, 1]
  //      CHECK:      %[[PACK:.*]] = linalg.pack %[[GENERIC]]
  // CHECK-SAME:          inner_tiles = [%[[C8_VSCALE]], %[[C4_VSCALE]]]
  // CHECK-SAME:          into %[[PACK_DEST]]
  // CHECK-SAME:          -> tensor<?x?x?x?xf32>
  //      CHECK:      scf.yield {{.*}}, %{{.*}} :
  //      CHECK:    return %[[RES]]#1
}

// -----

// Consumer fusion - both producer step and pack inner tile are scalable, step (8*vscale)
// is an integer multiple of the inner tile (4*vscale) but not equal to it. The corresponding
// `Multiple` hint is passed to the tiling interface. Fusion succeeds via the aligned (non-equal)
// path, so the outer dim of the fused pack tile stays dynamic (`8*vscale ceildiv 4*vscale`).

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK:       #[[$MAP_MIN:.+]] = affine_map<(d0)[s0] -> (-d0 + 256, s0)>
// CHECK:       #[[$MAP_CEILDIV:.+]] = affine_map<(d0)[s0] -> (d0 ceildiv s0)>
// CHECK-LABEL: func.func @fuse_scalable_pack_consumer_aligned_scalable
// CHECK-SAME:      %[[ARG0:.+]]: tensor<256x128xf32>, %[[ARG1:.+]]: tensor<256x128xf32>, %[[ARG2:.+]]: tensor<256x128xf32>, %[[DEST:.+]]: tensor<?x?x?x?xf32>
func.func @fuse_scalable_pack_consumer_aligned_scalable(
    %arg0: tensor<256x128xf32>, %arg1: tensor<256x128xf32>,
    %arg2: tensor<256x128xf32>, %dest: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c256 = arith.constant 256 : index
  %vscale = vector.vscale
  %c4_vscale = arith.muli %c4, %vscale : index
  %c8_vscale = arith.muli %c8, %vscale : index
  
  // Loop tile size (8 * vscale) is a multiple of the inner tile size of the consumer `linalg.pack` (4 * vscale).
  %0 = scf.for %iv = %c0 to %c256 step %c8_vscale iter_args(%out = %arg2) -> (tensor<256x128xf32>) {
    %sz = affine.min affine_map<(d0)[s0] -> (-d0 + 256, s0)>(%iv)[%c8_vscale]
    %ext_out = tensor.extract_slice %out[%iv, 0] [%sz, 128] [1, 1]
        : tensor<256x128xf32> to tensor<?x128xf32>
    %ext_a = tensor.extract_slice %arg0[%iv, 0] [%sz, 128] [1, 1]
        : tensor<256x128xf32> to tensor<?x128xf32>
    %ext_b = tensor.extract_slice %arg1[%iv, 0] [%sz, 128] [1, 1]
        : tensor<256x128xf32> to tensor<?x128xf32>
    %computed = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%ext_a, %ext_b : tensor<?x128xf32>, tensor<?x128xf32>)
        outs(%ext_out : tensor<?x128xf32>) {
      ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
        %mul = arith.mulf %in0, %in1 : f32
        linalg.yield %mul : f32
    } -> tensor<?x128xf32>
    %inserted = tensor.insert_slice %computed into %out[%iv, 0] [%sz, 128] [1, 1]
        : tensor<?x128xf32> into tensor<256x128xf32>
    scf.yield %inserted : tensor<256x128xf32>
  }

  %pack = linalg.pack %0 outer_dims_perm = [0, 1]
      inner_dims_pos = [0, 1] inner_tiles = [%c4_vscale, %c4_vscale]
      into %dest : tensor<256x128xf32> -> tensor<?x?x?x?xf32>
  return %pack : tensor<?x?x?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %pack = transform.structured.match ops{["linalg.pack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    // The `Multiple` hint is passed to hint the alignment between the loop tile size
    // 8 * vscale and the inner tile size 4 * vscale.
    %a, %b = transform.test.fuse_consumer %pack into (%loop) inner_tile_alignments = [Multiple, Unknown]
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
  //  CHECK-DAG:    %[[C4:.*]] = arith.constant 4 : index
  //  CHECK-DAG:    %[[C8:.*]] = arith.constant 8 : index
  //  CHECK-DAG:    %[[VSCALE:.*]] = vector.vscale
  //  CHECK-DAG:    %[[C4_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C4]] : index
  //  CHECK-DAG:    %[[C8_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C8]] : index
  //      CHECK:    %[[RES:.*]]:2 = scf.for {{.*}} step %[[C8_VSCALE]]
  // CHECK-SAME:        iter_args(%{{.*}} = %[[ARG2]], %{{.*}} = %[[DEST]])
  //      CHECK:      %[[SZ:.*]] = affine.min #[[$MAP_MIN]](%{{.*}})[%[[C8_VSCALE]]]
  //      CHECK:      %[[GENERIC:.*]] = linalg.generic
  //      CHECK:      %[[OUTER:.*]] = affine.apply #[[$MAP_CEILDIV]](%[[SZ]])[%[[C4_VSCALE]]]
  //      CHECK:      %[[PACK_DEST:.*]] = tensor.extract_slice %{{.*}}[%{{.*}}, 0, 0, 0] [%[[OUTER]], %{{.*}}, %{{.*}}, %{{.*}}] [1, 1, 1, 1]
  //      CHECK:      %[[PACK:.*]] = linalg.pack %[[GENERIC]]
  // CHECK-SAME:          inner_tiles = [%[[C4_VSCALE]], %[[C4_VSCALE]]]
  // CHECK-SAME:          into %[[PACK_DEST]]
  // CHECK-SAME:          -> tensor<?x?x?x?xf32>
  //      CHECK:      scf.yield {{.*}}, %{{.*}} :
  //      CHECK:    return %[[RES]]#1
}

// -----

// Consumer fusion (negative): linalg.pack with scalable inner tiles and no alignment hint.
// The relationship between the loop tile size (8 * vscale) and the inner tile size (8 * vscale)
// cannot be decided statically. Without a user hint that asserts this, fusion fails.

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @negative_fuse_scalable_pack_consumer_no_hint(
    %arg0: tensor<?x128xf32>, %arg1: tensor<?x128xf32>,
    %arg2: tensor<?x128xf32>, %dest: tensor<?x?x?x?xf32>, %ub: index)
    -> tensor<?x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c4_vscale = arith.muli %c4, %vscale : index
  %c8_vscale = arith.muli %c8, %vscale : index

  %0 = scf.for %iv = %c0 to %ub step %c8_vscale iter_args(%out = %arg2) -> (tensor<?x128xf32>) {
    %sz = affine.min affine_map<(d0)[s0, s1] -> (s1 - d0, s0)>(%iv)[%c8_vscale, %ub]
    %ext_out = tensor.extract_slice %out[%iv, 0] [%sz, 128] [1, 1]
        : tensor<?x128xf32> to tensor<?x128xf32>
    %ext_a = tensor.extract_slice %arg0[%iv, 0] [%sz, 128] [1, 1]
        : tensor<?x128xf32> to tensor<?x128xf32>
    %ext_b = tensor.extract_slice %arg1[%iv, 0] [%sz, 128] [1, 1]
        : tensor<?x128xf32> to tensor<?x128xf32>
    %computed = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%ext_a, %ext_b : tensor<?x128xf32>, tensor<?x128xf32>)
        outs(%ext_out : tensor<?x128xf32>) {
      ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
        %mul = arith.mulf %in0, %in1 : f32
        linalg.yield %mul : f32
    } -> tensor<?x128xf32>
    %inserted = tensor.insert_slice %computed into %out[%iv, 0] [%sz, 128] [1, 1]
        : tensor<?x128xf32> into tensor<?x128xf32>
    scf.yield %inserted : tensor<?x128xf32>
  }

  // expected-error @below {{'linalg.pack' op failed to fuse consumer of slice}}
  %pack = linalg.pack %0 outer_dims_perm = [0, 1]
      inner_dims_pos = [0, 1] inner_tiles = [%c8_vscale, %c4_vscale]
      into %dest : tensor<?x128xf32> -> tensor<?x?x?x?xf32>
  return %pack : tensor<?x?x?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %pack = transform.structured.match ops{["linalg.pack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    // No inner_tile_alignments hint passed to signal the equality between the loop tile size
    // 8 * vscale and the inner tile size 8 * vscale.
    %a, %b = transform.test.fuse_consumer %pack into (%loop)
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Consumer fusion with a transposing outer_dims_perm. The hint is read in source
// (operand) order, so `Equal` on source dim 0 (loop step 8*vscale == inner tile
// 8*vscale) collapses that outer tile to 1; outer_dims_perm = [1, 0] then places
// it at result position 1 (`tensor<?x1x?x?xf32>`), not position 0.

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @fuse_pack_consumer_transposed_outer
func.func @fuse_pack_consumer_transposed_outer(
    %arg0: tensor<256x128xf32>, %arg1: tensor<256x128xf32>,
    %arg2: tensor<256x128xf32>, %dest: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c256 = arith.constant 256 : index
  %vscale = vector.vscale
  %c4_vscale = arith.muli %c4, %vscale : index
  %c8_vscale = arith.muli %c8, %vscale : index

  %0 = scf.for %iv = %c0 to %c256 step %c8_vscale iter_args(%out = %arg2) -> (tensor<256x128xf32>) {
    // 256 is not a static multiple of 8*vscale, so a real tiling clamps the
    // per-iteration tile to affine.min(256 - iv, 8*vscale).
    %sz = affine.min affine_map<(d0)[s0] -> (-d0 + 256, s0)>(%iv)[%c8_vscale]
    %ext_out = tensor.extract_slice %out[%iv, 0] [%sz, 128] [1, 1]
        : tensor<256x128xf32> to tensor<?x128xf32>
    %ext_a = tensor.extract_slice %arg0[%iv, 0] [%sz, 128] [1, 1]
        : tensor<256x128xf32> to tensor<?x128xf32>
    %ext_b = tensor.extract_slice %arg1[%iv, 0] [%sz, 128] [1, 1]
        : tensor<256x128xf32> to tensor<?x128xf32>
    %computed = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%ext_a, %ext_b : tensor<?x128xf32>, tensor<?x128xf32>)
        outs(%ext_out : tensor<?x128xf32>) {
      ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
        %mul = arith.mulf %in0, %in1 : f32
        linalg.yield %mul : f32
    } -> tensor<?x128xf32>
    %inserted = tensor.insert_slice %computed into %out[%iv, 0] [%sz, 128] [1, 1]
        : tensor<?x128xf32> into tensor<256x128xf32>
    scf.yield %inserted : tensor<256x128xf32>
  }

  %pack = linalg.pack %0 outer_dims_perm = [1, 0]
      inner_dims_pos = [0, 1] inner_tiles = [%c8_vscale, %c4_vscale]
      into %dest : tensor<256x128xf32> -> tensor<?x?x?x?xf32>
  return %pack : tensor<?x?x?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %pack = transform.structured.match ops{["linalg.pack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    // The `Equal` hint is passed to hint the equality between the loop tile size 8 * vscale
    // and the inner tile size 8 * vscale.
    %a, %b = transform.test.fuse_consumer %pack into (%loop) inner_tile_alignments = [Equal, Unknown]
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
  //  CHECK-DAG:    %[[C4:.*]] = arith.constant 4 : index
  //  CHECK-DAG:    %[[C8:.*]] = arith.constant 8 : index
  //  CHECK-DAG:    %[[VSCALE:.*]] = vector.vscale
  //  CHECK-DAG:    %[[C8_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C8]] : index
  //  CHECK-DAG:    %[[C4_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C4]] : index
  //      CHECK:    scf.for {{.*}} step %[[C8_VSCALE]]
  //      CHECK:      linalg.pack
  // CHECK-SAME:          outer_dims_perm = [1, 0]
  // CHECK-SAME:          inner_tiles = [%[[C8_VSCALE]], %[[C4_VSCALE]]]
  // CHECK-SAME:          -> tensor<?x1x?x?xf32>
  //      CHECK:      scf.yield
}
