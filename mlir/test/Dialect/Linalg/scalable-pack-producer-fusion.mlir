// RUN: mlir-opt %s --transform-interpreter=entry-point=ones_equal -canonicalize -cse | FileCheck %s --check-prefixes=ONES_FUSED
// RUN: mlir-opt %s --transform-interpreter=entry-point=ones_multiple -canonicalize -cse | FileCheck %s --check-prefixes=ONES_UNFUSED
// RUN: mlir-opt %s --transform-interpreter=entry-point=ones_nohint -canonicalize -cse | FileCheck %s --check-prefixes=ONES_UNFUSED
// RUN: mlir-opt %s --transform-interpreter=entry-point=inner_equal -canonicalize -cse | FileCheck %s --check-prefixes=INNER
// RUN: mlir-opt %s --transform-interpreter=entry-point=inner_multiple -canonicalize -cse | FileCheck %s --check-prefixes=INNER
// RUN: mlir-opt %s --transform-interpreter=entry-point=inner_nohint -canonicalize -cse | FileCheck %s --check-prefixes=INNER

// Producer fusion of a scalable `linalg.pack` into a tiled elementwise consumer.
// A pack producer can only be fused when the consumer requests full inner tiles.
//
// The `inner_tile_alignments` hint is a per-dimension keyword list:
//   - `Equal`:    the loop tile size equals the pack/unpack inner tile size.
//   - `Multiple`: the loop tile size is an integer multiple of the inner tile.
//   - `Unknown`:  the default; nothing is asserted for that dimension.

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @pack_elemwise(%src: tensor<?x?xf32>, %pack_dest: tensor<?x?x?x?xf32>,
    %out: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
  %pack = linalg.pack %src inner_dims_pos = [0, 1]
      inner_tiles = [%c8_vscale, %c4_vscale] into %pack_dest
      : tensor<?x?xf32> -> tensor<?x?x?x?xf32>
  %generic = linalg.generic {indexing_maps = [#map, #map],
                             iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%pack : tensor<?x?x?x?xf32>) outs(%out : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %o: f32):
      %e = math.exp %in : f32
      linalg.yield %e : f32
  } -> tensor<?x?x?x?xf32>
  return %generic : tensor<?x?x?x?xf32>
}

module attributes {transform.with_named_sequence} {
  // Tile the consumer's outer dims by 1. `Equal` -> the pack fuses.
  transform.named_sequence @ones_equal(%arg1: !transform.any_op {transform.readonly}) {
    %g = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %t, %loops:2 = transform.structured.fuse %g tile_sizes [1, 1, 0, 0] inner_tile_alignments = [Equal, Equal]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
  // ONES_FUSED-LABEL: func.func @pack_elemwise
  //   ONES_FUSED-DAG:    %[[C1:.*]] = arith.constant 1 : index
  //       ONES_FUSED:    %[[RES:.*]] = scf.for {{.*}} step %[[C1]]
  //       ONES_FUSED:      scf.for {{.*}} step %[[C1]]
  // The pack is fused into the loop: it reads a source slice and produces a
  // collapsed 1x1 outer tile that the generic consumes directly.
  //       ONES_FUSED:        %[[SRC:.*]] = tensor.extract_slice %{{.*}} : tensor<?x?xf32> to tensor<?x?xf32>
  //       ONES_FUSED:        %[[PACK:.*]] = linalg.pack %[[SRC]]
  //  ONES_FUSED-SAME:            -> tensor<1x1x?x?xf32>
  //       ONES_FUSED:        linalg.generic {{.*}} ins(%[[PACK]]
  //       ONES_FUSED:    return %[[RES]]

  // Same input, `Multiple` hint: not sufficient, the pack is not fused.
  transform.named_sequence @ones_multiple(%arg1: !transform.any_op {transform.readonly}) {
    %g = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %t, %loops:2 = transform.structured.fuse %g tile_sizes [1, 1, 0, 0] inner_tile_alignments = [Multiple, Multiple]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
  // Same input, no hint: the pack is not fused.
  transform.named_sequence @ones_nohint(%arg1: !transform.any_op {transform.readonly}) {
    %g = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %t, %loops:2 = transform.structured.fuse %g tile_sizes [1, 1, 0, 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
  // ONES_UNFUSED-LABEL: func.func @pack_elemwise
  // Without `Equal` the full-inner-tile requirement cannot be proven (scalable),
  // so the pack stays outside the loop as a full pack and the tiled generic reads
  // slices of its result.
  //       ONES_UNFUSED:    %[[PACK:.*]] = linalg.pack
  //  ONES_UNFUSED-SAME:        -> tensor<?x?x?x?xf32>
  //       ONES_UNFUSED:    %[[RES:.*]] = scf.for
  //       ONES_UNFUSED:      scf.for
  //       ONES_UNFUSED:        tensor.extract_slice %[[PACK]]
  //       ONES_UNFUSED:        linalg.generic
  //       ONES_UNFUSED:    return %[[RES]]

  // Tile the packed inner-tile dims. The pack is never fused (its inner tiles are not tileable),
  // regardless of the hint - only the generic is tiled. All three sequences produce the same IR.

  transform.named_sequence @inner_equal(%arg1: !transform.any_op {transform.readonly}) {
    %g = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %t, %loops:2 = transform.structured.fuse %g tile_sizes [0, 0, 8, 4] inner_tile_alignments = [Equal, Equal]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
  transform.named_sequence @inner_multiple(%arg1: !transform.any_op {transform.readonly}) {
    %g = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %t, %loops:2 = transform.structured.fuse %g tile_sizes [0, 0, 8, 4] inner_tile_alignments = [Multiple, Multiple]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
  transform.named_sequence @inner_nohint(%arg1: !transform.any_op {transform.readonly}) {
    %g = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %t, %loops:2 = transform.structured.fuse %g tile_sizes [0, 0, 8, 4]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
  // INNER-LABEL: func.func @pack_elemwise
  //       INNER:    %[[PACK:.*]] = linalg.pack
  //  INNER-SAME:        -> tensor<?x?x?x?xf32>
  //       INNER:    scf.for
  //       INNER:      scf.for
  //       INNER:        tensor.extract_slice %[[PACK]]
  //       INNER:        linalg.generic
}
