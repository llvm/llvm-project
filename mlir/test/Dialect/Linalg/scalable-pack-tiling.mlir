// RUN: mlir-opt %s --transform-interpreter=entry-point=ones_equal -canonicalize -cse | FileCheck %s --check-prefixes=ONES
// RUN: mlir-opt %s --transform-interpreter=entry-point=ones_multiple -canonicalize -cse | FileCheck %s --check-prefixes=ONES
// RUN: mlir-opt %s --transform-interpreter=entry-point=ones_nohint -canonicalize -cse | FileCheck %s --check-prefixes=ONES
// RUN: mlir-opt %s --transform-interpreter=entry-point=twofour_equal -canonicalize -cse | FileCheck %s --check-prefixes=ALIGNED
// RUN: mlir-opt %s --transform-interpreter=entry-point=twofour_multiple -canonicalize -cse | FileCheck %s --check-prefixes=ALIGNED
// RUN: mlir-opt %s --transform-interpreter=entry-point=twofour_nohint -canonicalize -cse | FileCheck %s --check-prefixes=ALIGNED

// Tiling of a scalable `linalg.pack`. A pack's tiling interface iterates over the
// packed (destination) OUTER dims only,  the inner tiles are not part of the
// iteration domain - so tiling is completely independent of any
// `inner_tile_alignments` hint.

func.func @pack(%src: tensor<128x256xf32>, %dest: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
  %pack = linalg.pack %src inner_dims_pos = [0, 1]
      inner_tiles = [%c8_vscale, %c4_vscale] into %dest
      : tensor<128x256xf32> -> tensor<?x?x?x?xf32>
  return %pack : tensor<?x?x?x?xf32>
}

module attributes {transform.with_named_sequence} {
  // Tile the two outer dims by 1. Hint-independent, the three sequences below produce identical IR.
  transform.named_sequence @ones_equal(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.pack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [1, 1] inner_tile_alignments = [Equal, Equal]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
  transform.named_sequence @ones_multiple(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.pack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [1, 1] inner_tile_alignments = [Multiple, Multiple]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
  transform.named_sequence @ones_nohint(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.pack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [1, 1]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
  // ONES-LABEL: func.func @pack
  //   ONES-DAG:    %[[C1:.*]] = arith.constant 1 : index
  //       ONES:    %[[RES:.*]] = scf.for {{.*}} step %[[C1]]
  //       ONES:      scf.for {{.*}} step %[[C1]]
  //       ONES:        %[[PACK:.*]] = linalg.pack
  //  ONES-SAME:            tensor<?x?xf32> -> tensor<1x1x?x?xf32>
  //       ONES:        tensor.insert_slice %[[PACK]]
  //       ONES:      scf.yield
  //       ONES:    scf.yield
  //       ONES:    return %[[RES]]

  // Tile the two outer dims by 2 and 4. Hint-independent.
  transform.named_sequence @twofour_equal(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.pack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [2, 4] inner_tile_alignments = [Equal, Equal]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
  transform.named_sequence @twofour_multiple(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.pack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [2, 4] inner_tile_alignments = [Multiple, Multiple]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
  transform.named_sequence @twofour_nohint(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.pack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [2, 4]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
  // ALIGNED-LABEL: func.func @pack
  // ALIGNED-DAG:    %[[C2:.*]] = arith.constant 2 : index
  // ALIGNED-DAG:    %[[C4:.*]] = arith.constant 4 : index
  //     ALIGNED:    %[[RES:.*]] = scf.for {{.*}} step %[[C2]]
  //     ALIGNED:      scf.for {{.*}} step %[[C4]]
  //     ALIGNED:        %[[SRC:.*]] = tensor.extract_slice %{{.*}} : tensor<128x256xf32> to tensor<?x?xf32>
  //     ALIGNED:        %[[PACK:.*]] = linalg.pack %[[SRC]]
  // ALIGNED-SAME:            tensor<?x?xf32> -> tensor<?x?x?x?xf32>
  //     ALIGNED:        tensor.insert_slice %[[PACK]]
  //     ALIGNED:      scf.yield
  //     ALIGNED:    scf.yield
  //     ALIGNED:    return %[[RES]]
}
