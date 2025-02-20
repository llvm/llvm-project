// RUN: mlir-opt %s --transform-interpreter --split-input-file -canonicalize | FileCheck %s

// For pack op, we use lowerPadLikeWithInsertSlice = false to ensure no insert_slice is generated.
// This allows linalg.transpose to be fused as a producer operation. In below testcase, linalg.transpose
// as a producer operation is fused into the scf.forall loop.

module {
  // CHECK-label: func @fuse_pack_as_producer
  // CHECK:       scf.forall {{.*}} {
  // CHECK:         %[[PRODUCER:.*]] = linalg.transpose
  // CHECK:         linalg.generic {{.*}} ins(%[[PRODUCER]]
  // CHECK:         scf.forall.in_parallel
  // CHECK:       }
  func.func @fuse_pack_as_producer(%src: tensor<128x256xf32>, %other: tensor<4x4x128x256xf32>)
      -> tensor<4x4x128x256xf32> {
    %dest = tensor.empty() : tensor<1x1x128x256xf32>
    %pack = linalg.pack %src inner_dims_pos = [0, 1] inner_tiles = [128, 256]
        into %dest : tensor<128x256xf32> -> tensor<1x1x128x256xf32>

    %out = tensor.empty() : tensor<4x4x128x256xf32>
    %res = linalg.generic
        {indexing_maps = [affine_map<(i, j, k, l) -> (0, 0, k, l)>,
                          affine_map<(i, j, k, l) -> (i, j, k, l)>,
                          affine_map<(i, j, k, l) -> (i, j, k, l)>],
         iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%pack, %other: tensor<1x1x128x256xf32>, tensor<4x4x128x256xf32>)
        outs(%out: tensor<4x4x128x256xf32>) {
      ^bb0(%pack_elem: f32, %other_elem: f32, %out_elem: f32):
        %r = arith.addf %pack_elem, %other_elem : f32
        linalg.yield %r : f32
    } -> tensor<4x4x128x256xf32>

    return %res : tensor<4x4x128x256xf32>
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      // Find and lower pack operation.
      %pack = transform.structured.match ops{["linalg.pack"]} in %arg1
        : (!transform.any_op) -> !transform.op<"linalg.pack">
      %paded, %expanded, %transpose = transform.structured.lower_pack %pack {lowerPadLikeWithInsertSlice = false}
        : (!transform.op<"linalg.pack">)
        -> (!transform.op<"tensor.pad">,
            !transform.op<"tensor.expand_shape">,
            !transform.op<"linalg.transpose">)

      %root = transform.structured.match ops{["linalg.generic"]} in %arg1
          : (!transform.any_op) -> !transform.any_op
      // Tile the lialg operation with parallel forall loop tiling [4, 4].
      %tiled_op, %forall_op = transform.structured.tile_using_forall %root num_threads [4, 4]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

      // Fuse the transpose operation into the tiled loop.
      transform.structured.fuse_into_containing_op %transpose into %forall_op
          : (!transform.op<"linalg.transpose">, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}

// -----
// For pack op, by default lowerPadLikeWithInsertSlice = true, which generates insert_slice and blocks fusion.
// In below testcase, tensor.insert_slice as a producer operation cannot be fused into the scf.forall loop.

module {
  // CHECK-label: func @fuse_pack_as_producer_blocked_by_insert_slice
  // CHECK:       %[[PRODUCER:.*]] = tensor.insert_slice
  // CHECK:       scf.forall {{.*}} {
  // CHECK:         linalg.generic {{.*}} ins(%[[PRODUCER]]
  // CHECK:         scf.forall.in_parallel
  // CHECK:       }
  func.func @fuse_pack_as_producer_blocked_by_insert_slice(%src: tensor<128x256xf32>, %other: tensor<4x4x128x256xf32>)
      -> tensor<4x4x128x256xf32> {
    %dest = tensor.empty() : tensor<1x1x128x256xf32>
    %pack = linalg.pack %src inner_dims_pos = [0, 1] inner_tiles = [128, 256]
        into %dest : tensor<128x256xf32> -> tensor<1x1x128x256xf32>

    %out = tensor.empty() : tensor<4x4x128x256xf32>
    %res = linalg.generic
        {indexing_maps = [affine_map<(i, j, k, l) -> (0, 0, k, l)>,
                          affine_map<(i, j, k, l) -> (i, j, k, l)>,
                          affine_map<(i, j, k, l) -> (i, j, k, l)>],
         iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%pack, %other: tensor<1x1x128x256xf32>, tensor<4x4x128x256xf32>)
        outs(%out: tensor<4x4x128x256xf32>) {
      ^bb0(%pack_elem: f32, %other_elem: f32, %out_elem: f32):
        %r = arith.addf %pack_elem, %other_elem : f32
        linalg.yield %r : f32
    } -> tensor<4x4x128x256xf32>

    return %res : tensor<4x4x128x256xf32>
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      // Find and lower pack operation.
      %pack = transform.structured.match ops{["linalg.pack"]} in %arg1
        : (!transform.any_op) -> !transform.op<"linalg.pack">
      %paded, %expanded, %transpose = transform.structured.lower_pack %pack
        : (!transform.op<"linalg.pack">)
        -> (!transform.op<"tensor.pad">,
            !transform.op<"tensor.expand_shape">,
            !transform.op<"linalg.transpose">)

      %root = transform.structured.match ops{["linalg.generic"]} in %arg1
          : (!transform.any_op) -> !transform.any_op
      // Tile the lialg operation with parallel forall loop tiling [4, 4].
      %tiled_op, %forall_op = transform.structured.tile_using_forall %root num_threads [4, 4]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

      // Fuse the transpose operation into the tiled loop.
      transform.structured.fuse_into_containing_op %transpose into %forall_op
          : (!transform.op<"linalg.transpose">, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}

// -----
// For unpack op, we use lowerUnpadLikeWithExtractSlice = false to ensure no extract_slice is generated.
// This allows linalg.transpose to be fused as a consumer operation. In below testcase, linalg.transpose
// as a consumer operation is fused into the scf.forall loop.
module {
  // CHECK-label: func @fuse_unpack_as_consumer
  // CHECK:       scf.forall {{.*}} {
  // CHECK:         %[[CONSUMER:.*]] = linalg.generic
  // CHECK:         linalg.transpose ins(%[[CONSUMER]]
  // CHECK:         scf.forall.in_parallel
  // CHECK:       }
  func.func @fuse_unpack_as_consumer(%src: tensor<4x4x128x256xf32>, %other: tensor<4x4x128x256xf32>)
      -> tensor<128x256xf32> {
    %out = tensor.empty() : tensor<1x1x128x256xf32>
    %res = linalg.generic
        {indexing_maps = [affine_map<(i, j, k, l) -> (i, j, k, l)>,
                          affine_map<(i, j, k, l) -> (i, j, k, l)>,
                          affine_map<(i, j, k, l) -> (0, 0, k, l)>],
         iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%src, %other: tensor<4x4x128x256xf32>, tensor<4x4x128x256xf32>)
        outs(%out: tensor<1x1x128x256xf32>) {
      ^bb0(%unpack_elem: f32, %other_elem: f32, %out_elem: f32):
        %r = arith.addf %unpack_elem, %other_elem : f32
        linalg.yield %r : f32
    } -> tensor<1x1x128x256xf32>

    %dest = tensor.empty() : tensor<128x256xf32>
    %unpack = linalg.unpack %res inner_dims_pos = [0, 1] inner_tiles = [128, 256]
        into %dest : tensor<1x1x128x256xf32> -> tensor<128x256xf32>

    return %unpack : tensor<128x256xf32>
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      // Find and lower unpack operation.
      %unpack = transform.structured.match ops{["linalg.unpack"]} in %arg1
          : (!transform.any_op) -> !transform.op<"linalg.unpack">
      transform.structured.lower_unpack %unpack {lowerUnpadLikeWithExtractSlice = false}
        : (!transform.op<"linalg.unpack">)
        -> (!transform.op<"tensor.empty">,
            !transform.op<"linalg.transpose">,
            !transform.op<"tensor.collapse_shape">,
            !transform.op<"tensor.extract_slice">)

      %root = transform.structured.match ops{["linalg.generic"]} in %arg1
          : (!transform.any_op) -> !transform.any_op
      // Tile the lialg operation with parallel forall loop tiling [4, 4].
      %tiled_op, %forall_op = transform.structured.tile_using_forall %root num_threads [4, 4]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

      // Fuse the consumer operation into the tiled loop.
      %slice_op = transform.structured.match ops{["tensor.parallel_insert_slice"]} in %forall_op
          : (!transform.any_op) -> !transform.op<"tensor.parallel_insert_slice">
      transform.test.fuse_consumer %slice_op
        : (!transform.op<"tensor.parallel_insert_slice">) -> (!transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}

// -----
// For unpack op, by default lowerUnpadLikeWithExtractSlice = true, which generates extract_slice and blocks fusion.
// In below testcase, tensor.extract_slice as a consumer operation cannot be fused into the scf.forall loop.
module {
  // CHECK-label: func @fuse_unpack_as_consumer_blocked_by_extract_slice
  // CHECK:       %[[CONSUMER:.*]] = scf.forall {{.*}} {
  // CHECK:         %[[ADDF:.*]] = linalg.generic
  // CHECK:         scf.forall.in_parallel
  // CHECK:           tensor.parallel_insert_slice %[[ADDF]]
  // CHECK:       }
  // CHECK:       tensor.extract_slice %[[CONSUMER]]
  func.func @fuse_unpack_as_consumer_blocked_by_extract_slice(%src: tensor<4x4x128x256xf32>, %other: tensor<4x4x128x256xf32>)
      -> tensor<128x256xf32> {
    %out = tensor.empty() : tensor<1x1x128x256xf32>
    %res = linalg.generic
        {indexing_maps = [affine_map<(i, j, k, l) -> (i, j, k, l)>,
                          affine_map<(i, j, k, l) -> (i, j, k, l)>,
                          affine_map<(i, j, k, l) -> (0, 0, k, l)>],
         iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%src, %other: tensor<4x4x128x256xf32>, tensor<4x4x128x256xf32>)
        outs(%out: tensor<1x1x128x256xf32>) {
      ^bb0(%unpack_elem: f32, %other_elem: f32, %out_elem: f32):
        %r = arith.addf %unpack_elem, %other_elem : f32
        linalg.yield %r : f32
    } -> tensor<1x1x128x256xf32>

    %dest = tensor.empty() : tensor<128x256xf32>
    %unpack = linalg.unpack %res inner_dims_pos = [0, 1] inner_tiles = [128, 256]
        into %dest : tensor<1x1x128x256xf32> -> tensor<128x256xf32>

    return %unpack : tensor<128x256xf32>
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      // Find and lower unpack operation.
      %unpack = transform.structured.match ops{["linalg.unpack"]} in %arg1
          : (!transform.any_op) -> !transform.op<"linalg.unpack">
      transform.structured.lower_unpack %unpack
        : (!transform.op<"linalg.unpack">)
        -> (!transform.op<"tensor.empty">,
            !transform.op<"linalg.transpose">,
            !transform.op<"tensor.collapse_shape">,
            !transform.op<"tensor.extract_slice">)

      %root = transform.structured.match ops{["linalg.generic"]} in %arg1
          : (!transform.any_op) -> !transform.any_op
      // Tile the lialg operation with parallel forall loop tiling [4, 4].
      %tiled_op, %forall_op = transform.structured.tile_using_forall %root num_threads [4, 4]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

      // Fuse the consumer operation into the tiled loop.
      %slice_op = transform.structured.match ops{["tensor.parallel_insert_slice"]} in %forall_op
          : (!transform.any_op) -> !transform.op<"tensor.parallel_insert_slice">
      // Note that we cannot apply transform.test.fuse_consumer here because the extract_slice
      // is not qualified consumer operation. Forcing this will yeild "could not fetch consumer
      // to fuse" error.
      transform.yield
    }
  }
}
