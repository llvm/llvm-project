// RUN: mlir-opt %s -transform-interpreter -canonicalize -split-input-file -verify-diagnostics | FileCheck %s

#map = affine_map<(d0) -> (d0)>
#floordiv3 = affine_map<(d0) -> (d0 floordiv 3)>

// CHECK-LABEL: func @tile_floordiv_tile_multiple_of_step
//       CHECK:   scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
//       CHECK:     tensor.extract_slice %{{.*}}[%[[IV]]] [6] [1] : tensor<12xf32> to tensor<6xf32>
//       CHECK:     tensor.extract_slice %{{.*}} [2] [1] : tensor<4xf32> to tensor<2xf32>
//       CHECK:     linalg.generic
func.func @tile_floordiv_tile_multiple_of_step(%arg0: tensor<12xf32>, %arg1: tensor<4xf32>, %out: tensor<12xf32>) -> tensor<12xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #floordiv3, #map], iterator_types = ["parallel"]}
    ins(%arg0, %arg1 : tensor<12xf32>, tensor<4xf32>) outs(%out : tensor<12xf32>) {
  ^bb0(%in: f32, %in_0: f32, %o: f32):
    %1 = arith.addf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<12xf32>
  return %0 : tensor<12xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:2 = transform.structured.tile_using_for %0 tile_sizes [6] : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">)
    transform.yield
  }
}

// -----

// The step being a multiple of the tile size is also aligned: every tile falls
// within a single `floordiv` bucket.

#map = affine_map<(d0) -> (d0)>
#floordiv4 = affine_map<(d0) -> (d0 floordiv 4)>

// CHECK-LABEL: func @tile_floordiv_step_multiple_of_tile
//       CHECK:   scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
//       CHECK:     tensor.extract_slice %{{.*}}[%[[IV]]] [2] [1] : tensor<12xf32> to tensor<2xf32>
//       CHECK:     tensor.extract_slice %{{.*}} [1] [1] : tensor<3xf32> to tensor<1xf32>
//       CHECK:     linalg.generic
func.func @tile_floordiv_step_multiple_of_tile(%arg0: tensor<12xf32>, %arg1: tensor<3xf32>, %out: tensor<12xf32>) -> tensor<12xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #floordiv4, #map], iterator_types = ["parallel"]}
    ins(%arg0, %arg1 : tensor<12xf32>, tensor<3xf32>) outs(%out : tensor<12xf32>) {
  ^bb0(%in: f32, %in_0: f32, %o: f32):
    %1 = arith.addf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<12xf32>
  return %0 : tensor<12xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:2 = transform.structured.tile_using_for %0 tile_sizes [2] : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">)
    transform.yield
  }
}

// -----

// A `floordiv` on a dimension that is not tiled is always safe.

#ident = affine_map<(d0, d1) -> (d0, d1)>
#floordiv3 = affine_map<(d0, d1) -> (d0, d1 floordiv 3)>

// CHECK-LABEL: func @tile_floordiv_untiled_dim
//       CHECK:   scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
//       CHECK:     linalg.generic
func.func @tile_floordiv_untiled_dim(%arg0: tensor<8x12xf32>, %arg1: tensor<8x4xf32>, %out: tensor<8x12xf32>) -> tensor<8x12xf32> {
  %0 = linalg.generic {indexing_maps = [#ident, #floordiv3, #ident], iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<8x12xf32>, tensor<8x4xf32>) outs(%out : tensor<8x12xf32>) {
  ^bb0(%in: f32, %in_0: f32, %o: f32):
    %1 = arith.addf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<8x12xf32>
  return %0 : tensor<8x12xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:2 = transform.structured.tile_using_for %0 tile_sizes [4, 0] : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">)
    transform.yield
  }
}

// -----

// `mod` obeys the same alignment rule as `floordiv`/`ceildiv`: with a tile size
// that is a multiple of the modulus the full `[0, modulus)` slice is taken and
// the tiled op re-applies the `mod` on local indices, so tiling is correct.

#map = affine_map<(d0) -> (d0)>
#mod3 = affine_map<(d0) -> (d0 mod 3)>

// CHECK-LABEL: func @tile_mod_aligned
//       CHECK:   scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
//       CHECK:     tensor.extract_slice %{{.*}}[%[[IV]]] [6] [1] : tensor<12xf32> to tensor<6xf32>
//       CHECK:     tensor.extract_slice %{{.*}} [3] [1] : tensor<3xf32> to tensor<3xf32>
//       CHECK:     linalg.generic
func.func @tile_mod_aligned(%arg0: tensor<12xf32>, %arg1: tensor<3xf32>, %out: tensor<12xf32>) -> tensor<12xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #mod3, #map], iterator_types = ["parallel"]}
    ins(%arg0, %arg1 : tensor<12xf32>, tensor<3xf32>) outs(%out : tensor<12xf32>) {
  ^bb0(%in: f32, %in_0: f32, %o: f32):
    %1 = arith.addf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<12xf32>
  return %0 : tensor<12xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:2 = transform.structured.tile_using_for %0 tile_sizes [6] : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">)
    transform.yield
  }
}

// -----

// `ceildiv` composes only from a step-aligned origin, so it is safe only when
// the step divides the tile size (unlike `floordiv`/`mod`).

#map = affine_map<(d0) -> (d0)>
#ceildiv3 = affine_map<(d0) -> (d0 ceildiv 3)>

// CHECK-LABEL: func @tile_ceildiv_tile_multiple_of_step
//       CHECK:   scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
//       CHECK:     tensor.extract_slice %{{.*}}[%[[IV]]] [6] [1] : tensor<12xf32> to tensor<6xf32>
//       CHECK:     tensor.extract_slice %{{.*}} [3] [1] : tensor<5xf32> to tensor<3xf32>
//       CHECK:     linalg.generic
func.func @tile_ceildiv_tile_multiple_of_step(%arg0: tensor<12xf32>, %arg1: tensor<5xf32>, %out: tensor<12xf32>) -> tensor<12xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #ceildiv3, #map], iterator_types = ["parallel"]}
    ins(%arg0, %arg1 : tensor<12xf32>, tensor<5xf32>) outs(%out : tensor<12xf32>) {
  ^bb0(%in: f32, %in_0: f32, %o: f32):
    %1 = arith.addf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<12xf32>
  return %0 : tensor<12xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:2 = transform.structured.tile_using_for %0 tile_sizes [6] : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">)
    transform.yield
  }
}

// -----

// A high-dimensional op with several distinct maps: the whole op is validated,
// so every semi-affine access must be aligned.
// Here `d0 floordiv 4` (tile 4) and `d2 mod 3` (tile 6) are both aligned.

#id = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#fdiv = affine_map<(d0, d1, d2) -> (d0 floordiv 4, d1)>
#mod = affine_map<(d0, d1, d2) -> (d2 mod 3)>

// CHECK-LABEL: func @tile_3d_generic_multi_map_aligned_floordiv_and_mod
//       CHECK:   scf.for %[[IV0:.*]] = %{{.*}} step %{{.*}}
//       CHECK:     scf.for %[[IV1:.*]] = %{{.*}} step %{{.*}}
//       CHECK:       scf.for %[[IV2:.*]] = %{{.*}} step %{{.*}}
//       CHECK:         tensor.extract_slice %{{.*}} [4, 3, 6] [1, 1, 1] : tensor<8x6x12xf32> to tensor<4x3x6xf32>
//       CHECK:         tensor.extract_slice %{{.*}} [1, 3] [1, 1] : tensor<2x6xf32> to tensor<1x3xf32>
//       CHECK:         tensor.extract_slice %{{.*}} [3] [1] : tensor<3xf32> to tensor<3xf32>
//       CHECK:         linalg.generic
func.func @tile_3d_generic_multi_map_aligned_floordiv_and_mod(%a0: tensor<8x6x12xf32>, %a1: tensor<2x6xf32>, %a2: tensor<3xf32>, %out: tensor<8x6x12xf32>) -> tensor<8x6x12xf32> {
  %0 = linalg.generic {indexing_maps = [#id, #fdiv, #mod, #id], iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%a0, %a1, %a2 : tensor<8x6x12xf32>, tensor<2x6xf32>, tensor<3xf32>) outs(%out : tensor<8x6x12xf32>) {
  ^bb0(%in0: f32, %in1: f32, %in2: f32, %o: f32):
    %s = arith.addf %in0, %in1 : f32
    %r = arith.addf %s, %in2 : f32
    linalg.yield %r : f32
  } -> tensor<8x6x12xf32>
  return %0 : tensor<8x6x12xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:4 = transform.structured.tile_using_for %0 tile_sizes [4, 3, 6] : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">, !transform.op<"scf.for">, !transform.op<"scf.for">)
    transform.yield
  }
}

// -----

// Provably misaligned `floordiv`: tile size 4 neither divides nor is divisible
// by the step 3. Tiling must be rejected.

#map = affine_map<(d0) -> (d0)>
#floordiv3 = affine_map<(d0) -> (d0 floordiv 3)>

func.func @negative_tile_floordiv_misaligned(%arg0: tensor<12xf32>, %arg1: tensor<4xf32>, %out: tensor<12xf32>) -> tensor<12xf32> {
  // expected-error @+3 {{'linalg.generic' op tiling is not supported for the semi-affine indexing map: tile size 4 for dimension d0 must divide or be divisible by the step 3}}
  // expected-error @+2 {{'linalg.generic' op failed to tile operation}}
  // expected-error @+1 {{'linalg.generic' op failed to generate tiling loops}}
  %0 = linalg.generic {indexing_maps = [#map, #floordiv3, #map], iterator_types = ["parallel"]}
    ins(%arg0, %arg1 : tensor<12xf32>, tensor<4xf32>) outs(%out : tensor<12xf32>) {
  ^bb0(%in: f32, %in_0: f32, %o: f32):
    %1 = arith.addf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<12xf32>
  return %0 : tensor<12xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:2 = transform.structured.tile_using_for %0 tile_sizes [4] : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">)
    transform.yield
  }
}

// -----

#map = affine_map<(d0) -> (d0)>
#ceildiv3 = affine_map<(d0) -> (d0 ceildiv 3)>

func.func @negative_tile_ceildiv_misaligned(%arg0: tensor<12xf32>, %arg1: tensor<5xf32>, %out: tensor<12xf32>) -> tensor<12xf32> {
  // expected-error @+3 {{'linalg.generic' op tiling is not supported for the semi-affine indexing map: tile size 4 for dimension d0 must be a multiple of the step 3}}
  // expected-error @+2 {{'linalg.generic' op failed to tile operation}}
  // expected-error @+1 {{'linalg.generic' op failed to generate tiling loops}}
  %0 = linalg.generic {indexing_maps = [#map, #ceildiv3, #map], iterator_types = ["parallel"]}
    ins(%arg0, %arg1 : tensor<12xf32>, tensor<5xf32>) outs(%out : tensor<12xf32>) {
  ^bb0(%in: f32, %in_0: f32, %o: f32):
    %1 = arith.addf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<12xf32>
  return %0 : tensor<12xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:2 = transform.structured.tile_using_for %0 tile_sizes [4] : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">)
    transform.yield
  }
}

// -----

// Unlike `floordiv`/`mod`, `ceildiv` is not safe when the step is a multiple of
// the tile size: the tile origin is not step-aligned.

#map = affine_map<(d0) -> (d0)>
#ceildiv4 = affine_map<(d0) -> (d0 ceildiv 4)>

func.func @negative_tile_ceildiv_step_multiple_of_tile(%arg0: tensor<12xf32>, %arg1: tensor<4xf32>, %out: tensor<12xf32>) -> tensor<12xf32> {
  // expected-error @+3 {{'linalg.generic' op tiling is not supported for the semi-affine indexing map: tile size 2 for dimension d0 must be a multiple of the step 4}}
  // expected-error @+2 {{'linalg.generic' op failed to tile operation}}
  // expected-error @+1 {{'linalg.generic' op failed to generate tiling loops}}
  %0 = linalg.generic {indexing_maps = [#map, #ceildiv4, #map], iterator_types = ["parallel"]}
    ins(%arg0, %arg1 : tensor<12xf32>, tensor<4xf32>) outs(%out : tensor<12xf32>) {
  ^bb0(%in: f32, %in_0: f32, %o: f32):
    %1 = arith.addf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<12xf32>
  return %0 : tensor<12xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:2 = transform.structured.tile_using_for %0 tile_sizes [2] : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">)
    transform.yield
  }
}

// -----

#map = affine_map<(d0) -> (d0)>
#mod3 = affine_map<(d0) -> (d0 mod 3)>

func.func @negative_tile_mod_misaligned(%arg0: tensor<12xf32>, %arg1: tensor<3xf32>, %out: tensor<12xf32>) -> tensor<12xf32> {
  // expected-error @+3 {{'linalg.generic' op tiling is not supported for the semi-affine indexing map: tile size 4 for dimension d0 must divide or be divisible by the step 3}}
  // expected-error @+2 {{'linalg.generic' op failed to tile operation}}
  // expected-error @+1 {{'linalg.generic' op failed to generate tiling loops}}
  %0 = linalg.generic {indexing_maps = [#map, #mod3, #map], iterator_types = ["parallel"]}
    ins(%arg0, %arg1 : tensor<12xf32>, tensor<3xf32>) outs(%out : tensor<12xf32>) {
  ^bb0(%in: f32, %in_0: f32, %o: f32):
    %1 = arith.addf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<12xf32>
  return %0 : tensor<12xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:2 = transform.structured.tile_using_for %0 tile_sizes [4] : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">)
    transform.yield
  }
}

// -----

// A semi-affine expression with a compound LHS (`d0 + d1`) over a tiled
// dimension cannot be proven tiling-safe and is conservatively rejected, even
// though the tile size would be aligned to the step for a bare dimension.

#ident = affine_map<(d0, d1) -> (d0, d1)>
#sum = affine_map<(d0, d1) -> ((d0 + d1) floordiv 4)>

func.func @negative_tile_floordiv_sum_lhs(%arg0: tensor<8x8xf32>, %arg1: tensor<4xf32>, %out: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+3 {{'linalg.generic' op tiling is not supported for the semi-affine indexing map: only a single iteration dimension divided by a positive constant step can be tiled over a tiled dimension}}
  // expected-error @+2 {{'linalg.generic' op failed to tile operation}}
  // expected-error @+1 {{'linalg.generic' op failed to generate tiling loops}}
  %0 = linalg.generic {indexing_maps = [#ident, #sum, #ident], iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<8x8xf32>, tensor<4xf32>) outs(%out : tensor<8x8xf32>) {
  ^bb0(%in: f32, %in_0: f32, %o: f32):
    %1 = arith.addf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:2 = transform.structured.tile_using_for %0 tile_sizes [4, 0] : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">)
    transform.yield
  }
}

// -----

// A single tiled dimension with a constant offset in the LHS (`d0 + 1`) is also
// a compound LHS and is conservatively rejected, even for an otherwise aligned
// tile size.

#map = affine_map<(d0) -> (d0)>
#offset = affine_map<(d0) -> ((d0 + 1) floordiv 3)>

func.func @negative_tile_floordiv_offset_lhs(%arg0: tensor<12xf32>, %arg1: tensor<5xf32>, %out: tensor<12xf32>) -> tensor<12xf32> {
  // expected-error @+3 {{'linalg.generic' op tiling is not supported for the semi-affine indexing map: only a single iteration dimension divided by a positive constant step can be tiled over a tiled dimension}}
  // expected-error @+2 {{'linalg.generic' op failed to tile operation}}
  // expected-error @+1 {{'linalg.generic' op failed to generate tiling loops}}
  %0 = linalg.generic {indexing_maps = [#map, #offset, #map], iterator_types = ["parallel"]}
    ins(%arg0, %arg1 : tensor<12xf32>, tensor<5xf32>) outs(%out : tensor<12xf32>) {
  ^bb0(%in: f32, %in_0: f32, %o: f32):
    %1 = arith.addf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<12xf32>
  return %0 : tensor<12xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:2 = transform.structured.tile_using_for %0 tile_sizes [6] : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">)
    transform.yield
  }
}

// -----

// High-dimensional op with several distinct maps where an aligned `floordiv`
// (`d0 floordiv 4`, tile 4) precedes a misaligned `mod` (`d2 mod 3`, tile 5).
// Validation must scan the whole op and reject on the later, offending map.

#id = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#fdiv = affine_map<(d0, d1, d2) -> (d0 floordiv 4, d1)>
#mod = affine_map<(d0, d1, d2) -> (d2 mod 3)>

func.func @negative_tile_3d_generic_misaligned_mod_in_later_map(%a0: tensor<8x6x12xf32>, %a1: tensor<2x6xf32>, %a2: tensor<3xf32>, %out: tensor<8x6x12xf32>) -> tensor<8x6x12xf32> {
  // expected-error @+3 {{'linalg.generic' op tiling is not supported for the semi-affine indexing map: tile size 5 for dimension d2 must divide or be divisible by the step 3}}
  // expected-error @+2 {{'linalg.generic' op failed to tile operation}}
  // expected-error @+1 {{'linalg.generic' op failed to generate tiling loops}}
  %0 = linalg.generic {indexing_maps = [#id, #fdiv, #mod, #id], iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%a0, %a1, %a2 : tensor<8x6x12xf32>, tensor<2x6xf32>, tensor<3xf32>) outs(%out : tensor<8x6x12xf32>) {
  ^bb0(%in0: f32, %in1: f32, %in2: f32, %o: f32):
    %s = arith.addf %in0, %in1 : f32
    %r = arith.addf %s, %in2 : f32
    linalg.yield %r : f32
  } -> tensor<8x6x12xf32>
  return %0 : tensor<8x6x12xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:4 = transform.structured.tile_using_for %0 tile_sizes [4, 3, 5] : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">, !transform.op<"scf.for">, !transform.op<"scf.for">)
    transform.yield
  }
}

// -----

// A `floordiv` on a statically shaped, tiled dimension (`d1`, tile 6, step 3)
// is validated even when a separate, dynamically shaped dimension (`d0`) is
// tiled but not part of the semi-affine map. The dynamic extent does not block
// checking the static access.

#id = affine_map<(d0, d1) -> (d0, d1)>
#floordiv3 = affine_map<(d0, d1) -> (d0, d1 floordiv 3)>

// CHECK-LABEL: func @tile_dynamic_dim_with_aligned_static_floordiv
//       CHECK:   scf.for
//       CHECK:     scf.for
//       CHECK:       linalg.generic
func.func @tile_dynamic_dim_with_aligned_static_floordiv(%arg0: tensor<?x12xf32>, %arg1: tensor<?x4xf32>, %out: tensor<?x12xf32>) -> tensor<?x12xf32> {
  %0 = linalg.generic {indexing_maps = [#id, #floordiv3, #id], iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<?x12xf32>, tensor<?x4xf32>) outs(%out : tensor<?x12xf32>) {
  ^bb0(%in: f32, %in_0: f32, %o: f32):
    %1 = arith.addf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<?x12xf32>
  return %0 : tensor<?x12xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:3 = transform.structured.tile_using_for %0 tile_sizes [8, 6] : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">, !transform.op<"scf.for">)
    transform.yield
  }
}

// -----

// Validate per dim check coverage. No blanket assumption in presence of dynamic dimensions.
// The static semi-affine dimension is still validated and rejected when misaligned.

#id = affine_map<(d0, d1) -> (d0, d1)>
#floordiv3 = affine_map<(d0, d1) -> (d0, d1 floordiv 3)>

func.func @negative_tile_dynamic_dim_with_misaligned_static_floordiv(%arg0: tensor<?x12xf32>, %arg1: tensor<?x4xf32>, %out: tensor<?x12xf32>) -> tensor<?x12xf32> {
  // expected-error @+3 {{'linalg.generic' op tiling is not supported for the semi-affine indexing map: tile size 4 for dimension d1 must divide or be divisible by the step 3}}
  // expected-error @+2 {{'linalg.generic' op failed to tile operation}}
  // expected-error @+1 {{'linalg.generic' op failed to generate tiling loops}}
  %0 = linalg.generic {indexing_maps = [#id, #floordiv3, #id], iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<?x12xf32>, tensor<?x4xf32>) outs(%out : tensor<?x12xf32>) {
  ^bb0(%in: f32, %in_0: f32, %o: f32):
    %1 = arith.addf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<?x12xf32>
  return %0 : tensor<?x12xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:3 = transform.structured.tile_using_for %0 tile_sizes [8, 4] : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">, !transform.op<"scf.for">)
    transform.yield
  }
}

// -----

// Fully dynamic shapes with static tile sizes: the loop range is dynamic, so
// the dimension is conservatively treated as tiled and the aligned `floordiv`
// (tile 6, step 3) is validated.

#map = affine_map<(d0) -> (d0)>
#floordiv3 = affine_map<(d0) -> (d0 floordiv 3)>

// CHECK-LABEL: func @tile_dynamic_shape_static_tile
//       CHECK:   scf.for
//       CHECK:     linalg.generic
func.func @tile_dynamic_shape_static_tile(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #floordiv3, #map], iterator_types = ["parallel"]}
    ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) outs(%out : tensor<?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %o: f32):
    %1 = arith.addf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:2 = transform.structured.tile_using_for %0 tile_sizes [6] : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">)
    transform.yield
  }
}

// -----

// A `floordiv` on a dynamically shaped dimension that is left untiled:
// its full extent has no static upper bound, so the tile size is dynamic
// (full dim size here) and the access is assumed valid regardless of the step.

#id = affine_map<(d0, d1) -> (d0, d1)>
#floordiv3 = affine_map<(d0, d1) -> (d0, d1 floordiv 3)>

// CHECK-LABEL: func @tile_untiled_dynamic_semi_affine_dim
//       CHECK:   scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
//       CHECK:     linalg.generic
func.func @tile_untiled_dynamic_semi_affine_dim(%arg0: tensor<8x?xf32>, %arg1: tensor<8x?xf32>, %out: tensor<8x?xf32>) -> tensor<8x?xf32> {
  %0 = linalg.generic {indexing_maps = [#id, #floordiv3, #id], iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<8x?xf32>, tensor<8x?xf32>) outs(%out : tensor<8x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %o: f32):
    %1 = arith.addf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<8x?xf32>
  return %0 : tensor<8x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:2 = transform.structured.tile_using_for %0 tile_sizes [4] : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">)
    transform.yield
  }
}

// -----

// A user-provided dynamic tile size (an SSA value) is assumed valid.

#map = affine_map<(d0) -> (d0)>
#floordiv3 = affine_map<(d0) -> (d0 floordiv 3)>

func.func private @get_tile_size() -> index

// CHECK-LABEL: func @tile_dynamic_tile_size_semi_affine
//       CHECK:   scf.for
//       CHECK:     linalg.generic
func.func @tile_dynamic_tile_size_semi_affine(%arg0: tensor<12xf32>, %arg1: tensor<4xf32>, %out: tensor<12xf32>) -> tensor<12xf32> {
  %sz = func.call @get_tile_size() : () -> index
  %0 = linalg.generic {indexing_maps = [#map, #floordiv3, #map], iterator_types = ["parallel"]}
    ins(%arg0, %arg1 : tensor<12xf32>, tensor<4xf32>) outs(%out : tensor<12xf32>) {
  ^bb0(%in: f32, %in_0: f32, %o: f32):
    %1 = arith.addf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<12xf32>
  return %0 : tensor<12xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %sz = transform.structured.match ops{["func.call"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:2 = transform.structured.tile_using_for %0 tile_sizes [%sz] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">)
    transform.yield
  }
}
