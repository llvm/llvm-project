// RUN: mlir-opt %s --transform-interpreter --split-input-file -canonicalize | FileCheck %s

// CHECK-LABEL: func.func @fuse_unary
func.func @fuse_unary(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {

  //     CHECK: %[[RES:.*]] = scf.for
  //     CHECK:    scf.for
  //     CHECK:       linalg.exp
  //     CHECK:       linalg.add
  //     CHECK: return %[[RES]]
  %0 = linalg.exp ins(%arg0 : tensor<?x?xf32>)
                             outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.add ins(%0, %arg0 : tensor<?x?xf32>, tensor<?x?xf32>)
                             outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.add"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.fuse %0 tile_sizes [32, 32] interchange [0, 1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @fuse_unary
func.func @fuse_unary(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {

  //     CHECK: %[[PARTIAL_RES:.*]] = scf.for
  //     CHECK:     scf.for
  //     CHECK:       linalg.exp
  //     CHECK:       linalg.add
  //     CHECK: %[[RES:.*]] = scf.for {{.*}}%[[PARTIAL_RES]]
  //     CHECK:     scf.for
  //     CHECK:       linalg.exp
  //     CHECK:       linalg.add
  //     CHECK: return %[[RES]]
  %0 = linalg.exp ins(%arg0 : tensor<?x?xf32>)
                             outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.add ins(%0, %arg0 : tensor<?x?xf32>, tensor<?x?xf32>)
                             outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.add"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.fuse %0 tile_sizes [32, 32] interchange [0, 1]
      : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">, !transform.any_op)
    transform.loop.peel %loops#0 : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @fuse_unary_param
func.func @fuse_unary_param(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {

  //     CHECK: %[[RES:.*]] = scf.for
  //     CHECK:    scf.for
  //     CHECK:       linalg.exp
  //     CHECK:       linalg.add
  //     CHECK: return %[[RES]]
  %0 = linalg.exp ins(%arg0 : tensor<?x?xf32>)
                             outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.add ins(%0, %arg0 : tensor<?x?xf32>, tensor<?x?xf32>)
                             outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.add"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.param.constant 32 : i32 -> !transform.param<i32>
    %2 = transform.param.constant 1 : i32 -> !transform.param<i32>
    %3, %loops:2 = transform.structured.fuse %0 tile_sizes [%1, 32] interchange [0, %2]
      : (!transform.any_op, !transform.param<i32>, !transform.param<i32>) ->
      (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @fuse_unary_forall
func.func @fuse_unary_forall(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {

  //     CHECK: %[[RES:.*]] = scf.forall
  //     CHECK:       linalg.exp
  //     CHECK:       linalg.add
  //     CHECK: return %[[RES]]
  %0 = linalg.exp ins(%arg0 : tensor<?x?xf32>)
                             outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.add ins(%0, %arg0 : tensor<?x?xf32>, tensor<?x?xf32>)
                             outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.add"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop = transform.structured.fuse %0 tile_sizes [32, 32] {use_forall}
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @interchange_reduction
//  CHECK-SAME: (%[[INPUT:.+]]: tensor<12x7x25xf32>)
func.func @interchange_reduction(%input: tensor<12x7x25xf32>) -> tensor<12x25xf32> {
  %five = arith.constant 5.0 : f32
  %init = tensor.empty() : tensor<12x25xf32>

//   CHECK-DAG: %[[INIT:.+]] = tensor.empty()
//   CHECK-DAG: %[[C5:.+]] = arith.constant 5 : index
//   CHECK-DAG: %[[C7:.+]] = arith.constant 7 : index
//   CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
//       CHECK: %[[RES:.*]] = scf.for %[[IV0:.+]] = %{{.+}} to %{{.+}} step %[[C5]] iter_args(%[[FOR_ARG0:.+]] = %[[INIT]])
//       CHECK:   scf.for %[[IV1:.+]] = %{{.+}} to %{{.+}} step %[[C7]] iter_args(%[[FOR_ARG1:.+]] = %[[FOR_ARG0]])
//       CHECK:     %[[OUT_SLICE0:.+]] = tensor.extract_slice %[[INPUT]][%[[IV0]], 0, %[[IV1]]]
//       CHECK:     %[[OUT_SLICE1:.+]] = tensor.extract_slice %[[FOR_ARG1]][%[[IV0]], %[[IV1]]]
//       CHECK:     %[[FILL:.+]] = linalg.fill {{.+}} outs(%[[OUT_SLICE1]] : tensor<?x?xf32>)
//       CHECK:     scf.for %[[IV2:.+]] = %{{.+}} to %{{.+}} step %[[C4]] iter_args(%[[FOR_ARG2:.+]] = %[[FILL]])
//       CHECK:       %[[IN_SLICE:.+]] = tensor.extract_slice %[[OUT_SLICE0]]
//       CHECK:       %[[OUT_SLICE2:.+]] = tensor.extract_slice %[[FOR_ARG2]][0, 0]
//       CHECK:       linalg.generic {{.+}} ins(%[[IN_SLICE]] : tensor<?x?x?xf32>) outs(%[[OUT_SLICE2]] : tensor<?x?xf32>)
//       CHECK: return %[[RES]]

  %fill = linalg.fill ins(%five : f32) outs(%init : tensor<12x25xf32>) -> tensor<12x25xf32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d2)>],
    iterator_types = ["parallel", "reduction", "parallel"]
  } ins(%input : tensor<12x7x25xf32>) outs(%fill : tensor<12x25xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %2 = arith.addf %arg0, %arg1 : f32
    linalg.yield %2 : f32
  } -> tensor<12x25xf32>
  func.return %0 : tensor<12x25xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.fuse %0 tile_sizes [5, 0, 7] interchange [0, 2, 1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %2, %loops_2 = transform.structured.tile_using_for %1 tile_sizes [0, 4]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @unpack_elemwise
// CHECK:         %[[RES:.*]] = scf.for
// CHECK:           scf.for
// CHECK:             linalg.unpack
// CHECK:             linalg.exp
// CHECK:         return %[[RES]]
func.func @unpack_elemwise(%arg0: tensor<16x48x8x8xf32>, %arg1: tensor<128x384xf32>) -> tensor<128x384xf32> {
  %0 = tensor.empty() : tensor<128x384xf32>
  %1 = linalg.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %0
      : tensor<16x48x8x8xf32> -> tensor<128x384xf32>
  %2 = linalg.exp ins(%1: tensor<128x384xf32>)
                             outs(%arg1: tensor<128x384xf32>) -> tensor<128x384xf32>
  return %2 : tensor<128x384xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.exp"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.fuse %0 tile_sizes [16, 32] interchange [0, 1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @pack_elemwise
// CHECK:         %[[RES:.*]] = scf.for
// CHECK:           scf.for
// CHECK:             linalg.pack
// CHECK:             linalg.exp
// CHECK:         return %[[RES]]
func.func @pack_elemwise(%arg0: tensor<128x384xf32>, %arg1: tensor<16x48x8x8xf32>) -> tensor<16x48x8x8xf32> {
  %0 = tensor.empty() : tensor<16x48x8x8xf32>
  %1 = linalg.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %0
      : tensor<128x384xf32> -> tensor<16x48x8x8xf32>
  %2 = linalg.exp ins(%1: tensor<16x48x8x8xf32>)
                             outs(%arg1: tensor<16x48x8x8xf32>) -> tensor<16x48x8x8xf32>
  return %2 : tensor<16x48x8x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.exp"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.fuse %0 tile_sizes [3, 5, 0, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @nofuse_pack_elemwise
// CHECK:         linalg.pack
// CHECK:         %[[RES:.*]] = scf.for
// CHECK:           scf.for
// CHECK:             linalg.exp
// CHECK:         return %[[RES]]
func.func @nofuse_pack_elemwise(%arg0: tensor<128x384xf32>, %arg1: tensor<16x48x8x8xf32>) -> tensor<16x48x8x8xf32> {
  %0 = tensor.empty() : tensor<16x48x8x8xf32>
  %1 = linalg.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %0
      : tensor<128x384xf32> -> tensor<16x48x8x8xf32>
  %2 = linalg.exp ins(%1: tensor<16x48x8x8xf32>)
                             outs(%arg1: tensor<16x48x8x8xf32>) -> tensor<16x48x8x8xf32>
  return %2 : tensor<16x48x8x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.exp"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:3 = transform.structured.fuse %0 tile_sizes [3, 5, 2, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @fuse_through_slice
func.func @fuse_through_slice(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {

  //     CHECK: %[[RES:.*]] = scf.for
  //     CHECK:     scf.for
  //     CHECK:       linalg.exp
  //     CHECK:       linalg.add
  //     CHECK: return %[[RES]]
  %0 = linalg.exp ins(%arg0 : tensor<?x?xf32>)
                             outs(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %arg1, %c0 : tensor<?x?xf32>
  %dim1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %1 = tensor.extract_slice %0 [1, 1] [%dim0, %dim1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %2 = linalg.add ins(%1, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
                             outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.add"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.fuse %0 tile_sizes [32, 32] interchange [0, 1] {apply_cleanup}
      : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @fuse_through_slice_and_cast_chain
func.func @fuse_through_slice_and_cast_chain(%arg0: tensor<100x100xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {

  //     CHECK: %[[RES:.*]] = scf.for
  //     CHECK:     scf.for
  //     CHECK:       linalg.exp
  //     CHECK:       linalg.add
  //     CHECK: return %[[RES]]
  %0 = linalg.exp ins(%arg0 : tensor<100x100xf32>)
                             outs(%arg0: tensor<100x100xf32>) -> tensor<100x100xf32>
  %1 = tensor.cast %0 : tensor<100x100xf32> to tensor<100x?xf32>
  %2 = tensor.extract_slice %1 [1, 1] [98, 98] [1, 1] : tensor<100x?xf32> to tensor<98x98xf32>
  %3 = tensor.cast %2 : tensor<98x98xf32> to tensor<?x?xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %arg1, %c0 : tensor<?x?xf32>
  %dim1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %4 = tensor.extract_slice %3 [1, 1] [%dim0, %dim1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %5 = linalg.add ins(%4, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
                             outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %5 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.add"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.fuse %0 tile_sizes [32, 32] interchange [0, 1] {apply_cleanup}
      : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @fuse_unrelated_slice
func.func @fuse_unrelated_slices(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<10x10xf32>) {

  //     CHECK: %[[SLICE1:.+]] = tensor.extract_slice
  //     CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[SLICE1]]
  //     CHECK: %[[RES:.*]] = scf.for
  //     CHECK:     scf.for
  //     CHECK:       linalg.exp
  //     CHECK:       linalg.add
  //     CHECK: return %[[RES]], %[[SLICE2]]
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %arg1, %c0 : tensor<?x?xf32>
  %dim1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %slice1 = tensor.extract_slice %arg0 [1, 1] [%dim0, %dim1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %slice2 = tensor.extract_slice %slice1 [1, 1] [10, 10] [1, 1] : tensor<?x?xf32> to tensor<10x10xf32>
  %0 = linalg.exp ins(%arg0 : tensor<?x?xf32>)
                             outs(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = tensor.extract_slice %0 [1, 1] [%dim0, %dim1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %2 = linalg.add ins(%1, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
                             outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %2, %slice2 : tensor<?x?xf32>, tensor<10x10xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.add"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.fuse %0 tile_sizes [32, 32] interchange [0, 1] {apply_cleanup}
      : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @bubble_up_extract_slice_through_expand_shape
//     CHECK: scf.for %[[X:[A-Za-z0-9]+]] = {{.*}}
//     CHECK:   scf.for %[[Y:[A-Za-z0-9]+]] = {{.*}}
//     CHECK:     scf.for %[[Z:[A-Za-z0-9]+]] = {{.*}}
//     CHECK:       %[[LINEAR_IDX:.+]] = affine.linearize_index disjoint [%[[X]], %[[Y]], %[[Z]]] by (2, 3, 10)
//     CHECK:       %[[SLICE:.+]] = tensor.extract_slice %{{.*}}[%[[LINEAR_IDX]]] [5] [1] : tensor<60xf32> to tensor<5xf32>
//     CHECK:       %[[EXPAND:.+]] = tensor.expand_shape %[[SLICE]] {{\[\[}}0, 1, 2]] output_shape [1, 1, 5]
//     CHECK:       linalg.exp ins(%[[EXPAND]]
func.func @bubble_up_extract_slice_through_expand_shape(%0: tensor<60xf32>) -> tensor<2x3x10xf32> {
  %expand = tensor.expand_shape %0 [[0, 1, 2]] output_shape [2, 3, 10] : tensor<60xf32> into tensor<2x3x10xf32>
  %empty = tensor.empty() : tensor<2x3x10xf32>
  %exp = linalg.exp ins(%expand : tensor<2x3x10xf32>) outs(%empty : tensor<2x3x10xf32>) -> tensor<2x3x10xf32>
  return %exp : tensor<2x3x10xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.exp"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %transformed, %loops:3 = transform.structured.fuse %0 tile_sizes [1, 1, 5] interchange [0, 1, 2] {apply_cleanup} : 
      (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">, !transform.any_op, !transform.any_op)
    transform.yield 
  }
}

// -----

// CHECK-LABEL: func.func @bubble_up_extract_slice_through_expand_shape_full_inner_dim
//     CHECK: scf.for %[[X:[A-Za-z0-9]+]] = {{.*}}
//     CHECK:   scf.for %[[Y:[A-Za-z0-9]+]] = {{.*}}
//     CHECK:       %[[LINEAR_IDX:.+]] = affine.linearize_index disjoint [%[[X]], %[[Y]]{{.*}} by (3, 4, 10)
//     CHECK:       %[[SLICE:.+]] = tensor.extract_slice %{{.*}}[%[[LINEAR_IDX]]] [20] [1] : tensor<120xf32> to tensor<20xf32>
//     CHECK:       %[[EXPAND:.+]] = tensor.expand_shape %[[SLICE]] {{\[\[}}0, 1, 2]] output_shape [1, 2, 10]
//     CHECK:       linalg.exp ins(%[[EXPAND]]
func.func @bubble_up_extract_slice_through_expand_shape_full_inner_dim(%0: tensor<120xf32>) -> tensor<3x4x10xf32> {
  %expand = tensor.expand_shape %0 [[0, 1, 2]] output_shape [3, 4, 10] : tensor<120xf32> into tensor<3x4x10xf32>
  %empty = tensor.empty() : tensor<3x4x10xf32>
  %exp = linalg.exp ins(%expand : tensor<3x4x10xf32>) outs(%empty : tensor<3x4x10xf32>) -> tensor<3x4x10xf32>
  return %exp : tensor<3x4x10xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.exp"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %transformed, %loops:2 = transform.structured.fuse %0 tile_sizes [1, 2, 0] interchange [0, 1, 2] {apply_cleanup} :
      (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">, !transform.any_op)
    transform.yield 
  }
}

// -----

// CHECK-LABEL: func.func @no_bubble_up_extract_slice_through_expand_shape_non_contiguous
//     CHECK: tensor.expand_shape
//     CHECK: scf.for
//     CHECK:   scf.for
//     CHECK:     scf.for
//     CHECK:       linalg.exp
func.func @no_bubble_up_extract_slice_through_expand_shape_non_contiguous(%0: tensor<120xf32>) -> tensor<3x4x10xf32> {
  %expand = tensor.expand_shape %0 [[0, 1, 2]] output_shape [3, 4, 10] : tensor<120xf32> into tensor<3x4x10xf32>
  %empty = tensor.empty() : tensor<3x4x10xf32>
  %exp = linalg.exp ins(%expand : tensor<3x4x10xf32>) outs(%empty : tensor<3x4x10xf32>) -> tensor<3x4x10xf32>
  return %exp : tensor<3x4x10xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.exp"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %transformed, %loops:3 = transform.structured.fuse %0 tile_sizes [1, 2, 5] interchange [0, 1, 2] {apply_cleanup} :
      (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">, !transform.any_op, !transform.any_op)
    transform.yield 
  }
}

// -----

// CHECK-LABEL: func.func @bubble_up_extract_slice_through_expand_shape_multiple_expanded_dims
//     CHECK: %[[C0:.+]] = arith.constant 0 : index
//     CHECK: scf.for %[[X:[A-Za-z0-9]+]] = {{.*}}
//     CHECK:   scf.for %[[Y:[A-Za-z0-9]+]] = {{.*}}
//     CHECK:     scf.for %[[Z:[A-Za-z0-9]+]] = {{.*}}
//     CHECK:       scf.for %[[W:[A-Za-z0-9]+]] = {{.*}}
//     CHECK:       %[[LINEAR_IDX0:.+]] = affine.linearize_index disjoint [%[[X]], %[[Y]], %[[C0]]] by (3, 4, 10)
//     CHECK:       %[[LINEAR_IDX1:.+]] = affine.linearize_index disjoint [%[[Z]], %[[W]]] by (7, 8)
//     CHECK:       %[[SLICE:.+]] = tensor.extract_slice %{{.*}}[%[[LINEAR_IDX0]], %[[LINEAR_IDX1]]] [20, 4] [1, 1] : tensor<120x56xf32> to tensor<20x4xf32>
//     CHECK:       %[[EXPAND:.+]] = tensor.expand_shape %[[SLICE]] {{\[\[}}0, 1, 2], [3, 4]] output_shape [1, 2, 10, 1, 4]
//     CHECK:       linalg.exp ins(%[[EXPAND]]
module {
  func.func @bubble_up_extract_slice_through_expand_shape_multiple_expanded_dims(%0: tensor<120x56xf32>) -> tensor<3x4x10x7x8xf32> {
    %expand = tensor.expand_shape %0 [[0, 1, 2], [3, 4]] output_shape [3, 4, 10, 7, 8] : tensor<120x56xf32> into tensor<3x4x10x7x8xf32>
    %empty = tensor.empty() : tensor<3x4x10x7x8xf32>
    %exp = linalg.exp ins(%expand : tensor<3x4x10x7x8xf32>) outs(%empty : tensor<3x4x10x7x8xf32>) -> tensor<3x4x10x7x8xf32>
    return %exp : tensor<3x4x10x7x8xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.exp"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %transformed, %loops:4 = transform.structured.fuse %0 tile_sizes [1, 2, 0, 1, 4] interchange [0, 1, 2, 3, 4] {apply_cleanup} :
      (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield 
  }
}

// -----

// CHECK-LABEL: func.func @bubble_up_extract_slice_through_expand_shape_and_fuse_with_expand_producer
//     CHECK: scf.for %[[X:[A-Za-z0-9]+]] = {{.*}}
//     CHECK:    %[[LINEAR_IDX:.+]] = affine.linearize_index disjoint [%[[X]], {{.*}} by (8, 32)
//     CHECK:    %[[SLICE:.+]] = tensor.extract_slice %{{.*}}[0, 0, %[[LINEAR_IDX]]] [1, 1800, 32] [1, 1, 1] : tensor<1x1800x256xf32> to tensor<1x1800x32xf32>
//     CHECK:    %[[ABS:.+]] = linalg.abs ins(%[[SLICE]]
//     CHECK:    %[[EXPAND:.+]] = tensor.expand_shape %[[ABS]] {{\[\[}}0], [1], [2, 3]] output_shape [1, 1800, 1, 32]
//     CHECK:    linalg.exp ins(%[[EXPAND]]
module {
  func.func @bubble_up_extract_slice_through_expand_shape_and_fuse_with_expand_producer(%0: tensor<1x1800x256xf32>) -> tensor<1x1800x8x32xf32> {
    %empty1 = tensor.empty() : tensor<1x1800x256xf32>
    %exp1 = linalg.abs ins(%0 : tensor<1x1800x256xf32>) outs(%empty1 : tensor<1x1800x256xf32>) -> tensor<1x1800x256xf32>
    %expand = tensor.expand_shape %exp1 [[0], [1], [2, 3]] output_shape [1, 1800, 8, 32] : tensor<1x1800x256xf32> into tensor<1x1800x8x32xf32>
    %empty2 = tensor.empty() : tensor<1x1800x8x32xf32>
    %exp2 = linalg.exp ins(%expand : tensor<1x1800x8x32xf32>) outs(%empty2 : tensor<1x1800x8x32xf32>) -> tensor<1x1800x8x32xf32>
    return %exp2 : tensor<1x1800x8x32xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.exp"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %transformed, %loops:1 = transform.structured.fuse %0 tile_sizes [0, 0, 1, 0] interchange [0, 1, 2, 3] {apply_cleanup} :
      (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">)
    transform.yield 
  }
}

// -----

// CHECK-LABEL: func.func @no_bubble_up_extract_slice_through_expand_shape_on_cleanup_false
//     CHECK: %[[EXPAND:.+]] = tensor.expand_shape {{.*}} {{\[\[}}0, 1, 2]] output_shape [2, 3, 10]
//     CHECK: scf.for %[[X:[A-Za-z0-9]+]] = {{.*}}
//     CHECK:   scf.for %[[Y:[A-Za-z0-9]+]] = {{.*}}
//     CHECK:     scf.for %[[Z:[A-Za-z0-9]+]] = {{.*}}
//     CHECK:       %[[SLICE:.+]] = tensor.extract_slice %[[EXPAND]]{{.*}} [1, 1, 5] [1, 1, 1] : tensor<2x3x10xf32> to tensor<1x1x5xf32>
//     CHECK:       linalg.exp ins(%[[SLICE]]
func.func @no_bubble_up_extract_slice_through_expand_shape_on_cleanup_false(%0: tensor<60xf32>) -> tensor<2x3x10xf32> {
  %expand = tensor.expand_shape %0 [[0, 1, 2]] output_shape [2, 3, 10] : tensor<60xf32> into tensor<2x3x10xf32>
  %empty = tensor.empty() : tensor<2x3x10xf32>
  %exp = linalg.exp ins(%expand : tensor<2x3x10xf32>) outs(%empty : tensor<2x3x10xf32>) -> tensor<2x3x10xf32>
  return %exp : tensor<2x3x10xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.exp"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %transformed, %loops:3 = transform.structured.fuse %0 tile_sizes [1, 1, 5] interchange [0, 1, 2] :
      (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">, !transform.any_op, !transform.any_op)
    transform.yield 
  }
}

// -----

// CHECK-LABEL:   func.func @bubble_up_extract_slice_through_collapse_shape(
// CHECK:      scf.for %[[X:[A-Za-z0-9]+]] = {{.*}} -> (tensor<8x1800x32xf32>) {
// CHECK:             %[[EXTRACT:.*]] = tensor.extract_slice
// CHECK:             %[[COLLAPSE:.*]] = tensor.collapse_shape %[[EXTRACT]]
// CHECK:             %[[EXP1:.*]] = linalg.exp ins(%[[COLLAPSE]]
func.func @bubble_up_extract_slice_through_collapse_shape(%0: tensor<1x8x1800x32xf32>) -> tensor<8x1800x32xf32> {
  %expand = tensor.collapse_shape %0 [[0, 1], [2], [3]] : tensor<1x8x1800x32xf32> into tensor<8x1800x32xf32>
  %empty = tensor.empty() : tensor<8x1800x32xf32>
  %exp = linalg.exp ins(%expand : tensor<8x1800x32xf32>) outs(%empty : tensor<8x1800x32xf32>) -> tensor<8x1800x32xf32>
  return %exp : tensor<8x1800x32xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.exp"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %transformed, %loops:1 = transform.structured.fuse %0 tile_sizes [1, 0, 0] interchange [0, 1, 2] {apply_cleanup} : 
      (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">)
    transform.yield 
  }
}

// -----

// CHECK-LABEL:   func.func @bubble_up_extract_slice_through_collapse_shape_with_collapse_producer(
// CHECK:           scf.for %[[X:[A-Za-z0-9]+]] = {{.*}}
// CHECK:             %[[EXTRACT:.*]] = tensor.extract_slice
// CHECK:             %[[ABS:.*]] = linalg.abs ins(%[[EXTRACT]]
// CHECK:             %[[COLLAPSE:.*]] = tensor.collapse_shape %[[ABS]]
// CHECK:             %[[EXP:.*]] = linalg.exp ins(%[[COLLAPSE]]
func.func @bubble_up_extract_slice_through_collapse_shape_with_collapse_producer(%0: tensor<1x8x1800x32xf32>) -> tensor<8x1800x32xf32> {
  %empty1 = tensor.empty() : tensor<1x8x1800x32xf32>
  %abs = linalg.abs ins(%0 : tensor<1x8x1800x32xf32>) outs(%empty1 : tensor<1x8x1800x32xf32>) -> tensor<1x8x1800x32xf32>
  %expand = tensor.collapse_shape %abs [[0, 1], [2], [3]] : tensor<1x8x1800x32xf32> into tensor<8x1800x32xf32>
  %empty2 = tensor.empty() : tensor<8x1800x32xf32>
  %exp = linalg.exp ins(%expand : tensor<8x1800x32xf32>) outs(%empty2 : tensor<8x1800x32xf32>) -> tensor<8x1800x32xf32>
  return %exp : tensor<8x1800x32xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.exp"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %transformed, %loops:1 = transform.structured.fuse %0 tile_sizes [1, 0, 0] interchange [0, 1, 2] {apply_cleanup} : 
      (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">)
    transform.yield 
  }
}
