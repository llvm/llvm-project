// RUN: mlir-opt -transform-interpreter -cse -mlir-print-local-scope -split-input-file -verify-diagnostics %s | FileCheck %s

// Check tile + fuse works with partial reduction outer parallel strategy.
// This also tests that the fusion logic correctly skips producers that are
// already inside the innermost loop (e.g., the rank-reducing slice of the
// fused fill), avoiding infinite loops in the fusion worklist.

module{
  func.func @tile_and_fuse_with_partial_reduction_outer_parallel(
      %arg0 : tensor<?x?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.0 : f32
    %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %empty = tensor.empty(%d0) : tensor<?xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
    %generic = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%arg0 : tensor<?x?xf32>) outs(%fill : tensor<?xf32>) {
      ^bb0(%b0 : f32, %b1 : f32):
        %0 = arith.addf %b0, %b1 : f32
        linalg.yield %0 : f32
    } -> tensor<?xf32>
    return %generic : tensor<?xf32>
  }
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %loop = transform.test.tile_and_fuse_outer_parallel_partial_reduction
      %generic tile_sizes = [128] 
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-LABEL: func @tile_and_fuse_with_partial_reduction_outer_parallel(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32>)
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//       CHECK:   %[[REDUCTION_NUM:.+]] = affine.apply affine_map<()[s0] -> (s0 ceildiv 128)>()[%[[D1]]]
//       CHECK:   %[[EMPTY:.+]] = tensor.empty(%[[D0]], %[[REDUCTION_NUM]])
//       CHECK:   %[[FORALL:.+]] = scf.forall (%[[IV0:[a-zA-Z0-9]+]]) =
//  CHECK-SAME:       shared_outs(%[[ITER_ARG:.+]] = %[[EMPTY]])
//   CHECK-DAG:     %[[TILESIZE:.+]] = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 128)>(%[[IV0]])[%[[D1]]]
//   CHECK-DAG:     %[[REDUCTION_IV:.+]] = affine.apply affine_map<()[s0] -> (s0 floordiv 128)>()[%[[IV0]]]
//   CHECK-DAG:     %[[ARG0_SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, %[[IV0]]] [%[[D0]], %[[TILESIZE]]] [1, 1]
//       CHECK:     %[[ITER_ARG_SLICE:.+]] = tensor.extract_slice %[[ITER_ARG]][0, %[[REDUCTION_IV]]] [%[[D0]], 1] [1, 1]
//       CHECK:     %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:         outs(%[[ITER_ARG_SLICE]] : tensor<?x1xf32>)
//       CHECK:     %[[REDUCING_SLICE:.+]] = tensor.extract_slice %[[FILL]][0, 0] [%[[D0]], 1] [1, 1] : tensor<?x1xf32> to tensor<?xf32>
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[ARG0_SLICE]] :
//  CHECK-SAME:         outs(%[[REDUCING_SLICE]] :
//       CHECK:     tensor.parallel_insert_slice %[[GENERIC]] into %[[ITER_ARG]]
//  CHECK-SAME:         [0, %[[REDUCTION_IV]]] [%[[D0]], 1] [1, 1]
//       CHECK:   %[[REDUCE:.+]] = linalg.reduce
//  CHECK-SAME:       ins(%[[FORALL]] :
//       CHECK:   return %[[REDUCE]]

// -----

// Check that linalg.index is correctly offset after partial reduction tiling.

module {
  func.func @partial_reduction_with_linalg_index(
      %arg0 : tensor<8x128xf32>) -> tensor<8xi32> {
    %c0_i32 = arith.constant 0 : i32
    %empty = tensor.empty() : tensor<8xi32>
    %fill = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<8xi32>) -> tensor<8xi32>
    %generic = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%arg0 : tensor<8x128xf32>) outs(%fill : tensor<8xi32>) {
      ^bb0(%b0 : f32, %b1 : i32):
        %idx = linalg.index 1 : index
        %idx_i32 = arith.index_cast %idx : index to i32
        %0 = arith.addi %idx_i32, %b1 : i32
        linalg.yield %0 : i32
    } -> tensor<8xi32>
    return %generic : tensor<8xi32>
  }
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %loop = transform.test.tile_and_fuse_outer_parallel_partial_reduction
      %generic tile_sizes = [32]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-LABEL: func @partial_reduction_with_linalg_index(
//       CHECK:   scf.forall (%[[IV0:[a-zA-Z0-9]+]]) =
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//       CHECK:       %[[LOCAL_IDX:.+]] = linalg.index 1 : index
//       CHECK:       affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%[[IV0]])[%[[LOCAL_IDX]]]
