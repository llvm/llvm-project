// RUN: mlir-opt %s -transform-interpreter | FileCheck %s

func.func @mmt4d_to_fma(%A: tensor<16x16x8x1xf32>, %B: tensor<16x16x8x1xf32>, %C_in: tensor<16x16x8x8xf32>) -> tensor<16x16x8x8xf32> {
  %res = linalg.mmt4d
                   ins(%A, %B: tensor<16x16x8x1xf32>, tensor<16x16x8x1xf32>)
                   outs(%C_in: tensor<16x16x8x8xf32>)
                   -> tensor<16x16x8x8xf32>
  return %res : tensor<16x16x8x8xf32>
}


// CHECK-LABEL:     @mmt4d_to_fma
// CHECK-COUNT-8:         vector.fma

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.op<"func.func">

    %mmt4d = transform.structured.match ops{["linalg.mmt4d"]} in %func : (!transform.op<"func.func">) -> !transform.any_op

    // Step 1: Tile
    // Tile parallel dims
    %tiled_linalg_op_p, %loops:4 = transform.structured.tile_using_for %mmt4d tile_sizes [1, 1, 0, 8, 8, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    // Tile reduction dims
    %tiled_linalg_op_r, %loops2:2 = transform.structured.tile_using_for %tiled_linalg_op_p tile_sizes [0, 0, 1, 0, 0, 1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Step 2: Vectorize
    transform.structured.vectorize %tiled_linalg_op_r : !transform.any_op

    // Step 3: Simplify
    // vector.multi_reduction --> vector.contract
    // Generates a 6-dim vector.contract with the dim matching the original MMT4D Op
    // and with the following split into parallel and reduction dims:
    //    * parallel, parallel, reduction, parallel, parallel, reduction
    transform.apply_patterns to %func {
      transform.apply_patterns.vector.reduction_to_contract
      // Reduce the rank of xfer ops. This transforms vector.contract to be
      // more matmul-like and to enable the lowering to outer product Ops.
      transform.apply_patterns.vector.transfer_permutation_patterns
    } : !transform.op<"func.func">

    // Hoisting and LICM - not strictly required
    %func_h = transform.structured.hoist_redundant_vector_transfers %func
      : (!transform.op<"func.func">) -> !transform.op<"func.func">
    %all_loops = transform.structured.match interface{LoopLikeInterface} in %func_h
      : (!transform.op<"func.func">) -> !transform.any_op
    transform.apply_licm to %all_loops : !transform.any_op
    transform.loop.hoist_loop_invariant_subsets %all_loops : !transform.any_op

    // Simplify the 6-dim vector.contract into a 3-dim matmul-like
    // vector.contract with the following split into parallel and reduction
    // dims:
    //    * parallel, parallel, reduction
    transform.apply_patterns to %func_h {
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
      transform.apply_patterns.canonicalization
    } : !transform.op<"func.func">

    // Step 4: Lower vector.contract to vector.fma via vector.outerproduct
    transform.apply_patterns to %func_h {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
      transform.apply_patterns.vector.lower_outerproduct
    } : !transform.op<"func.func">
    transform.yield
  }
}
