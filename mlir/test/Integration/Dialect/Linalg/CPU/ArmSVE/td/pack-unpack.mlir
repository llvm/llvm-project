module @transforms attributes { transform.with_named_sequence } {

  //===----------------------------------------------------------------------===//
  // TD sequence _without_ vectorization
  //===----------------------------------------------------------------------===//
  transform.named_sequence @__transform_main_basic(%module: !transform.any_op {transform.consume}) {
    %pack = transform.structured.match ops{["linalg.pack"]} in %module : (!transform.any_op) -> !transform.any_op
    %unpack = transform.structured.match ops{["linalg.unpack"]} in %module : (!transform.any_op) -> !transform.any_op

    // 1.1 Tile the linalg.pack Op so that we can decompose it into e.g. tensor.pad
    //    and other lower-level Ops (see step 2.1)
    %tiled_pack_op_p, %loops_pack:2 = transform.structured.tile_using_for %pack tile_sizes [1, 1]
       : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // 1.2 Tile the linalg.unpack Op so that we can decompose it into e.g. tensor.pad
    //    and other lower-level Ops (see step 2.2)
    %tiled_unpack_op_p, %loops_unpack:2 = transform.structured.tile_using_for %unpack tile_sizes [4, 1]
       : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // 2.1. Decompose tiled PackOp into lower-level Ops
    %func_op_pack = transform.get_parent_op %tiled_pack_op_p {isolated_from_above} : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op_pack {
      transform.apply_patterns.linalg.decompose_pack_unpack
      transform.apply_patterns.linalg.decompose_pad
    } : !transform.op<"func.func">

    transform.apply_patterns to %func_op_pack {
      transform.apply_patterns.tensor.fold_tensor_subset_ops
      transform.apply_patterns.canonicalization
    } : !transform.op<"func.func">

    // 2.2. Decompose tiled UnpackOp into lower-level Ops
    %func_op_unpack = transform.get_parent_op %tiled_unpack_op_p {isolated_from_above} : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op_unpack {
      transform.apply_patterns.linalg.decompose_pack_unpack
    } : !transform.op<"func.func">

    transform.apply_patterns to %func_op_unpack {
      transform.apply_patterns.tensor.fold_tensor_subset_ops
      transform.apply_patterns.canonicalization
    } : !transform.op<"func.func">

   // 3. Bufferize before lowering to LLVM
   %bufferize = transform.bufferization.one_shot_bufferize %module
     {bufferize_function_boundaries=true} : (!transform.any_op) -> !transform.any_op

   // 4. Canonicalize
   %func_op_bufferized = transform.structured.match ops{["func.func"]} in %bufferize : (!transform.any_op) -> !transform.op<"func.func">
   transform.apply_patterns to %func_op_bufferized {
     transform.apply_patterns.canonicalization
   } : !transform.op<"func.func">

    transform.yield
  }

  //===----------------------------------------------------------------------===//
  // TD sequence _with_ vectorization
  //===----------------------------------------------------------------------===//
  transform.named_sequence @__transform_main_vectorized(%module: !transform.any_op {transform.consume}) {
    %pack = transform.structured.match ops{["linalg.pack"]} in %module : (!transform.any_op) -> !transform.any_op
    %unpack = transform.structured.match ops{["linalg.unpack"]} in %module : (!transform.any_op) -> !transform.any_op

    // 1.1 Tile the linalg.pack Op so that we can decompose it into e.g. tensor.pad
    //    and other lower-level Ops (see step 2.1)
    %tiled_pack_op_p, %loops_pack:2 = transform.structured.tile_using_for %pack tile_sizes [1, 1]
       : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // 1.2 Tile the linalg.unpack Op 
    %tiled_unpack_op_p, %loops_unpack:2 = transform.structured.tile_using_for %unpack tile_sizes [1, 1]
       : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // 2.1. Decompose tiled PackOp into lower-level Ops
    %func_op_pack = transform.get_parent_op %tiled_pack_op_p {isolated_from_above} : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op_pack {
      transform.apply_patterns.linalg.decompose_pack_unpack
      transform.apply_patterns.linalg.decompose_pad
    } : !transform.op<"func.func">

    transform.apply_patterns to %func_op_pack {
      transform.apply_patterns.tensor.fold_tensor_subset_ops
      transform.apply_patterns.canonicalization
    } : !transform.op<"func.func">

    // 2.2. Vectorize tiled UnpackOp into lower-level Ops
    %func_op_unpack = transform.get_parent_op %tiled_unpack_op_p {isolated_from_above} : (!transform.any_op) -> !transform.op<"func.func">
    transform.structured.vectorize %tiled_unpack_op_p vector_sizes  [1, 1, 4, [4]]  {assume_dynamic_dims_match_vec_sizes} : !transform.any_op

    transform.apply_patterns to %func_op_unpack {
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.lower_masked_transfers
      transform.apply_patterns.vector.sink_ops
    } : !transform.op<"func.func">

    // 3. Bufferize
    %bufferize = transform.bufferization.one_shot_bufferize %module
     {bufferize_function_boundaries=true} : (!transform.any_op) -> !transform.any_op

    // 4. Canonicalize
    %func_op_bufferized = transform.structured.match ops{["func.func"]} in %bufferize : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op_bufferized {
     transform.apply_patterns.canonicalization
    } : !transform.op<"func.func">

    transform.yield
  }
}
