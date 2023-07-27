// RUN: mlir-opt %s --test-transform-dialect-interpreter --split-input-file | FileCheck %s

// CHECK-LABEL: func @matmul_tensors
func.func @matmul_tensors(
  %arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<8x32xf32>)
    -> tensor<8x32xf32> {
// CHECK-NOT: linalg
// CHECK: vector.extract {{.*}} : vector<8x4xf32>
// CHECK: vector.store {{.*}} : memref<8x32xf32>, vector<4xf32>
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<8x16xf32>, tensor<16x32xf32>)
                     outs(%arg2: tensor<8x32xf32>)
    -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

transform.sequence failures(propagate) {
^bb1(%module_op: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %module_op : (!transform.any_op) -> !transform.any_op
  %1, %loops:3 = transform.structured.tile %0 [8, 4, 2]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
  %2 = get_parent_op %1 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
  transform.structured.vectorize %2 : (!transform.any_op) -> !transform.any_op
  %b = transform.bufferization.one_shot_bufferize
      layout{IdentityLayoutMap} %module_op
      {bufferize_function_boundaries = true, allow_return_allocs = true}
      : (!transform.any_op) -> !transform.any_op

  %f = transform.structured.match ops{["func.func"]} in %b
    : (!transform.any_op) -> !transform.any_op

  // TODO: group these lower-level controls into various properly named vector
  // lowering TD macros.
  transform.apply_patterns to %f {
    transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
  } : !transform.any_op

  transform.apply_patterns to %f {
    transform.apply_patterns.vector.transfer_permutation_patterns
  } : !transform.any_op

  transform.apply_patterns to %f {
    transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = "innerparallel"
  } : !transform.any_op

  transform.apply_patterns to %f {
    transform.apply_patterns.vector.split_transfer_full_partial split_transfer_strategy = "linalg-copy"
  } : !transform.any_op

  transform.apply_patterns to %f {
    transform.apply_patterns.vector.transfer_to_scf max_transfer_rank = 1 full_unroll = true
  } : !transform.any_op

  transform.apply_patterns to %f {
    transform.apply_patterns.vector.lower_transfer max_transfer_rank = 1
  } : !transform.any_op

  transform.apply_patterns to %f {
    transform.apply_patterns.vector.lower_shape_cast
  } : !transform.any_op

  transform.apply_patterns to %f {
    transform.apply_patterns.vector.lower_transpose lowering_strategy = "shuffle_1d"
  } : !transform.any_op
}
