// RUN: mlir-opt %s --test-transform-dialect-interpreter -test-transform-dialect-erase-schedule --test-lower-to-llvm --split-input-file | FileCheck %s

// CHECK-LABEL: llvm.func @matmul_tensors
func.func @matmul_tensors(
  %arg0: tensor<2x4xf32>, %arg1: tensor<4x6xf32>, %arg2: tensor<2x6xf32>)
    -> tensor<2x6xf32> {
// CHECK-NOT: linalg
// CHECK: llvm.intr.fmuladd{{.*}}
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<2x4xf32>, tensor<4x6xf32>)
                     outs(%arg2: tensor<2x6xf32>)
    -> tensor<2x6xf32>
  return %0 : tensor<2x6xf32>
}

transform.sequence failures(propagate) {
^bb1(%module_op: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %module_op : (!transform.any_op) -> !transform.any_op
  %1, %loops:3 = transform.structured.tile %0 [2, 2, 2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
  %2 = get_parent_op %1 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
  transform.structured.vectorize %2 : (!transform.any_op) -> !transform.any_op
  %b = transform.bufferization.one_shot_bufferize layout{IdentityLayoutMap}
      %module_op {bufferize_function_boundaries = true}
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
