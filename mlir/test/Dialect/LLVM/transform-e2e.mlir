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
^bb1(%module_op: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %module_op : (!pdl.operation) -> !pdl.operation
  %1, %loops:3 = transform.structured.tile %0 [2, 2, 2] : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
  %2 = get_closest_isolated_parent %1 : (!pdl.operation) -> !pdl.operation
  transform.structured.vectorize %2
  %b = transform.bufferization.one_shot_bufferize layout{IdentityLayoutMap}
      %module_op {bufferize_function_boundaries = true}
      : (!pdl.operation) -> !pdl.operation

  %f = transform.structured.match ops{["func.func"]} in %b
    : (!pdl.operation) -> !pdl.operation

  // TODO: group these lower-level controls into various properly named vector
  // lowering TD macros.
  %func = transform.vector.lower_contraction %f
    lowering_strategy = "outerproduct" 
      : (!pdl.operation) -> !pdl.operation

  %func_2 = transform.vector.apply_transfer_permutation_patterns %func
      : (!pdl.operation) -> !pdl.operation

  %func_3 = transform.vector.lower_multi_reduction %func_2
    lowering_strategy = "innerparallel"
      : (!pdl.operation) -> !pdl.operation

  %func_4 = transform.vector.split_transfer_full_partial %func_3
    split_transfer_strategy = "linalg-copy"
      : (!pdl.operation) -> !pdl.operation

  %func_5 = transform.vector.transfer_to_scf %func_4
    max_transfer_rank = 1 full_unroll = true
      : (!pdl.operation) -> !pdl.operation

  %func_6 = transform.vector.lower_transfer %func_5
    max_transfer_rank = 1
      : (!pdl.operation) -> !pdl.operation

  %func_7 = transform.vector.lower_shape_cast %func_6
    : (!pdl.operation) -> !pdl.operation

  %func_8 = transform.vector.lower_transpose %func_7
    lowering_strategy = "shuffle"
      : (!pdl.operation) -> !pdl.operation
}
