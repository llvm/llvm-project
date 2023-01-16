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
  %0 = transform.structured.match ops{["linalg.matmul"]} in %module_op
  %1, %loops:3 = transform.structured.tile %0 [2, 2, 2]
  %2 = get_closest_isolated_parent %1 : (!pdl.operation) -> !pdl.operation
  transform.structured.vectorize %2
  transform.bufferization.one_shot_bufferize layout{IdentityLayoutMap} %module_op
    {bufferize_function_boundaries = true}
  %func = transform.structured.match ops{["func.func"]} in %module_op
  transform.vector.lower_vectors %func multireduction_lowering = "innerreduction"
}
