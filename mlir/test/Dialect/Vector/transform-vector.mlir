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
^bb1(%module_op: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %module_op : (!pdl.operation) -> !pdl.operation
  %1, %loops:3 = transform.structured.tile %0 [8, 4, 2] : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
  %2 = get_closest_isolated_parent %1 : (!pdl.operation) -> !pdl.operation
  transform.structured.vectorize %2
  transform.bufferization.one_shot_bufferize %module_op

  %func = transform.structured.match ops{["func.func"]} in %module_op : (!pdl.operation) -> !pdl.operation
  transform.vector.lower_vectors %func multireduction_lowering = "innerreduction"
}
