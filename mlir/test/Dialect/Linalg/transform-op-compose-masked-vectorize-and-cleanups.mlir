// RUN: mlir-opt %s -test-transform-dialect-interpreter -split-input-file | FileCheck %s

//      CHECK-LABEL: masked_matmul
func.func @masked_matmul(%module: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {

  //      CHECK: %[[MLHS:.*]] = vector.create_mask {{.*}} : vector<8x8xi1>
  //      CHECK: %[[LHS:.*]] = vector.transfer_read %{{.*}}, %[[MLHS]] {in_bounds = [true, true]} : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<8x8xf32> 
  //      CHECK: %[[MRHS:.*]] = vector.create_mask {{.*}} : vector<8x8xi1> 
  //      CHECK: %[[RHS:.*]] = vector.transfer_read %{{.*}}, %[[MRHS]] {in_bounds = [true, true]} : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<8x8xf32>
  //      CHECK: %[[MACC:.*]] = vector.create_mask {{.*}} : vector<8x8xi1>
  //      CHECK: %[[ACC:.*]] = vector.transfer_read {{.*}}, %[[MACC]] {in_bounds = [true, true]} : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<8x8xf32> 
  //      CHECK: %[[MRES:.*]] = vector.create_mask {{.*}} : vector<8x8x8xi1>
  //      CHECK: %[[RES:.*]] = vector.mask %[[MRES]] { vector.contract
  // CHECK-SAME:   : vector<8x8xf32>, vector<8x8xf32> into vector<8x8xf32>
  // CHECK-SAME:   : vector<8x8x8xi1> -> vector<8x8xf32>
  //      CHECK: vector.transfer_write %[[RES]], %{{.*}}, %[[MACC]] {in_bounds = [true, true]} : vector<8x8xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>> 
  linalg.matmul ins(%module, %arg1 : memref<?x?xf32>, memref<?x?xf32>) outs(%arg2 : memref<?x?xf32>)
  return
}

transform.sequence  failures(propagate) {
^bb0(%module: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %module 
    : (!transform.any_op) -> !transform.any_op
  %tiled_linalg_op, %loops:3 = transform.structured.tile %0[64, 128, 256] 
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
  %tiled_linalg_op_0, %loops_1:3 = transform.structured.tile %tiled_linalg_op[8, 8, 8] 
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
  transform.structured.masked_vectorize %tiled_linalg_op_0 vector_sizes [8, 8, 8] 
    : !transform.any_op

  %func = transform.structured.match ops{["func.func"]} in %module 
    : (!transform.any_op) -> !transform.any_op
  apply_patterns to %func {
    transform.apply_patterns.vector.lower_masked_transfers
    transform.apply_patterns.vector.transfer_permutation_patterns
    transform.apply_patterns.vector.reduction_to_contract
  } : !transform.any_op
}
