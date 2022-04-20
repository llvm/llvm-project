// RUN: mlir-opt %s -test-linalg-transform-patterns=test-tile-fuse-and-distribute-options -split-input-file | FileCheck %s

//      CHECK: #[[MULMAP:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
//      CHECK: #[[ADDMAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
//      CHECK: func @fill_matmul_tensors(
// CHECK-SAME:    %[[TA:[0-9a-z]+]]: tensor<?x?xf32>
// CHECK-SAME:    %[[TB:[0-9a-z]+]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
func.func @fill_matmul_tensors(
  %arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
//  CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
//  CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//  CHECK-DAG: %[[BIDY:.*]] = gpu.block_id y
//  CHECK-DAG: %[[NBLOCKSY:.*]] = gpu.grid_dim y
//  CHECK-DAG: %[[BIDX:.*]] = gpu.block_id x
//  CHECK-DAG: %[[NBLOCKSX:.*]] = gpu.grid_dim x
//  CHECK-DAG: %[[INIT:.+]] = linalg.init_tensor
//      CHECK: %[[MUL:.+]] = affine.apply #[[MULMAP]]()[%[[BIDY]], %[[C8]]]
//      CHECK: %[[LBY:.+]] = affine.apply #[[ADDMAP]]()[%[[MUL]], %[[C0]]]
//      CHECK: %[[STEPY:.+]] = affine.apply #[[MULMAP]]()[%[[NBLOCKSY]], %[[C8]]]
//      CHECK: %[[TD0:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC0:.*]] = %[[INIT]]) -> (tensor<?x?xf32>) {
//      CHECK: %[[MUL:.+]] = affine.apply #[[MULMAP]]()[%[[BIDX]], %[[C8]]]
//      CHECK: %[[LBX:.+]] = affine.apply #[[ADDMAP]]()[%[[MUL]], %[[C0]]]
//      CHECK: %[[STEPX:.+]] = affine.apply #[[MULMAP]]()[%[[NBLOCKSX]], %[[C8]]]
//      CHECK:   %[[TD1:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC1:.*]] = %[[TC0]]) -> (tensor<?x?xf32>) {
//      CHECK:     %[[SLICE:.+]] = tensor.extract_slice %[[TC1]]
//      CHECK:     %[[FILL:.+]] = linalg.fill ins(%{{.+}}{{.*}}outs(%[[SLICE]]
//      CHECK:     %[[sTD2:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC2:.*]] = %[[FILL]]) -> (tensor<?x?xf32>) {
//      CHECK:       %[[sTA:.*]] = tensor.extract_slice %[[TA]][{{.*}}] : tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:       %[[sTB:.*]] = tensor.extract_slice %[[TB]][{{.*}}] : tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:       %[[sTC:.*]] = tensor.extract_slice %[[TC2]][{{.*}}] : tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:       %[[sTD:.*]] = linalg.matmul ins(%[[sTA]], %[[sTB]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME:                                  outs(%[[sTC]] : tensor<?x?xf32>)  -> tensor<?x?xf32>
//      CHECK:       %[[TD:.*]] = tensor.insert_slice %[[sTD]] into %[[TC2]][{{.*}}]  : tensor<?x?xf32> into tensor<?x?xf32>
//      CHECK:       scf.yield %[[TD]] : tensor<?x?xf32>
//      CHECK:     %[[TD2:.*]] = tensor.insert_slice %[[sTD2]] into %[[TC1]][{{.*}}]  : tensor<?x?xf32> into tensor<?x?xf32>
//      CHECK:     scf.yield %[[TD2]] : tensor<?x?xf32>
//      CHECK:   scf.yield %[[TD1]] : tensor<?x?xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %2 = linalg.init_tensor [%0, %1] : tensor<?x?xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = linalg.matmul {__internal_linalg_transform__ = "tensors_fuse_distribute1"}
       ins(%arg0, %arg1: tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%3: tensor<?x?xf32>)
    -> tensor<?x?xf32>

//      CHECK: return %[[TD0]] : tensor<?x?xf32>
  return %4 : tensor<?x?xf32>
}
