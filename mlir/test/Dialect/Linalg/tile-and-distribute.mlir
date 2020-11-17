// RUN: mlir-opt %s -test-linalg-transform-patterns=test-tile-and-distribute-options -split-input-file | FileCheck %s

func @gemm1(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>)
{
  linalg.matmul {__internal_linalg_transform__ = "distribute1"}
    ins(%a, %b: memref<?x?xf32>, memref<?x?xf32>)
   outs(%c: memref<?x?xf32>)
  return
}
//  CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0] -> (s0 * 8)>
//      CHECK: func @gemm1(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<?x?xf32>
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<?x?xf32>
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9_]*]]: memref<?x?xf32>
//      CHECK: %[[BIDY:.*]] = "gpu.block_id"() {dimension = "y"}
//      CHECK: %[[BIDX:.*]] = "gpu.block_id"() {dimension = "x"}
//      CHECK: scf.for %[[ARG3:.*]] =
//      CHECK:   %[[OFFSETY:.*]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//      CHECK:   %[[SV1:.*]] = subview %[[ARG0]][%[[OFFSETY]], %[[ARG3]]]
//      CHECK:   %[[OFFSETX:.*]] = affine.apply #[[MAP0]]()[%[[BIDX]]]
//      CHECK:   %[[SV2:.*]] = subview %[[ARG1]][%[[ARG3]], %[[OFFSETX]]]
//      CHECK:   %[[OFFSETY_2:.*]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//      CHECK:   %[[OFFSETX:.*]] = affine.apply #[[MAP0]]()[%[[BIDX]]]
//      CHECK:   %[[SV3:.*]] = subview %[[ARG2]][%[[OFFSETY_2]], %[[OFFSETX]]]
//      CHECK:   linalg.matmul ins(%[[SV1]], %[[SV2]]{{.*}} outs(%[[SV3]]

// -----

func @gemm2(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>)
{
  linalg.matmul  {__internal_linalg_transform__ = "distribute2"}
    ins(%a, %b: memref<?x?xf32>, memref<?x?xf32>)
   outs(%c:memref<?x?xf32>)
  return
}
//  CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0] -> (s0 * 8)>
//      CHECK: func @gemm2(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<?x?xf32>
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<?x?xf32>
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9_]*]]: memref<?x?xf32>
//  CHECK-DAG: %[[BIDY:.*]] = "gpu.block_id"() {dimension = "y"}
//  CHECK-DAG: %[[BIDX:.*]] = "gpu.block_id"() {dimension = "x"}
//      CHECK: %[[ITERY:.*]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//      CHECK: %[[ITERX:.*]] = affine.apply #[[MAP0]]()[%[[BIDX]]]
//      CHECK: %[[INBOUNDSY:.*]] = cmpi "slt", %[[ITERY]], %{{.*}}
//      CHECK: %[[INBOUNDSX:.*]] = cmpi "slt", %[[ITERX]], %{{.*}}
//      CHECK: %[[INBOUNDS:.*]] = and %[[INBOUNDSY]], %[[INBOUNDSX]]
//      CHECK: scf.if %[[INBOUNDS]]
//      CHECK:   scf.for %[[ARG3:.*]] =
//      CHECK:     %[[OFFSETY:.*]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//      CHECK:     %[[SV1:.*]] = subview %[[ARG0]][%[[OFFSETY]], %[[ARG3]]]
//      CHECK:     %[[OFFSETX:.*]] = affine.apply #[[MAP0]]()[%[[BIDX]]]
//      CHECK:     %[[SV2:.*]] = subview %[[ARG1]][%[[ARG3]], %[[OFFSETX]]]
//      CHECK:     %[[OFFSETY_2:.*]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//      CHECK:     %[[OFFSETX_2:.*]] = affine.apply #[[MAP0]]()[%[[BIDX]]]
//      CHECK:     %[[SV3:.*]] = subview %[[ARG2]][%[[OFFSETY_2]], %[[OFFSETX_2]]]
//      CHECK:     linalg.matmul ins(%[[SV1]], %[[SV2]]{{.*}} outs(%[[SV3]]

// -----

func @gemm3(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>)
{
  linalg.matmul {__internal_linalg_transform__ = "distribute3"}
    ins(%a, %b: memref<?x?xf32>, memref<?x?xf32>)
   outs(%c: memref<?x?xf32>)
  return
}
//  CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0] -> (s0 * 8)>
//      CHECK: func @gemm3(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<?x?xf32>
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<?x?xf32>
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9_]*]]: memref<?x?xf32>
//      CHECK: %[[BIDY:.*]] = "gpu.block_id"() {dimension = "y"}
//      CHECK: %[[NBLOCKSY:.*]] = "gpu.grid_dim"() {dimension = "y"}
//      CHECK: %[[BIDX:.*]] = "gpu.block_id"() {dimension = "x"}
//      CHECK: %[[NBLOCKSX:.*]] = "gpu.grid_dim"() {dimension = "x"}
//      CHECK: %[[LBY:.*]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//      CHECK: %[[STEPY:.*]] = affine.apply #[[MAP0]]()[%[[NBLOCKSY]]]
//      CHECK: %[[LBX:.*]] = affine.apply #[[MAP0]]()[%[[BIDX]]]
//      CHECK: %[[STEPX:.*]] = affine.apply #[[MAP0]]()[%[[NBLOCKSX]]]
//      CHECK: scf.parallel (%[[ARG3:.*]], %[[ARG4:.*]]) = (%[[LBY]], %[[LBX]]) to (%{{.*}}, %{{.*}}) step (%[[STEPY]], %[[STEPX]])
//      CHECK:   scf.for %[[ARG5:.*]] =
//      CHECK:     %[[SV1:.*]] = subview %[[ARG0]][%[[ARG3]], %[[ARG5]]]
//      CHECK:     %[[SV2:.*]] = subview %[[ARG1]][%[[ARG5]], %[[ARG4]]]
//      CHECK:     %[[SV3:.*]] = subview %[[ARG2]][%[[ARG3]], %[[ARG4]]]
//      CHECK:     linalg.matmul ins(%[[SV1]], %[[SV2]]{{.*}} outs(%[[SV3]]

// -----

func @gemm4(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>)
{
  linalg.matmul {__internal_linalg_transform__ = "distribute4"}
    ins(%a, %b: memref<?x?xf32>, memref<?x?xf32>)
   outs(%c: memref<?x?xf32>)
  return
}
//  CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0] -> (s0 * 8)>
//      CHECK: func @gemm4(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<?x?xf32>
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<?x?xf32>
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9_]*]]: memref<?x?xf32>
//      CHECK: %[[BIDY:.*]] = "gpu.block_id"() {dimension = "y"}
//      CHECK: %[[BIDX:.*]] = "gpu.block_id"() {dimension = "x"}
//      CHECK: %[[LBX:.*]] = affine.apply #[[MAP0]]()[%[[BIDX]]]
//      CHECK: %[[INBOUNDS:.*]] = cmpi "slt", %[[LBX]], %{{.*}}
//      CHECK: scf.if %[[INBOUNDS]]
//      CHECK:   scf.for %[[ARG3:.*]] =
//      CHECK:     %[[OFFSETY:.*]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//      CHECK:     %[[SV1:.*]] = subview %[[ARG0]][%[[OFFSETY]], %[[ARG3]]]
//      CHECK:     %[[OFFSETX:.*]] = affine.apply #[[MAP0]]()[%[[BIDX]]]
//      CHECK:     %[[SV2:.*]] = subview %[[ARG1]][%[[ARG3]], %[[OFFSETX]]]
//      CHECK:     %[[OFFSETY_2:.*]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//      CHECK:     %[[OFFSETX_2:.*]] = affine.apply #[[MAP0]]()[%[[BIDX]]]
//      CHECK:     %[[SV3:.*]] = subview %[[ARG2]][%[[OFFSETY_2]], %[[OFFSETX_2]]]
//      CHECK:     linalg.matmul ins(%[[SV1]], %[[SV2]]{{.*}} outs(%[[SV3]]

// -----

func @gemm5(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>)
{
  linalg.matmul {__internal_linalg_transform__ = "distribute5"}
    ins(%a, %b: memref<?x?xf32>, memref<?x?xf32>)
   outs(%c: memref<?x?xf32>)
  return
}
//  CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0] -> (s0 * 8)>
//      CHECK: func @gemm5(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<?x?xf32>
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<?x?xf32>
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9_]*]]: memref<?x?xf32>
//      CHECK: %[[BIDY:.*]] = "gpu.block_id"() {dimension = "y"}
//      CHECK: %[[BIDX:.*]] = "gpu.block_id"() {dimension = "x"}
//      CHECK: %[[NBLOCKSX:.*]] = "gpu.grid_dim"() {dimension = "x"}
//      CHECK: %[[LBY:.*]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//      CHECK: %[[LBX:.*]] = affine.apply #[[MAP0]]()[%[[BIDX]]]
//      CHECK: %[[STEPX:.*]] = affine.apply #[[MAP0]]()[%[[NBLOCKSX]]]
//      CHECK: %[[INBOUNDS:.*]] = cmpi "slt", %[[LBY]], %{{.*}}
//      CHECK: scf.if %[[INBOUNDS]]
//      CHECK:   scf.parallel (%[[ARG3.*]]) = (%[[LBX]]) to (%{{.*}}) step (%[[STEPX]])
//      CHECK:     scf.for %[[ARG4:.*]] =
//      CHECK:      %[[OFFSETY:.*]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//      CHECK:       %[[SV1:.*]] = subview %[[ARG0]][%[[OFFSETY]], %[[ARG4]]]
//      CHECK:       %[[SV2:.*]] = subview %[[ARG1]][%[[ARG4]], %[[ARG3]]]
//      CHECK:       %[[OFFSETY_2:.*]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//      CHECK:       %[[SV3:.*]] = subview %[[ARG2]][%[[OFFSETY_2]], %[[ARG3]]]
//      CHECK:       linalg.matmul ins(%[[SV1]], %[[SV2]]{{.*}} outs(%[[SV3]]

// -----

func @gemm6(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>)
{
  linalg.matmul {__internal_linalg_transform__ = "distribute6"}
    ins(%a, %b: memref<?x?xf32>, memref<?x?xf32>)
   outs(%c: memref<?x?xf32>)
  return
}
//  CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0] -> (s0 * 8)>
//      CHECK: func @gemm6(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<?x?xf32>
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<?x?xf32>
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9_]*]]: memref<?x?xf32>
//      CHECK: %[[BIDY:.*]] = "gpu.block_id"() {dimension = "y"}
//      CHECK: %[[NBLOCKSY:.*]] = "gpu.grid_dim"() {dimension = "y"}
//      CHECK: %[[BIDX:.*]] = "gpu.block_id"() {dimension = "x"}
//      CHECK: %[[LBY:.*]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//      CHECK: %[[STEPY:.*]] = affine.apply #[[MAP0]]()[%[[NBLOCKSY]]]
//      CHECK: scf.parallel (%[[ARG3.*]]) = (%[[LBY]]) to (%{{.*}}) step (%[[STEPY]])
//      CHECK:   scf.for %[[ARG4:.*]] =
//      CHECK:     %[[SV1:.*]] = subview %[[ARG0]][%[[ARG3]], %[[ARG4]]]
//      CHECK:     %[[OFFSETX:.*]] = affine.apply #[[MAP0]]()[%[[BIDX]]]
//      CHECK:     %[[SV2:.*]] = subview %[[ARG1]][%[[ARG4]], %[[OFFSETX]]]
//      CHECK:     %[[OFFSETX_2:.*]] = affine.apply #[[MAP0]]()[%[[BIDX]]]
//      CHECK:     %[[SV3:.*]] = subview %[[ARG2]][%[[ARG3]], %[[OFFSETX_2]]]
//      CHECK:     linalg.matmul ins(%[[SV1]], %[[SV2]]{{.*}} outs(%[[SV3]]

// -----

// CHECK-LABEL: func @matmul_tensors(
// CHECK-SAME:    %[[TA:[0-9a-z]+]]: tensor<?x?xf32>
// CHECK-SAME:    %[[TB:[0-9a-z]+]]: tensor<?x?xf32>
// CHECK-SAME:    %[[TC:[0-9a-z]+]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
func @matmul_tensors(
  %arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
//      CHECK: %[[C8:.*]] = constant 8 : index
//      CHECK: %[[BIDY:.*]] = "gpu.block_id"() {dimension = "y"}
//      CHECK: %[[NBLOCKSY:.*]] = "gpu.grid_dim"() {dimension = "y"}
//      CHECK: %[[BIDX:.*]] = "gpu.block_id"() {dimension = "x"}
//      CHECK: %[[NBLOCKSX:.*]] = "gpu.grid_dim"() {dimension = "x"}
//      CHECK: %[[LBY:.*]] = muli %[[BIDY]], %[[C8]] : index
//      CHECK: %[[STEPY:.*]] = muli %[[NBLOCKSY]], %[[C8]] : index
//      CHECK: %[[TD0:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC0:.*]] = %[[TC]]) -> (tensor<?x?xf32>) {
//      CHECK: %[[LBX:.*]] = muli %[[BIDX]], %[[C8]] : index
//      CHECK: %[[STEPX:.*]] = muli %[[NBLOCKSX]], %[[C8]] : index
//      CHECK:   %[[TD1:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC1:.*]] = %[[TC0]]) -> (tensor<?x?xf32>) {
//      CHECK:     %[[TD2:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC2:.*]] = %[[TC1]]) -> (tensor<?x?xf32>) {
//      CHECK:       %[[sTA:.*]] = subtensor %[[TA]][{{.*}}] : tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:       %[[sTB:.*]] = subtensor %[[TB]][{{.*}}] : tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:       %[[sTC:.*]] = subtensor %[[TC2]][{{.*}}] : tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:       %[[sTD:.*]] = linalg.matmul ins(%[[sTA]], %[[sTB]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME:                                  init(%[[sTC]] : tensor<?x?xf32>)  -> tensor<?x?xf32>
//      CHECK:       %[[TD:.*]] = subtensor_insert %[[sTD]] into %[[TC2]][{{.*}}]  : tensor<?x?xf32> into tensor<?x?xf32>
//      CHECK:       scf.yield %[[TD]] : tensor<?x?xf32>
//      CHECK:     scf.yield %[[TD2]] : tensor<?x?xf32>
//      CHECK:   scf.yield %[[TD1]] : tensor<?x?xf32>
  %0 = linalg.matmul {__internal_linalg_transform__ = "tensors_distribute1"}
       ins(%arg0, %arg1: tensor<?x?xf32>, tensor<?x?xf32>)
      init(%arg2: tensor<?x?xf32>)
    -> tensor<?x?xf32>

//      CHECK: return %[[TD0]] : tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

