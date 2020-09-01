// RUN: mlir-opt %s -test-linalg-transform-patterns=test-tile-and-distribute-options -split-input-file | FileCheck %s

func @gemm1(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>)
{
  linalg.matmul %a, %b, %c {__internal_linalg_transform__ = "distribute1"}
    : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>)
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
//      CHECK:   linalg.matmul %[[SV1]], %[[SV2]], %[[SV3]]

// -----

func @gemm2(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>)
{
  linalg.matmul %a, %b, %c {__internal_linalg_transform__ = "distribute2"}
    : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>)
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
//      CHECK:     linalg.matmul %[[SV1]], %[[SV2]], %[[SV3]]

// -----

func @gemm3(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>)
{
  linalg.matmul %a, %b, %c {__internal_linalg_transform__ = "distribute3"}
    : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>)
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
//      CHECK:     linalg.matmul %[[SV1]], %[[SV2]], %[[SV3]]

// -----

func @gemm4(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>)
{
  linalg.matmul %a, %b, %c {__internal_linalg_transform__ = "distribute4"}
    : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>)
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
//      CHECK:     linalg.matmul %[[SV1]], %[[SV2]], %[[SV3]]

// -----

func @gemm5(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>)
{
  linalg.matmul %a, %b, %c {__internal_linalg_transform__ = "distribute5"}
    : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>)
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
//      CHECK:       linalg.matmul %[[SV1]], %[[SV2]], %[[SV3]]

// -----

func @gemm6(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>)
{
  linalg.matmul %a, %b, %c {__internal_linalg_transform__ = "distribute6"}
    : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>)
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
//      CHECK:     linalg.matmul %[[SV1]], %[[SV2]], %[[SV3]]
