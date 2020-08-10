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
//      CHECK: %[[T1:.*]] = "gpu.block_id"() {dimension = "y"}
//      CHECK: %[[T2:.*]] = "gpu.block_id"() {dimension = "x"}
//      CHECK: scf.for %[[ARG3:.*]] =
//      CHECK:   %[[T3:.*]] = affine.apply #[[MAP0]]()[%[[T1]]]
//      CHECK:   %[[SV1:.*]] = subview %[[ARG0]][%[[T3]], %[[ARG3]]]
//      CHECK:   %[[T11:.*]] = affine.apply #[[MAP0]]()[%[[T2]]]
//      CHECK:   %[[SV2:.*]] = subview %[[ARG1]][%[[ARG3]], %[[T11]]]
//      CHECK:   %[[T15:.*]] = affine.apply #[[MAP0]]()[%[[T1]]]
//      CHECK:   %[[T18:.*]] = affine.apply #[[MAP0]]()[%[[T2]]]
//      CHECK:   %[[SV3:.*]] = subview %[[ARG2]][%[[T15]], %[[T18]]]
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
//      CHECK: %[[T3:.*]] = "gpu.block_id"() {dimension = "y"}
//      CHECK: %[[T4:.*]] = affine.apply #[[MAP0]]()[%[[T3]]]
//      CHECK: %[[T5:.*]] = "gpu.block_id"() {dimension = "x"}
//      CHECK: %[[T6:.*]] = affine.apply #[[MAP0]]()[%[[T5]]]
//      CHECK: %[[T7:.*]] = cmpi "slt", %[[T4]], %{{.*}}
//      CHECK: %[[T8:.*]] = cmpi "slt", %[[T6]], %{{.*}}
//      CHECK: %[[T9:.*]] = and %[[T7]], %[[T8]]
//      CHECK: scf.if %[[T9]]
//      CHECK:   scf.for %[[ARG3:.*]] =
//      CHECK:     %[[T10:.*]] = affine.apply #[[MAP0]]()[%[[T3]]]
//      CHECK:     %[[SV1:.*]] = subview %[[ARG0]][%[[T10]], %[[ARG3]]]
//      CHECK:     %[[T18:.*]] = affine.apply #[[MAP0]]()[%[[T5]]]
//      CHECK:     %[[SV2:.*]] = subview %[[ARG1]][%[[ARG3]], %[[T18]]]
//      CHECK:     %[[T22:.*]] = affine.apply #[[MAP0]]()[%[[T3]]]
//      CHECK:     %[[T25:.*]] = affine.apply #[[MAP0]]()[%[[T5]]]
//      CHECK:     %[[SV3:.*]] = subview %[[ARG2]][%[[T22]], %[[T25]]]
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
//      CHECK: %[[T3:.*]] = "gpu.block_id"() {dimension = "y"}
//      CHECK: %[[T4:.*]] = "gpu.grid_dim"() {dimension = "y"}
//      CHECK: %[[T5:.*]] = affine.apply #[[MAP0]]()[%[[T3]]]
//      CHECK: %[[T6:.*]] = affine.apply #[[MAP0]]()[%[[T4]]]
//      CHECK: %[[T7:.*]] = "gpu.block_id"() {dimension = "x"}
//      CHECK: %[[T8:.*]] = "gpu.grid_dim"() {dimension = "x"}
//      CHECK: %[[T9:.*]] = affine.apply #[[MAP0]]()[%[[T7]]]
//      CHECK: %[[T10:.*]] = affine.apply #[[MAP0]]()[%[[T8]]]
//      CHECK: scf.parallel (%[[ARG3:.*]], %[[ARG4:.*]]) = (%[[T5]], %[[T9]]) to (%{{.*}}, %{{.*}}) step (%[[T6]], %[[T10]])
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
//      CHECK: %[[T2:.*]] = "gpu.block_id"() {dimension = "y"}
//      CHECK: %[[T3:.*]] = "gpu.block_id"() {dimension = "x"}
//      CHECK: %[[T4:.*]] = affine.apply #[[MAP0]]()[%[[T3]]]
//      CHECK: %[[T5:.*]] = cmpi "slt", %[[T4]], %{{.*}}
//      CHECK: scf.if %[[T5]]
//      CHECK:   scf.for %[[ARG3:.*]] =
//      CHECK:     %[[T6:.*]] = affine.apply #[[MAP0]]()[%[[T2]]]
//      CHECK:     %[[SV1:.*]] = subview %[[ARG0]][%[[T6]], %[[ARG3]]]
//      CHECK:     %[[T14:.*]] = affine.apply #[[MAP0]]()[%[[T3]]]
//      CHECK:     %[[SV2:.*]] = subview %[[ARG1]][%[[ARG3]], %[[T14]]]
//      CHECK:     %[[T18:.*]] = affine.apply #[[MAP0]]()[%[[T2]]]
//      CHECK:     %[[T21:.*]] = affine.apply #[[MAP0]]()[%[[T3]]]
//      CHECK:     %[[SV3:.*]] = subview %[[ARG2]][%[[T18]], %[[T21]]]
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
//      CHECK: %[[T3:.*]] = "gpu.block_id"() {dimension = "y"}
//      CHECK: %[[T4:.*]] = affine.apply #[[MAP0]]()[%[[T3]]]
//      CHECK: %[[T5:.*]] = "gpu.block_id"() {dimension = "x"}
//      CHECK: %[[T6:.*]] = "gpu.grid_dim"() {dimension = "x"}
//      CHECK: %[[T7:.*]] = affine.apply #[[MAP0]]()[%[[T5]]]
//      CHECK: %[[T8:.*]] = affine.apply #[[MAP0]]()[%[[T6]]]
//      CHECK: %[[T9:.*]] = cmpi "slt", %[[T4]], %{{.*}}
//      CHECK: scf.if %[[T9]]
//      CHECK:   scf.parallel (%[[ARG3.*]]) = (%[[T7]]) to (%{{.*}}) step (%[[T8]])
//      CHECK:     scf.for %[[ARG4:.*]] =
//      CHECK:      %[[T10:.*]] = affine.apply #[[MAP0]]()[%[[T3]]]
//      CHECK:       %[[SV1:.*]] = subview %[[ARG0]][%[[T10]], %[[ARG4]]]
//      CHECK:       %[[SV2:.*]] = subview %[[ARG1]][%[[ARG4]], %[[ARG3]]]
//      CHECK:       %[[T21:.*]] = affine.apply #[[MAP0]]()[%[[T3]]]
//      CHECK:       %[[SV3:.*]] = subview %[[ARG2]][%[[T21]], %[[ARG3]]]
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
//      CHECK: %[[T2:.*]] = "gpu.block_id"() {dimension = "y"}
//      CHECK: %[[T3:.*]] = "gpu.grid_dim"() {dimension = "y"}
//      CHECK: %[[T4:.*]] = affine.apply #[[MAP0]]()[%[[T2]]]
//      CHECK: %[[T5:.*]] = affine.apply #[[MAP0]]()[%[[T3]]]
//      CHECK: %[[T6:.*]] = "gpu.block_id"() {dimension = "x"}
//      CHECK: scf.parallel (%[[ARG3.*]]) = (%[[T4]]) to (%{{.*}}) step (%[[T5]])
//      CHECK:   scf.for %[[ARG4:.*]] =
//      CHECK:     %[[SV1:.*]] = subview %[[ARG0]][%[[ARG3]], %[[ARG4]]]
//      CHECK:     %[[T14:.*]] = affine.apply #[[MAP0]]()[%[[T6]]]
//      CHECK:     %[[SV2:.*]] = subview %[[ARG1]][%[[ARG4]], %[[T14]]]
//      CHECK:     %[[T20:.*]] = affine.apply #[[MAP0]]()[%[[T6]]]
//      CHECK:     %[[SV3:.*]] = subview %[[ARG2]][%[[ARG3]], %[[T20]]]
//      CHECK:     linalg.matmul %[[SV1]], %[[SV2]], %[[SV3]]
