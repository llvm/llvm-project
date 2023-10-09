// RUN: mlir-opt -test-tiling-interface=lower-to-scalar-using-scf-for -split-input-file %s | FileCheck %s

func.func @gemm(%arg0 : memref<?x?xf32>, %arg1 : memref<?x?xf32>,
  %arg2 : memref<?x?xf32>) {
  linalg.matmul ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>)
      outs(%arg2 : memref<?x?xf32>)
  return
}
// CHECK-LABEL: func @gemm
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: memref<?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: memref<?x?xf32>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: memref<?x?xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[M:.+]] = memref.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[K:.+]] = memref.dim %[[ARG0]], %[[C1]]
//   CHECK-DAG:   %[[N:.+]] = memref.dim %[[ARG1]], %[[C1]]
//       CHECK:   scf.for %[[IV0:[a-zA-Z0-9]+]] = %[[C0]] to %[[M]] step %[[C1]]
//       CHECK:     scf.for %[[IV1:[a-zA-Z0-9]+]] = %[[C0]] to %[[N]] step %[[C1]]
//       CHECK:       scf.for %[[IV2:[a-zA-Z0-9]+]] = %[[C0]] to %[[K]] step %[[C1]]
//   CHECK-DAG:         %[[LHS:.+]] = memref.load %[[ARG0]][%[[IV0]], %[[IV2]]]
//   CHECK-DAG:         %[[RHS:.+]] = memref.load %[[ARG1]][%[[IV2]], %[[IV1]]]
//   CHECK-DAG:         %[[OUT:.+]] = memref.load %[[ARG2]][%[[IV0]], %[[IV1]]]
//       CHECK:         %[[MULF:.+]] = arith.mulf %[[LHS]], %[[RHS]]
//       CHECK:         %[[ADDF:.+]] = arith.addf %[[OUT]], %[[MULF]]
//       CHECK:         memref.store %[[ADDF]], %[[ARG2]][%[[IV0]], %[[IV1]]]

// -----

func.func @indexed_generic(%arg0 : memref<200x300xi32>, %arg1 : memref<300xi16>,
    %arg2 : memref<200xi8>, %arg3 : memref<300x200xi64>) {
  linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1, d0)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1, %arg2 : memref<200x300xi32>, memref<300xi16>, memref<200xi8>)
      outs(%arg3 : memref<300x200xi64>) {
    ^bb0(%b0 : i32, %b1 : i16, %b2 : i8, %b3 : i64):
      %0 = linalg.index 0 : index
      %1 = arith.index_cast %0 : index to i16
      %2 = arith.muli %b1, %1 : i16
      %3 = linalg.index 1 : index
      %4 = arith.index_cast %3 : index to i8
      %5 = arith.muli %b2, %4 : i8
      %6 = arith.extsi %2 : i16 to i32
      %7 = arith.extsi %5 : i8 to i32
      %8 = arith.addi %6, %7 : i32
      %9 = arith.addi %8, %b0 : i32
      %10 = arith.extsi %9 : i32 to i64
      linalg.yield %10 : i64
    }
  return
}
// CHECK-LABEL: func @indexed_generic
//  CHECK-SAME:     %[[ARG0:.+]]: memref<200x300xi32>
//  CHECK-SAME:     %[[ARG1:.+]]: memref<300xi16>
//  CHECK-SAME:     %[[ARG2:.+]]: memref<200xi8>
//  CHECK-SAME:     %[[ARG3:.+]]: memref<300x200xi64>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C200:.+]] = arith.constant 200 : index
//   CHECK-DAG:   %[[C300:.+]] = arith.constant 300 : index
//       CHECK:   scf.for %[[IV0:[a-zA-Z0-9]+]] = %[[C0]] to %[[C200]] step %[[C1]]
//       CHECK:     scf.for %[[IV1:[a-zA-Z0-9]+]] = %[[C0]] to %[[C300]] step %[[C1]]
//   CHECK-DAG:       %[[B0:.+]] = memref.load %[[ARG0]][%[[IV0]], %[[IV1]]]
//   CHECK-DAG:       %[[B1:.+]] = memref.load %[[ARG1]][%[[IV1]]]
//   CHECK-DAG:       %[[B2:.+]] = memref.load %[[ARG2]][%[[IV0]]]
//       CHECK:       %[[T1:.+]] = arith.index_cast %[[IV0]]
//       CHECK:       %[[T2:.+]] = arith.muli %[[B1]], %[[T1]]
//       CHECK:       %[[T4:.+]] = arith.index_cast %[[IV1]]
//       CHECK:       %[[T5:.+]] = arith.muli %[[B2]], %[[T4]]
//       CHECK:       %[[T6:.+]] = arith.extsi %[[T2]]
//       CHECK:       %[[T7:.+]] = arith.extsi %[[T5]]
//       CHECK:       %[[T8:.+]] = arith.addi %[[T6]], %[[T7]]
//       CHECK:       %[[T9:.+]] = arith.addi %[[T8]], %[[B0]]
//       CHECK:       %[[T10:.+]] = arith.extsi %[[T9]]
//       CHECK:       memref.store %[[T10]], %[[ARG3]][%[[IV1]], %[[IV0]]]

// -----

func.func @conv_strides_and_dilation(%arg0 : memref<?x?x?x?xf32>, %arg1 : memref<?x?x?x?xf32>,
  %arg2 : memref<?x?x?x?xf32>) {
  linalg.conv_2d_nhwc_hwcf {
      strides = dense<[1, 2]> : tensor<2xi64>,
      dilations = dense<[3, 4]> : tensor<2xi64>}
      ins(%arg0, %arg1 : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>)
      outs(%arg2 : memref<?x?x?x?xf32>)
  return
}
//  CHECK-DAG:  #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 + d4 * 3)>
//  CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 2 + d5 * 4)>
//       CHECK: func @conv_strides_and_dilation(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: memref<?x?x?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: memref<?x?x?x?xf32>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: memref<?x?x?x?xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//   CHECK-DAG:   %[[N:.+]] = memref.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[C:.+]] = memref.dim %[[ARG0]], %[[C3]]
//   CHECK-DAG:   %[[H:.+]] = memref.dim %[[ARG1]], %[[C0]]
//   CHECK-DAG:   %[[W:.+]] = memref.dim %[[ARG1]], %[[C1]]
//   CHECK-DAG:   %[[F:.+]] = memref.dim %[[ARG1]], %[[C3]]
//   CHECK-DAG:   %[[P:.+]] = memref.dim %[[ARG2]], %[[C1]]
//   CHECK-DAG:   %[[Q:.+]] = memref.dim %[[ARG2]], %[[C2]]
//       CHECK:   scf.for %[[IV0:[a-zA-Z0-9]+]] = %[[C0]] to %[[N]] step %[[C1]]
//       CHECK:     scf.for %[[IV1:[a-zA-Z0-9]+]] = %[[C0]] to %[[P]] step %[[C1]]
//       CHECK:       scf.for %[[IV2:[a-zA-Z0-9]+]] = %[[C0]] to %[[Q]] step %[[C1]]
//       CHECK:         scf.for %[[IV3:[a-zA-Z0-9]+]] = %[[C0]] to %[[F]] step %[[C1]]
//       CHECK:           scf.for %[[IV4:[a-zA-Z0-9]+]] = %[[C0]] to %[[H]] step %[[C1]]
//       CHECK:             scf.for %[[IV5:[a-zA-Z0-9]+]] = %[[C0]] to %[[W]] step %[[C1]]
//       CHECK:               scf.for %[[IV6:[a-zA-Z0-9]+]] = %[[C0]] to %[[C]] step %[[C1]]
//   CHECK-DAG:                 %[[I:.+]] = affine.apply #[[MAP0]](%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]], %[[IV4]], %[[IV5]], %[[IV6]])
//   CHECK-DAG:                 %[[J:.+]] = affine.apply #[[MAP1]](%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]], %[[IV4]], %[[IV5]], %[[IV6]])
//   CHECK-DAG:                 %[[T9:.+]] = memref.load %[[ARG0]][%[[IV0]], %[[I]], %[[J]], %[[IV6]]]
//   CHECK-DAG:                 %[[T10:.+]] = memref.load %[[ARG1]][%[[IV4]], %[[IV5]], %[[IV6]], %[[IV3]]]
//   CHECK-DAG:                 %[[T11:.+]] = memref.load %[[ARG2]][%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]
//       CHECK:                 %[[T12:.+]] = arith.mulf %[[T9]], %[[T10]]
//       CHECK:                 %[[T13:.+]] = arith.addf %[[T11]], %[[T12]]
//       CHECK:                 memref.store %[[T13]], %[[ARG2]][%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]

// -----

func.func @pool_strides_and_dilation(%arg0 : memref<?x?x?x?xf32>, %arg1 : memref<?x?xf32>,
  %arg2 : memref<?x?x?x?xf32>) {
  linalg.pooling_nhwc_max {
      strides = dense<[1, 2]> : tensor<2xi64>,
      dilations = dense<[3, 4]> : tensor<2xi64>}
      ins(%arg0, %arg1 : memref<?x?x?x?xf32>, memref<?x?xf32>)
      outs(%arg2 : memref<?x?x?x?xf32>)
  return
}
//  CHECK-DAG:  #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1 + d4 * 3)>
//  CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2 * 2 + d5 * 4)>
//       CHECK: func @pool_strides_and_dilation
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: memref<?x?x?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: memref<?x?xf32>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: memref<?x?x?x?xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//   CHECK-DAG:   %[[N:.+]] = memref.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[C:.+]] = memref.dim %[[ARG0]], %[[C3]]
//   CHECK-DAG:   %[[H:.+]] = memref.dim %[[ARG1]], %[[C0]]
//   CHECK-DAG:   %[[W:.+]] = memref.dim %[[ARG1]], %[[C1]]
//   CHECK-DAG:   %[[P:.+]] = memref.dim %[[ARG2]], %[[C1]]
//   CHECK-DAG:   %[[Q:.+]] = memref.dim %[[ARG2]], %[[C2]]
//       CHECK:   scf.for %[[IV0:[a-zA-Z0-9]+]] = %[[C0]] to %[[N]] step %[[C1]]
//       CHECK:     scf.for %[[IV1:[a-zA-Z0-9]+]] = %[[C0]] to %[[P]] step %[[C1]]
//       CHECK:       scf.for %[[IV2:[a-zA-Z0-9]+]] = %[[C0]] to %[[Q]] step %[[C1]]
//       CHECK:         scf.for %[[IV3:[a-zA-Z0-9]+]] = %[[C0]] to %[[C]] step %[[C1]]
//       CHECK:           scf.for %[[IV4:[a-zA-Z0-9]+]] = %[[C0]] to %[[H]] step %[[C1]]
//       CHECK:             scf.for %[[IV5:[a-zA-Z0-9]+]] = %[[C0]] to %[[W]] step %[[C1]]
//   CHECK-DAG:               %[[I:.+]] = affine.apply #[[MAP0]](%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]], %[[IV4]], %[[IV5]])
//   CHECK-DAG:               %[[J:.+]] = affine.apply #[[MAP1]](%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]], %[[IV4]], %[[IV5]])
//   CHECK-DAG:               %[[T8:.+]] = memref.load %[[ARG0]][%[[IV0]], %[[I]], %[[J]], %[[IV3]]]
//   CHECK-DAG:               %[[T9:.+]] = memref.load %[[ARG2]][%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]
//       CHECK:               %[[T10:.+]] = arith.maximumf %[[T9]], %[[T8]]
//       CHECK:               memref.store %[[T10]], %[[ARG2]][%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]

// -----

func.func @map(%lhs: memref<64xf32>,
    %rhs: memref<64xf32>, %out: memref<64xf32>) {
  linalg.map ins(%lhs, %rhs : memref<64xf32>, memref<64xf32>)
             outs(%out : memref<64xf32>)
    (%in: f32, %in_0: f32) {
      %0 = arith.addf %in, %in_0 : f32
      linalg.yield %0 : f32
    }
  return
}
// CHECK-LABEL: func.func @map(
// CHECK-SAME:    %[[LHS:[a-zA-Z0-9]+]]: memref<64xf32>,
// CHECK-SAME:    %[[RHS:[a-zA-Z0-9]+]]: memref<64xf32>,
// CHECK-SAME:    %[[OUT:[a-zA-Z0-9]+]]: memref<64xf32>) {

// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index

// CHECK:     scf.for %[[I:.*]] = %[[C0]] to %[[C64]] step %[[C1]] {
// CHECK:       %[[LHS_ELEM:.*]] = memref.load %[[LHS]][%[[I]]]
// CHECK:       %[[RHS_ELEM:.*]] = memref.load %[[RHS]][%[[I]]]
// CHECK:       %[[ADD:.*]] = arith.addf %[[LHS_ELEM]], %[[RHS_ELEM]]
// CHECK:       memref.store %[[ADD]], %[[OUT]][%[[I]]]

// -----

func.func @transpose(%arg0: memref<16x32x64xf32>,
                               %arg1: memref<32x64x16xf32>) {
  linalg.transpose ins(%arg0 : memref<16x32x64xf32>)
                   outs(%arg1 : memref<32x64x16xf32>) permutation = [1, 2, 0]
  return
}
// CHECK-LABEL: func.func @transpose(
// CHECK-SAME:    %[[IN:[a-zA-Z0-9]+]]: memref<16x32x64xf32>,
// CHECK-SAME:    %[[OUT:[a-zA-Z0-9]+]]: memref<32x64x16xf32>)

// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index

// CHECK:     scf.for %[[I:.*]] = %[[C0]] to %[[C16]] step %[[C1]] {
// CHECK:       scf.for %[[J:.*]] = %[[C0]] to %[[C32]] step %[[C1]] {
// CHECK:         scf.for %[[K:.*]] = %[[C0]] to %[[C64]] step %[[C1]] {
// CHECK:           %[[ELEM:.*]] = memref.load %[[IN]][%[[I]], %[[J]], %[[K]]]
// CHECK:           memref.store %[[ELEM]], %[[OUT]][%[[J]], %[[K]], %[[I]]]

// -----

func.func @reduce(%arg0: memref<16x32x64xf32>,
                  %arg1: memref<16x64xf32>) {
  linalg.reduce ins(%arg0 : memref<16x32x64xf32>)
                outs(%arg1 : memref<16x64xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %0 = arith.addf %in, %init : f32
      linalg.yield %0 : f32
    }
  return
}
// CHECK-LABEL: func.func @reduce(
// CHECK-SAME:    %[[IN:[a-zA-Z0-9]+]]: memref<16x32x64xf32>,
// CHECK-SAME:    %[[OUT:[a-zA-Z0-9]+]]: memref<16x64xf32>

// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index

// CHECK:     scf.for %[[I:.*]] = %[[C0]] to %[[C16]] step %[[C1]] {
// CHECK:       scf.for %[[J:.*]] = %[[C0]] to %[[C32]] step %[[C1]] {
// CHECK:         scf.for %[[K:.*]] = %[[C0]] to %[[C64]] step %[[C1]] {
// CHECK:           %[[IN_ELEM:.*]] = memref.load %[[IN]][%[[I]], %[[J]], %[[K]]]
// CHECK:           %[[OUT_ELEM:.*]] = memref.load %[[OUT]][%[[I]], %[[K]]]
// CHECK:           %[[ADD:.*]] = arith.addf %[[IN_ELEM]], %[[OUT_ELEM]]
// CHECK:           memref.store %[[ADD]], %[[OUT]][%[[I]], %[[K]]]

// -----

func.func @broadcast(%input: memref<8x32xf32>,
                     %init: memref<8x16x32xf32>) {
  linalg.broadcast
      ins(%input:memref<8x32xf32>)
      outs(%init:memref<8x16x32xf32>)
      dimensions = [1]
  func.return
}
// CHECK-LABEL: func.func @broadcast(
// CHECK-SAME:    %[[IN:[a-zA-Z0-9]+]]: memref<8x32xf32>,
// CHECK-SAME:    %[[OUT:[a-zA-Z0-9]+]]: memref<8x16x32xf32>

// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index

// CHECK:     scf.for %[[I:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {
// CHECK:       scf.for %[[J:.*]] = %[[C0]] to %[[C16]] step %[[C1]] {
// CHECK:         scf.for %[[K:.*]] = %[[C0]] to %[[C32]] step %[[C1]] {
// CHECK:           %[[ELEM:.*]] = memref.load %[[IN]][%[[I]], %[[K]]]
// CHECK:           memref.store %[[ELEM]], %[[OUT]][%[[I]], %[[J]], %[[K]]]
