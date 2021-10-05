// RUN: mlir-opt %s -linalg-tile="tile-sizes=2,3,0,0,4" | FileCheck %s -check-prefix=TILE-23004

// TILE-23004-DAG: #[[$D0x30pS0x10:.*]] = affine_map<(d0) -> (d0 * 30)>
// TILE-23004-DAG: #[[$S0x10p90D0x30pS1:.*]] = affine_map<(d0)[s0, s1] -> (s0 * 10 + 51, d0 * -30 + s0 * 10 + s1 * 30 - 39)>
// TILE-23004-DAG: #[[$strided4D:.*]] = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3)>
// TILE-23004-DAG: #[[$bound_map_2:.*]] = affine_map<(d0)[s0] -> (2, -d0 + s0)>
// TILE-23004-DAG: #[[$bound_map_3:.*]] = affine_map<(d0)[s0] -> (3, -d0 + s0)>
// TILE-23004-DAG: #[[$bound_map_4:.*]] = affine_map<(d0)[s0] -> (4, -d0 + s0)>

func @conv(%arg0: memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, %arg1: memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, %arg2: memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>) {
  linalg.conv(%arg0, %arg1, %arg2) {dilations = [10, 20], strides = [30, 40]} : memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>
  return
}
//       TILE-23004: func @conv(
//  TILE-23004-SAME:   %[[ARG0:[a-zA-Z0-9_]*]]: memref<?x?x?x?xf32, #[[$strided4D]]>
//  TILE-23004-SAME:   %[[ARG1:[a-zA-Z0-9_]*]]: memref<?x?x?x?xf32, #[[$strided4D]]>
//  TILE-23004-SAME:   %[[ARG2:[a-zA-Z0-9_]*]]: memref<?x?x?x?xf32, #[[$strided4D]]>)
//   TILE-23004-DAG:   %[[C0:.*]] = constant 0 : index
//   TILE-23004-DAG:   %[[C2:.*]] = constant 2 : index
//   TILE-23004-DAG:   %[[C3:.*]] = constant 3 : index
//   TILE-23004-DAG:   %[[C4:.*]] = constant 4 : index
//       TILE-23004:   %[[Z0:.*]] = memref.dim %[[ARG0]], %c0 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       TILE-23004:   %[[Q:.*]] = memref.dim %[[ARG0]], %c2 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       TILE-23004:   %[[B:.*]] = memref.dim %[[ARG1]], %c0 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       TILE-23004:   %[[X0:.*]] = memref.dim %[[ARG2]], %c1 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       TILE-23004:   scf.for %[[ivI:.*]] = %{{.*}} to %[[B]] step %{{.*}} {
//       TILE-23004:     scf.for %[[ivJ:.*]] = %{{.*}} to %[[X0]] step %{{.*}} {
//       TILE-23004:       scf.for %[[ivK:.*]] = %{{.*}} to %[[Q]] step %{{.*}} {
//       TILE-23004:         %[[Z0_1:.*]] = memref.dim %[[ARG0]], %c0 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       TILE-23004:         %[[Z1:.*]] = memref.dim %[[ARG0]], %c1 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       TILE-23004:         %[[szK:.*]] = affine.min #[[$bound_map_4]](%[[ivK]])[%[[Q]]]
//       TILE-23004:         %[[K:.*]] = memref.dim %[[ARG0]], %c3 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       TILE-23004:         %[[FilterView:.*]] = memref.subview %{{.*}}[0, 0, %[[ivK]], 0] [%[[Z0_1]], %[[Z1]], %[[szK]], %[[K]]] [1, 1, 1, 1] : memref<?x?x?x?xf32, #[[$strided4D]]> to memref<?x?x?x?xf32, #[[$strided4D]]>
//
//       TILE-23004:         %[[J1:.*]] = affine.apply #[[$D0x30pS0x10]](%[[ivJ]])
//       TILE-23004:         %[[I1pStep:.*]] = affine.min #[[$S0x10p90D0x30pS1]](%[[ivJ]])[%[[Z0]], %[[X0]]]
//       TILE-23004:         %[[SZ2:.*]] = memref.dim %[[ARG1]], %c2 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       TILE-23004:         %[[sz3:.*]] = affine.min #[[$bound_map_4]](%[[ivK]])[%[[Q]]]
//       TILE-23004:         %[[InputView:.*]] = memref.subview %{{.*}}[%[[ivI]], %[[J1]], 0, %[[ivK]]] [%{{.*}}, %{{.*}}, %[[SZ2]], %[[sz3]]] [1, 1, 1, 1] : memref<?x?x?x?xf32, #[[$strided4D]]> to memref<?x?x?x?xf32, #[[$strided4D]]>
//
//       TILE-23004:         %[[X0:.*]] = memref.dim %[[ARG2]], %c2 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       TILE-23004:         %[[X1:.*]] = memref.dim %[[ARG2]], %c3 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       TILE-23004:         %[[OutputView:.*]] = memref.subview %{{.*}}[%[[ivI]], %[[ivJ]], 0, 0] [%{{.*}}, %{{.*}}, %[[X0]], %[[X1]]] [1, 1, 1, 1] : memref<?x?x?x?xf32, #[[$strided4D]]> to memref<?x?x?x?xf32, #[[$strided4D]]>
//
//       TILE-23004:         linalg.conv(%[[FilterView]], %[[InputView]], %[[OutputView]]) {dilations = [10, 20], strides = [30, 40]} : memref<?x?x?x?xf32, #[[$strided4D]]>, memref<?x?x?x?xf32, #[[$strided4D]]>, memref<?x?x?x?xf32, #[[$strided4D]]>
