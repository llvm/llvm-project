// RUN: mlir-opt %s -linalg-tile="linalg-tile-sizes=2,3,0,0,4" | FileCheck %s -check-prefix=TILE-23004
// RUN: mlir-opt %s -linalg-tile="linalg-tile-sizes=2" | FileCheck %s -check-prefix=TILE-20000

// TILE-23004-DAG: #[[strided4D:.*]] = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3)>
// TILE-20000-DAG: #[[strided4D:.*]] = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3)>
// TILE-20000-DAG: #[[minmap:.*]] = affine_map<(d0, d1, d2) -> (d0, d1 - d2)>
// TILE-20000-DAG: #[[subviewstride:.*]] = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3 * s4)>

func @conv_padding(%arg0: memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, %arg1: memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, %arg2: memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>) {
  linalg.conv(%arg0, %arg1, %arg2) {dilations = [10, 20], padding = dense<[[1, 1], [0, 1]]> : tensor<2x2xi64>, strides = [30, 40]} : memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>
  return
}
// TILE-23004-LABEL: func @conv_padding(
//  TILE-23004-SAME:   %[[ARG0:[a-zA-Z0-9_]*]]: memref<?x?x?x?xf32, #[[strided4D]]>
//  TILE-23004-SAME:   %[[ARG1:[a-zA-Z0-9_]*]]: memref<?x?x?x?xf32, #[[strided4D]]>
//  TILE-23004-SAME:   %[[ARG2:[a-zA-Z0-9_]*]]: memref<?x?x?x?xf32, #[[strided4D]]>)
//       TILE-23004:         linalg.conv(%[[ARG0]], %[[ARG1]], %[[ARG2]])

// TILE-20000-LABEL: func @conv_padding(
//  TILE-20000-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<?x?x?x?xf32, #[[strided4D]]>
//  TILE-20000-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<?x?x?x?xf32, #[[strided4D]]>
//  TILE-20000-SAME: %[[ARG2:[a-zA-Z0-9_]*]]: memref<?x?x?x?xf32, #[[strided4D]]>)
//   TILE-20000-DAG:   %[[C0:.*]] = constant 0 : index
//   TILE-20000-DAG:   %[[C1:.*]] = constant 1 : index
//   TILE-20000-DAG:   %[[C2:.*]] = constant 2 : index
//       TILE-20000:   %[[B:.*]] = dim %[[ARG1]], 0
//       TILE-20000:   loop.for %[[ivI:.*]] = %[[C0]] to %[[B]] step %[[C2]] {
//       TILE-20000:     %[[DIM10:.*]] = dim %[[ARG1]], 0
//       TILE-20000:     %[[EXTENT:.*]] = affine.min #[[minmap]](%[[C2]], %[[DIM10]], %[[ivI]])
//       TILE-20000:     %[[DIM11:.*]] = dim %[[ARG1]], 1
//       TILE-20000:     %[[DIM12:.*]] = dim %[[ARG1]], 2
//       TILE-20000:     %[[DIM13:.*]] = dim %[[ARG1]], 3
//       TILE-20000:     %[[SUBVIEW1:.*]] = subview %[[ARG1]][%[[ivI]], %[[C0]], %[[C0]], %[[C0]]] [%[[EXTENT]], %[[DIM11]], %[[DIM12]], %[[DIM13]]]
//       TILE-20000:     %[[DIM20:.*]] = dim %[[ARG2]], 0
//       TILE-20000:     %[[EXTENT:.*]] = affine.min #[[minmap]](%[[C2]], %[[DIM20]], %[[ivI]])
//       TILE-20000:     %[[DIM21:.*]] = dim %[[ARG2]], 1
//       TILE-20000:     %[[DIM22:.*]] = dim %[[ARG2]], 2
//       TILE-20000:     %[[DIM23:.*]] = dim %[[ARG2]], 3
//       TILE-20000:     %[[SUBVIEW2:.*]] = subview %[[ARG2]][%[[ivI]], %[[C0]], %[[C0]], %[[C0]]] [%[[EXTENT]], %[[DIM21]], %[[DIM22]], %[[DIM23]]]
//       TILE-20000:     linalg.conv(%[[ARG0]], %[[SUBVIEW1]], %[[SUBVIEW2]])
