// RUN: mlir-opt %s -linalg-tile="tile-sizes=2,3" | FileCheck %s

//  CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0)[s0, s1] -> (-d0 + s0 + s1 - 1, s1 + 1)>
//  CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0)[s0, s1] -> (-d0 + s0 + s1 - 1, s1 + 2)>
//  CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 2)>
//  CHECK-DAG: #[[MAP3:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 3)>

func.func @conv(%arg0 : memref<?x?xf32>, %arg1 : memref<?x?xf32>, %arg2 : memref<?x?xf32>) {
  linalg.conv_2d ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>) outs(%arg2 : memref<?x?xf32>)
  return
}

//       CHECK: func @conv
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]*]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]*]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]*]]: memref<?x?xf32>
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[T0:.*]] = memref.dim %[[ARG1]], %[[C0]]
//   CHECK-DAG:   %[[T1:.*]] = memref.dim %[[ARG1]], %[[C1]]
//   CHECK-DAG:   %[[T2:.*]] = memref.dim %[[ARG2]], %[[C0]]
//   CHECK-DAG:   %[[T3:.*]] = memref.dim %[[ARG2]], %[[C1]]
//       CHECK:   scf.for %[[ARG3:.*]] = %[[C0]] to %[[T2]] step %[[C2]]
//       CHECK:     scf.for %[[ARG4:.*]] = %[[C0]] to %[[T3]] step %[[C3]]
//       CHECK:       %[[T4:.*]] = affine.min #[[MAP0]](%[[ARG3]])[%[[T2]], %[[T0]]]
//       CHECK:       %[[T5:.*]] = affine.min #[[MAP1]](%[[ARG4]])[%[[T3]], %[[T1]]]
//       CHECK:       %[[SV1:.*]] = memref.subview %[[ARG0]][%[[ARG3]], %[[ARG4]]] [%[[T4]], %[[T5]]]
//       CHECK:       %[[T6:.*]] = affine.min #[[MAP2]](%[[ARG3]])[%[[T2]]
//       CHECK:       %[[T7:.*]] = affine.min #[[MAP3]](%[[ARG4]])[%[[T3]]]
//       CHECK:       %[[SV2:.*]] = memref.subview %[[ARG2]][%[[ARG3]], %[[ARG4]]] [%[[T6]], %[[T7]]]
//       CHECK:       linalg.conv_2d
//  CHECK-SAME:         ins(%[[SV1]], %[[ARG1]]
//  CHECK-SAME:         outs(%[[SV2]]
