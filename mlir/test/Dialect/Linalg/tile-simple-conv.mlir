// RUN: mlir-opt %s -linalg-tile="tile-sizes=2,3,4" | FileCheck %s

//  CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0)[s0] -> (2, -d0 + s0)>
//  CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0)[s0, s1] -> (s0 + 2, -d0 + s0 + s1 - 1)>
//  CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0)[s0, s1] -> (s0 + 3, -d0 + s0 + s1 - 1)>
//  CHECK-DAG: #[[MAP4:.*]] = affine_map<(d0)[s0] -> (3, -d0 + s0)>
//  CHECK-DAG: #[[MAP5:.*]] = affine_map<(d0)[s0] -> (4, -d0 + s0)>

func @conv(%arg0 : memref<?x?x?x?xf32>, %arg1 : memref<?x?x?x?xf32>, %arg2 : memref<?x?x?x?xf32>) {
  linalg.conv(%arg0, %arg1, %arg2) : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
  return
}

//       CHECK: func @conv
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]*]]: memref<?x?x?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]*]]: memref<?x?x?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]*]]: memref<?x?x?x?xf32>
//   CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = constant 2 : index
//   CHECK-DAG:   %[[C3:.*]] = constant 3 : index
//   CHECK-DAG:   %[[C4:.*]] = constant 4 : index
//       CHECK:   %[[T0:.*]] = memref.dim %[[ARG0]], %[[C0]]
//       CHECK:   %[[T1:.*]] = memref.dim %[[ARG0]], %[[C1]]
//       CHECK:   %[[T2:.*]] = memref.dim %[[ARG1]], %[[C0]]
//       CHECK:   %[[T3:.*]] = memref.dim %[[ARG2]], %[[C1]]
//       CHECK:   %[[T4:.*]] = memref.dim %[[ARG2]], %[[C2]]
//       CHECK:   scf.for %[[ARG3:.*]] = %[[C0]] to %[[T2]] step %[[C2]]
//       CHECK:     scf.for %[[ARG4:.*]] = %[[C0]] to %[[T3]] step %[[C3]]
//       CHECK:       scf.for %[[ARG5:.*]] = %[[C0]] to %[[T4]] step %[[C4]]
//       CHECK:         %[[T6:.*]] = affine.min #[[MAP0]](%[[ARG3]])[%[[T2]]]
//       CHECK:         %[[T8:.*]] = affine.min #[[MAP1]](%[[ARG4]])[%[[T0]], %[[T3]]]
//       CHECK:         %[[T10:.*]] = affine.min #[[MAP2]](%[[ARG5]])[%[[T1]], %[[T4]]]
//       CHECK:         %[[T11:.*]] = memref.dim %[[ARG1]], %[[C3]]
//       CHECK:         %[[SV1:.*]] = memref.subview %[[ARG1]][%[[ARG3]], %[[ARG4]], %[[ARG5]], 0]
//  CHECK-SAME:                                        [%[[T6]], %[[T8]], %[[T10]], %[[T11]]]
//       CHECK:         %[[T14:.*]] = affine.min #[[MAP0]](%[[ARG3]])[%[[T2]]
//       CHECK:         %[[T16:.*]] = affine.min #[[MAP4]](%[[ARG4]])[%[[T3]]]
//       CHECK:         %[[T18:.*]] = affine.min #[[MAP5]](%[[ARG5]])[%[[T4]]
//       CHECK:         %[[T19:.*]] = memref.dim %[[ARG2]], %[[C3]]
//       CHECK:         %[[SV2:.*]] = memref.subview %[[ARG2]][%[[ARG3]], %[[ARG4]], %[[ARG5]], 0]
//  CHECK-SAME:                                        [%[[T14]], %[[T16]], %[[T18]], %[[T19]]]
//       CHECK:         linalg.conv(%[[ARG0]], %[[SV1]], %[[SV2]])
