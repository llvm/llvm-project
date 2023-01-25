// RUN: mlir-opt %s -test-transform-dialect-interpreter -canonicalize | FileCheck %s

//  CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 2)>
//  CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 3)>
//  CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0)[s0] -> (d0 + s0 - 1)>

func.func @conv(%arg0 : memref<?x?xf32>, %arg1 : memref<?x?xf32>, %arg2 : memref<?x?xf32>) {
  linalg.conv_2d ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>) outs(%arg2 : memref<?x?xf32>)
  return
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.conv_2d"]} in %arg1
    %1, %loop:2 = transform.structured.tile %0 [2, 3] : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
}

//       CHECK: func @conv
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]*]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]*]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]*]]: memref<?x?xf32>
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[KH:.*]] = memref.dim %[[ARG1]], %[[C0]]
//   CHECK-DAG:   %[[KW:.*]] = memref.dim %[[ARG1]], %[[C1]]
//   CHECK-DAG:   %[[H:.*]] = memref.dim %[[ARG2]], %[[C0]]
//   CHECK-DAG:   %[[W:.*]] = memref.dim %[[ARG2]], %[[C1]]
//       CHECK:   scf.for %[[I:.*]] = %[[C0]] to %[[H]] step %[[C2]]
//       CHECK:     %[[T4:.*]] = affine.min #[[MAP0]](%[[I]])[%[[H]]]
//       CHECK:     scf.for %[[J:.*]] = %[[C0]] to %[[W]] step %[[C3]]
//   CHECK-DAG:       %[[T5:.*]] = affine.min #[[MAP1]](%[[J]])[%[[W]]]
//   CHECK-DAG:       %[[T6:.*]] = affine.apply #[[MAP2]](%[[T4]])[%[[KH]]]
//   CHECK-DAG:       %[[T7:.*]] = affine.apply #[[MAP2]](%[[T5]])[%[[KW]]]
//   CHECK-DAG:       %[[SVIN:.*]] = memref.subview %[[ARG0]][%[[I]], %[[J]]] [%[[T6]], %[[T7]]]
//   CHECK-DAG:       %[[SVKER:.*]] = memref.subview %[[ARG1]][0, 0] [%[[KH]], %[[KW]]]
//   CHECK-DAG:       %[[SVOUT:.*]] = memref.subview %[[ARG2]][%[[I]], %[[J]]] [%[[T4]], %[[T5]]]
//       CHECK:       linalg.conv_2d
//  CHECK-SAME:         ins(%[[SVIN]], %[[SVKER]]
//  CHECK-SAME:         outs(%[[SVOUT]]
