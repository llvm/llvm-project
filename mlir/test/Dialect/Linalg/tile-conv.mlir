// RUN: mlir-opt %s -transform-interpreter -canonicalize -split-input-file | FileCheck %s

//  CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 2)>
//  CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 3)>
//  CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0)[s0] -> (d0 + s0 - 1)>

func.func @conv(%arg0 : memref<?x?xf32>, %arg1 : memref<?x?xf32>, %arg2 : memref<?x?xf32>) {
  linalg.conv_2d ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>) outs(%arg2 : memref<?x?xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.conv_2d"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop:2 = transform.structured.tile_using_for %0 [2, 3] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
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
//       CHECK:     scf.for %[[J:.*]] = %[[C0]] to %[[W]] step %[[C3]]
//   CHECK-DAG:     %[[T4:.*]] = affine.min #[[MAP0]](%[[I]])[%[[H]]]
//   CHECK-DAG:       %[[T5:.*]] = affine.min #[[MAP1]](%[[J]])[%[[W]]]
//   CHECK-DAG:       %[[T6:.*]] = affine.apply #[[MAP2]](%[[T4]])[%[[KH]]]
//   CHECK-DAG:       %[[T7:.*]] = affine.apply #[[MAP2]](%[[T5]])[%[[KW]]]
//   CHECK-DAG:       %[[SVIN:.*]] = memref.subview %[[ARG0]][%[[I]], %[[J]]] [%[[T6]], %[[T7]]]
//   CHECK-DAG:       %[[SVKER:.*]] = memref.subview %[[ARG1]][0, 0] [%[[KH]], %[[KW]]]
//   CHECK-DAG:       %[[SVOUT:.*]] = memref.subview %[[ARG2]][%[[I]], %[[J]]] [%[[T4]], %[[T5]]]
//       CHECK:       linalg.conv_2d
//  CHECK-SAME:         ins(%[[SVIN]], %[[SVKER]]
//  CHECK-SAME:         outs(%[[SVOUT]]

// -----

//  CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 2)>
//  CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 3)>
//  CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 4)>
//  CHECK-DAG: #[[MAP3:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 5)>
//  CHECK-DAG: #[[MAP4:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 6)>
//  CHECK-DAG: #[[MAP5:.*]] = affine_map<(d0)[s0] -> (d0 + s0 - 1)>

func.func @grouped_conv_2D(%arg0 : memref<?x?x?x?x?xf32>, %arg1 : memref<?x?x?x?x?xf32>, %arg2 : memref<?x?x?x?x?xf32>) {
  linalg.grouped_conv_nd {layouts = ["ngcs", "gfcs", "ngfs"]} ins(%arg0, %arg1 : memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>) outs(%arg2 : memref<?x?x?x?x?xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.grouped_conv_nd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop:5 = transform.structured.tile_using_for %0 [2, 3, 4, 5, 6] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

//       CHECK: func @grouped_conv_2D
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]*]]: memref<?x?x?x?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]*]]: memref<?x?x?x?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]*]]: memref<?x?x?x?x?xf32>
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
//   CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
//   CHECK-DAG:   %[[C6:.*]] = arith.constant 6 : index
//   CHECK-DAG:   %[[BATCH:.*]] = memref.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[GROUPS:.*]] = memref.dim %[[ARG0]], %[[C1]]
//   CHECK-DAG:   %[[IN_CHANNELS:.*]] = memref.dim %[[ARG0]], %[[C2]]
//   CHECK-DAG:   %[[OUT_CHANNELS:.*]] = memref.dim %[[ARG1]], %[[C1]]
//   CHECK-DAG:   %[[KW:.*]] = memref.dim %[[ARG1]], %[[C3]]
//   CHECK-DAG:   %[[KH:.*]] = memref.dim %[[ARG1]], %[[C4]]
//   CHECK-DAG:   %[[W:.*]] = memref.dim %[[ARG2]], %[[C3]]
//   CHECK-DAG:   %[[H:.*]] = memref.dim %[[ARG2]], %[[C4]]
//       CHECK:   scf.for %[[I:.*]] = %[[C0]] to %[[BATCH]] step %[[C2]]
//       CHECK:     %[[T4:.*]] = affine.min #[[MAP0]](%[[I]])[%[[BATCH]]]
//       CHECK:     scf.for %[[J:.*]] = %[[C0]] to %[[GROUPS]] step %[[C3]]
//       CHECK:       %[[T5:.*]] = affine.min #[[MAP1]](%[[J]])[%[[GROUPS]]]
//       CHECK:     scf.for %[[K:.*]] = %[[C0]] to %[[OUT_CHANNELS]] step %[[C4]]
//   CHECK-DAG:       %[[T6:.*]] = affine.min #[[MAP2]](%[[K]])[%[[OUT_CHANNELS]]]
//       CHECK:       scf.for %[[L:.*]] = %[[C0]] to %[[W]] step %[[C5]]
//   CHECK-DAG:         %[[T7:.*]] = affine.min #[[MAP3]](%[[L]])[%[[W]]]
//       CHECK:         scf.for %[[M:.*]] = %[[C0]] to %[[H]] step %[[C6]]
//   CHECK-DAG:           %[[T8:.*]] = affine.min #[[MAP4]](%[[M]])[%[[H]]]
//   CHECK-DAG:           %[[T9:.*]] = affine.apply #[[MAP5]](%[[T7]])[%[[KW]]]
//   CHECK-DAG:           %[[T10:.*]] = affine.apply #[[MAP5]](%[[T8]])[%[[KH]]]
//   CHECK-DAG:           %[[SVIN:.*]] = memref.subview %[[ARG0]][%[[I]], %[[J]], 0, %[[L]], %[[M]]] [%[[T4]], %[[T5]], %[[IN_CHANNELS]], %[[T9]], %[[T10]]]
//   CHECK-DAG:           %[[SVKER:.*]] = memref.subview %[[ARG1]][%[[J]], %[[K]], 0, 0, 0] [%[[T5]], %[[T6]], %[[IN_CHANNELS]], %[[KW]], %[[KH]]]
//   CHECK-DAG:           %[[SVOUT:.*]] = memref.subview %[[ARG2]][%[[I]], %[[J]], %[[K]], %[[L]], %[[M]]] [%[[T4]], %[[T5]], %[[T6]], %[[T7]], %[[T8]]]
//       CHECK:           linalg.grouped_conv_nd {layouts = ["ngcs", "gfcs", "ngfs"]}
//  CHECK-SAME:             ins(%[[SVIN]], %[[SVKER]]
//  CHECK-SAME:             outs(%[[SVOUT]]