// RUN: mlir-opt -test-tiling-interface=tile-using-scf-forall -split-input-file %s | FileCheck %s

func.func @simple_matmul(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul {__internal_transform__ = "simple_gemm"}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0] -> (10, -d0 + s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0] -> (20, -d0 + s0)>
//      CHECK: func.func @simple_matmul(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[M:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[K:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[N:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//      CHECK:   %[[RESULT:.+]] = scf.forall (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:[a-zA-Z0-9]+]]) =
// CHECK-SAME:       (0, 0) to (%[[M]], %[[N]]) step (10, 20) shared_outs(%[[INIT:.+]] = %[[ARG2]])
//      CHECK:     %[[TS_Y:.+]] = affine.min #[[MAP0]](%[[IV0]])[%[[M]]]
//      CHECK:     %[[TS_X:.+]] = affine.min #[[MAP1]](%[[IV1]])[%[[N]]]
//      CHECK:     %[[LHS_TILE:.+]] = tensor.extract_slice %[[ARG0]]
// CHECK-SAME:         [%[[IV0]], 0] [%[[TS_Y]], %[[K]]] [1, 1]
//      CHECK:     %[[RHS_TILE:.+]] = tensor.extract_slice %[[ARG1]]
// CHECK-SAME:         [0, %[[IV1]]] [%[[K]], %[[TS_X]]] [1, 1]
//      CHECK:     %[[INIT_TILE:.+]] = tensor.extract_slice %[[INIT]]
// CHECK-SAME:         [%[[IV0]], %[[IV1]]] [%[[TS_Y]], %[[TS_X]]] [1, 1]
//      CHECK:     %[[GEMM_TILE:.+]] = linalg.matmul
// CHECK-SAME:         ins(%[[LHS_TILE]], %[[RHS_TILE]] :
// CHECK-SAME:         outs(%[[INIT_TILE]] :
//      CHECK:     scf.forall.in_parallel {
//      CHECK:       tensor.parallel_insert_slice %[[GEMM_TILE]] into %[[INIT]]
// CHECK-SAME:           [%[[IV0]], %[[IV1]]] [%[[TS_Y]], %[[TS_X]]] [1, 1]
//      CHECK:   return %[[RESULT]]
