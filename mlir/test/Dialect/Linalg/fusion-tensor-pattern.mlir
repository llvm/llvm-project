// RUN: mlir-opt %s -test-linalg-tensor-fusion-transform-patterns -canonicalize -cse -split-input-file -verify-diagnostics | FileCheck %s

module {
  func @matmul_fusion(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
                      %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>,
                      %arg4: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>   // <MxN1> <N1xN2>
    %1 = linalg.matmul {__internal_linalg_transform__ = "lhs_fusion"}
      ins(%0, %arg3 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg4 : tensor<?x?xf32>) -> tensor<?x?xf32>   // <MxN2> <N2xN3>
    return %1 : tensor<?x?xf32>
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0] -> (32, -d0 + s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (32, d0 - d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0)[s0] -> (16, -d0 + s0)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0)[s0] -> (64, -d0 + s0)>
//  CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1) -> (64, d0 - d1)>
//      CHECK: func @matmul_fusion
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//  CHECK-DAG:   %[[C32:.+]] = constant 32 : index
//  CHECK-DAG:   %[[C64:.+]] = constant 64 : index
//  CHECK-DAG:   %[[C16:.+]] = constant 16 : index
//  CHECK-DAG:   %[[M:.+]] = dim %[[ARG0]], %[[C0]]
//      CHECK:   %[[RESULT:.+]] = scf.for %[[IV0:[a-zA-Z0-9]+]] =
// CHECK-SAME:     %[[C0]] to %[[M]] step %[[C32]]
// CHECK-SAME:     iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<?x?xf32>) {
//      CHECK:     %[[TILE_M:.+]] = affine.min #[[MAP0]](%[[IV0]])[%[[M]]]
//      CHECK:     %[[M_2:.+]] = dim %[[ARG6]], %[[C0]]
//      CHECK:     %[[TILE_M_2:.+]] = affine.min #[[MAP1]](%[[M_2]], %[[IV0]])
//      CHECK:     %[[N3:.+]] = dim %[[ARG6]], %[[C1]]
//      CHECK:     %[[ST_ARG6:.+]] = subtensor %[[ARG6]][%[[IV0]], 0]
// CHECK-SAME:       [%[[TILE_M_2]], %[[N3]]]
//      CHECK:     %[[N2:.+]] = dim %[[ARG1]], %[[C1]]
//      CHECK:     %[[N1:.+]] = dim %[[ARG0]], %[[C1]]
//      CHECK:     %[[ST_ARG0:.+]] = subtensor %[[ARG0]][%[[IV0]], 0]
// CHECK-SAME:       [%[[TILE_M]], %[[N1]]]
//      CHECK:     %[[ST_ARG1:.+]] = subtensor %[[ARG1]][0, 0]
// CHECK-SAME:       [%[[N1]], %[[N2]]]
//      CHECK:     %[[ST_ARG2:.+]] = subtensor %[[ARG2]][%[[IV0]], 0]
// CHECK-SAME:       [%[[TILE_M]], %[[N2]]]
//      CHECK:     %[[LHS:.+]] = linalg.matmul
// CHECK-SAME:       __internal_linalg_transform__ = "after_lhs_fusion_producer"
// CHECK-SAME:       ins(%[[ST_ARG0]], %[[ST_ARG1]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME:       outs(%[[ST_ARG2]] : tensor<?x?xf32>)
//      CHECK:     %[[N3_2:.+]] = dim %[[ARG3]], %[[C1]]
//      CHECK:     %[[YIELD0:.+]] = scf.for %[[IV1:[a-zA-Z0-9]+]] =
// CHECK-SAME:       %[[C0]] to %[[N3_2]] step %[[C64]]
// CHECK-SAME:       iter_args(%[[ARG8:.+]] = %[[ST_ARG6]]) -> (tensor<?x?xf32>) {
//      CHECK:       %[[YIELD1:.+]] = scf.for %[[IV2:[a-zA-Z0-9]+]] =
// CHECK-SAME:         %[[C0]] to %[[N2]] step %[[C16]]
// CHECK-SAME:         iter_args(%[[ARG10:.+]] = %[[ARG8]]) -> (tensor<?x?xf32>) {
//      CHECK:         %[[TILE_N2:.+]] = affine.min #[[MAP2]](%[[IV2]])[%[[N2]]]
//      CHECK:         %[[ST_LHS:.+]] = subtensor %[[LHS]][0, %[[IV2]]]
// CHECK-SAME:           [%[[TILE_M]], %[[TILE_N2]]]
//      CHECK:         %[[N2_3:.+]] = dim %[[ARG3]], %[[C0]]
//      CHECK:         %[[TILE_N2_2:.+]] = affine.min #[[MAP2]](%[[IV2]])[%[[N2_3]]]
//      CHECK:         %[[TILE_N3:.+]] = affine.min #[[MAP3]](%[[IV1]])[%[[N3_2]]]
//      CHECK:         %[[ST_ARG3:.+]] = subtensor %[[ARG3]][%[[IV2]], %[[IV1]]]
// CHECK-SAME:           [%[[TILE_N2_2]], %[[TILE_N3]]]
//      CHECK:         %[[M_4:.+]] = dim %[[ARG10]], %[[C0]]
//      CHECK:         %[[N3_3:.+]] = dim %[[ARG10]], %[[C1]]
//      CHECK:         %[[TILE_N3_2:.+]] = affine.min #[[MAP4]](%[[N3_3]], %[[IV1]])
//      CHECK:         %[[ST_ARG4:.+]] = subtensor %[[ARG10]][0, %[[IV1]]]
// CHECK-SAME:           [%[[M_4]], %[[TILE_N3_2]]]
//      CHECK:         %[[ST_RESULT:.+]] = linalg.matmul
// CHECK-SAME:           __internal_linalg_transform__ = "after_lhs_fusion"
// CHECK-SAME:           ins(%[[ST_LHS]], %[[ST_ARG3]]
// CHECK-SAME:             : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME:           outs(%[[ST_ARG4]] : tensor<?x?xf32>)
//      CHECK:         %[[UPDATE1:.+]] = subtensor_insert %[[ST_RESULT]]
// CHECK-SAME:           into %[[ARG10]][0, %[[IV1]]] [%[[M_4]], %[[TILE_N3_2]]]
//      CHECK:         scf.yield %[[UPDATE1]]
//      CHECK:       }
//      CHECK:       scf.yield %[[YIELD1]]
//      CHECK:     }
//      CHECK:     %[[UPDATE0:.+]] = subtensor_insert %[[YIELD0]] into
// CHECK-SAME:       %[[ARG6]][%[[IV0]], 0] [%[[TILE_M_2]], %[[N3]]]
//      CHECK:     scf.yield %[[UPDATE0]]
//      CHECK:   }
//      CHECK:   return %[[RESULT]]

// -----

module {
  func @matmul_plus_matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
                           %arg2: tensor<?x?xf32>) -> tensor<?x?xf32>{
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %0 = dim %arg2, %c0 : tensor<?x?xf32>
    %1 = dim %arg2, %c1 : tensor<?x?xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %3 = dim %2, %c0 : tensor<?x?xf32>
    %4 = dim %2, %c1 : tensor<?x?xf32>
    %5 = linalg.init_tensor [%3, %4] : tensor<?x?xf32>
    %6 = linalg.generic
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"],
       __internal_linalg_transform__ = "transpose_fusion"}
      ins(%2, %2 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%5 : tensor<?x?xf32>) {
      ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32) :
        %7 = addf %arg3, %arg4 : f32
        linalg.yield %7 : f32
      } -> tensor<?x?xf32>
    return %6 : tensor<?x?xf32>
  }
}
//       CHECK: func @matmul_plus_matmul
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//       CHECK:   %[[RESULT:.+]] = scf.for %[[IV0:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     iter_args(%[[ARG4:.+]] = %{{[a-zA-Z0-9_]+}})
//       CHECK:     %[[YIELD:.+]] = scf.for %[[IV1:[a-zA-Z0-9_]+]]
//  CHECK-SAME:       iter_args(%[[ARG6:.+]] = %[[ARG4]])
//       CHECK:       %[[ST_ARG6:.+]] = subtensor %[[ARG6]][%[[IV0]], %[[IV1]]]
//       CHECK:       %[[ST_ARG0:.+]] = subtensor %[[ARG0]][%[[IV0]], 0]
//       CHECK:       %[[ST_ARG1:.+]] = subtensor %[[ARG1]][0, %[[IV1]]]
//       CHECK:       %[[ST_ARG2:.+]] = subtensor %[[ARG2]][%[[IV0]], %[[IV1]]]
//       CHECK:       %[[LHS:.+]] = linalg.matmul
//  CHECK-SAME:         ins(%[[ST_ARG0]], %[[ST_ARG1]]
//  CHECK-SAME:           : tensor<?x?xf32>, tensor<?x?xf32>)
//  CHECK-SAME:         outs(%[[ST_ARG2]] : tensor<?x?xf32>)
//       CHECK:       %[[ST_RESULT:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[LHS]] : tensor<?x?xf32>)
//  CHECK-SAME:         outs(%[[ST_ARG6]] : tensor<?x?xf32>)
//       CHECK:       %[[UPDATE:.+]] = subtensor_insert %[[ST_RESULT]]
//  CHECK-SAME:         into %[[ARG6]][%[[IV0]], %[[IV1]]]
//       CHECK:       scf.yield %[[UPDATE]]
//       CHECK:     scf.yield %[[YIELD]]
//       CHECK:   return %[[RESULT]]
