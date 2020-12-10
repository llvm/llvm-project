// RUN: mlir-opt %s -test-linalg-fusion-transform-patterns -canonicalize -cse -split-input-file -verify-diagnostics | FileCheck %s

module {
  func @basic_fusion(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
                     %arg2: memref<?x?xf32>) {
    %cst = constant 0.000000e+00 : f32
    linalg.fill(%arg2, %cst) : memref<?x?xf32>, f32
    linalg.matmul {__internal_linalg_transform__ = "basic_fusion"}
      ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>)
      outs(%arg2 : memref<?x?xf32>)
    return
  }
}

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0] -> (32, -d0 + s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0)[s0] -> (64, -d0 + s0)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0)[s0] -> (16, -d0 + s0)>
//      CHECK: func @basic_fusion
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: memref<?x?xf32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//  CHECK-DAG:   %[[C32:.+]] = constant 32 : index
//  CHECK-DAG:   %[[C64:.+]] = constant 64 : index
//  CHECK-DAG:   %[[C16:.+]] = constant 16 : index
//  CHECK-DAG:   %[[CST:.+]] = constant 0.0{{.*}} : f32
//  CHECK-DAG:   linalg.fill(%[[ARG2]], %[[CST]])
// CHECK-SAME:   __internal_linalg_transform__ = "after_basic_fusion_original"
//  CHECK-DAG:   %[[M:.+]] = dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[N:.+]] = dim %[[ARG1]], %[[C1]]
//      CHECK:   scf.parallel (%[[IV0:.+]], %[[IV1:.+]]) =
// CHECK-SAME:     to (%[[M]], %[[N]])
// CHECK-SAME:     step (%[[C32]], %[[C64]]) {
//      CHECK:     %[[TILE_M:.+]] = affine.min #[[MAP0]](%[[IV0]])[%[[M]]]
//      CHECK:     %[[K:.+]] = dim %[[ARG0]], %[[C1]]
//      CHECK:     %[[SV1:.+]] = subview %[[ARG0]][%[[IV0]], 0]
// CHECK-SAME:       [%[[TILE_M]], %[[K]]]
//      CHECK:     %[[K_2:.+]] = dim %[[ARG1]], %[[C0]]
//      CHECK:     %[[TILE_N:.+]] = affine.min #[[MAP2]](%[[IV1]])[%[[N]]]
//      CHECK:     %[[SV2:.+]] = subview %[[ARG1]][0, %[[IV1]]]
// CHECK-SAME:       %[[K_2]], %[[TILE_N]]
//      CHECK:     %[[M_2:.+]] = dim %[[ARG2]], %[[C0]]
//      CHECK:     %[[TILE_M_2:.+]] = affine.min #[[MAP0]](%[[IV0]])[%[[M_2]]]
//      CHECK:     %[[N_2:.+]] = dim %[[ARG2]], %[[C1]]
//      CHECK:     %[[TILE_N_2:.+]] = affine.min #[[MAP2]](%[[IV1]])[%[[N_2]]]
//      CHECK:     %[[SV3:.+]] = subview %[[ARG2]][%[[IV0]], %[[IV1]]]
// CHECK-SAME:       [%[[TILE_M_2]], %[[TILE_N_2]]]
//      CHECK:     %[[SV3_2:.+]] = subview %[[ARG2]][%[[IV0]], %[[IV1]]]
// CHECK-SAME:       [%[[TILE_M]], %[[TILE_N]]]
//      CHECK:     linalg.fill(%[[SV3_2]], %[[CST]])
// CHECK-SAME:       __internal_linalg_transform__ = "after_basic_fusion_producer"
//      CHECK:     scf.for %[[IV2:.+]] = %[[C0]] to %[[K]] step %[[C16]] {
//      CHECK:       %[[TILE_K:.+]] = affine.min #[[MAP3]](%[[IV2]])[%[[K]]]
//      CHECK:       %[[SV4:.+]] = subview %[[SV1]][0, %[[IV2]]]
// CHECK-SAME:         [%[[TILE_M]], %[[TILE_K]]]
//      CHECK:       %[[TILE_K_2:.+]] = affine.min #[[MAP3]](%[[IV2]])[%[[K_2]]]
//      CHECK:       %[[SV5:.+]] = subview %[[SV2]][%[[IV2]], 0]
// CHECK-SAME:         [%[[TILE_K_2]], %[[TILE_N]]]
//      CHECK:       linalg.matmul
// CHECK-SAME:         __internal_linalg_transform__ = "after_basic_fusion"
// CHECK-SAME:         ins(%[[SV4]], %[[SV5]]
// CHECK-SAME:           : memref<?x?xf32, #[[MAP1]]>, memref<?x?xf32, #[[MAP1]]>)
// CHECK-SAME:         outs(%[[SV3]] : memref<?x?xf32, #[[MAP1]]>)
//      CHECK:     }
//      CHECK:   }
//      CHECK:   linalg.matmul
// CHECK-SAME:     __internal_linalg_transform__ = "after_basic_fusion_original"

// -----

module {
  func @rhs_fusion(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
                              %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>) {
    %cst = constant 0.000000e+00 : f32
    linalg.copy(%arg1, %arg2) : memref<?x?xf32>, memref<?x?xf32>
    linalg.fill(%arg3, %cst) : memref<?x?xf32>, f32
    linalg.matmul {__internal_linalg_transform__ = "rhs_fusion"}
      ins(%arg0, %arg2 : memref<?x?xf32>, memref<?x?xf32>)
      outs(%arg3 : memref<?x?xf32>)
    return
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0] -> (64, -d0 + s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0)[s0] -> (32, -d0 + s0)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0)[s0] -> (16, -d0 + s0)>
//      CHECK: func @rhs_fusion
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: memref<?x?xf32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: memref<?x?xf32>
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//  CHECK-DAG:   %[[C32:.+]] = constant 32 : index
//  CHECK-DAG:   %[[C64:.+]] = constant 64 : index
//  CHECK-DAG:   %[[C16:.+]] = constant 16 : index
//  CHECK-DAG:   %[[CST:.+]] = constant 0.0{{.*}} : f32
//  CHECK-DAG:   linalg.copy(%[[ARG1]], %[[ARG2]])
// CHECK-SAME:   __internal_linalg_transform__ = "after_rhs_fusion_original"
//  CHECK-DAG:   %[[N:.+]] = dim %[[ARG2]], %[[C1]]
//      CHECK:   scf.parallel (%[[IV0:.+]]) =
// CHECK-SAME:     (%[[C0]]) to (%[[N]]) step (%[[C64]]) {
//      CHECK:     %[[K:.+]] = dim %[[ARG2]], %[[C0]]
//      CHECK:     %[[TILE_N:.+]] = affine.min #[[MAP0]](%[[IV0]])[%[[N]]]
//      CHECK:     %[[SV1:.+]] = subview %[[ARG2]][0, %[[IV0]]]
// CHECK-SAME:       [%[[K]], %[[TILE_N]]]
//      CHECK:     %[[M:.+]] = dim %[[ARG3]], %[[C0]]
//      CHECK:     %[[N_2:.+]] = dim %[[ARG3]], %[[C1]]
//      CHECK:     %[[TILE_N_2:.+]] = affine.min #[[MAP0]](%[[IV0]])[%[[N_2]]]
//      CHECK:     %[[SV2:.+]] = subview %[[ARG3]][0, %[[IV0]]]
// CHECK-SAME:       [%[[M]], %[[TILE_N_2]]]
//      CHECK:     %[[K_2:.+]] = dim %[[ARG1]], %[[C0]]
//      CHECK:     %[[SV3:.+]] = subview %[[ARG1]][0, %[[IV0]]]
// CHECK-SAME:       [%[[K_2]], %[[TILE_N]]]
//      CHECK:     %[[SV3_2:.+]] = subview %[[ARG2]][0, %[[IV0]]]
// CHECK-SAME:       [%[[K_2]], %[[TILE_N]]]
//      CHECK:     linalg.copy(%[[SV3]], %[[SV3_2]])
// CHECK-SAME:       __internal_linalg_transform__ = "after_rhs_fusion_producer"
//  CHECK-NOT:     linalg.fill
//  CHECK-DAG:     %[[M_2:.+]] = dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:     %[[K_2:.+]] = dim %[[ARG0]], %[[C1]]
//      CHECK:     scf.parallel (%[[IV1:.+]]) =
// CHECK-SAME:       (%[[C0]]) to (%[[M_2]]) step (%[[C32]]) {
// CHECK-NEXT:       scf.for %[[IV2:.+]] = %[[C0]] to %[[K_2]] step %[[C16]] {
//      CHECK:         %[[TILE_M:.+]] = affine.min #[[MAP2]](%[[IV1]])[%[[M_2]]]
//      CHECK:         %[[TILE_K:.+]] = affine.min #[[MAP3]](%[[IV2]])[%[[K_2]]]
//      CHECK:         %[[SV4:.+]] = subview %[[ARG0]][%[[IV1]], %[[IV2]]]
// CHECK-SAME:           [%[[TILE_M]], %[[TILE_K]]]
//      CHECK:         %[[TILE_K_2:.+]] = affine.min #[[MAP3]](%[[IV2]])[%[[K]]]
//      CHECK:         %[[SV5:.+]] = subview %[[SV1]][%[[IV2]], 0]
// CHECK-SAME:           [%[[TILE_K_2]], %[[TILE_N]]]
//      CHECK:         %[[TILE_M_2:.+]] = affine.min #[[MAP2]](%[[IV1]])[%[[M]]]
//      CHECK:         %[[SV6:.+]] = subview %[[SV2]][%[[IV1]], 0]
// CHECK-SAME:           [%[[TILE_M_2]], %[[TILE_N_2]]]
//      CHECK:         linalg.matmul
// CHECK-SAME:           __internal_linalg_transform__ = "after_rhs_fusion"
// CHECK-SAME:           ins(%[[SV4]], %[[SV5]]
// CHECK-SAME:             : memref<?x?xf32, #[[MAP1]]>, memref<?x?xf32, #[[MAP1]]>)
// CHECK-SAME:           outs(%[[SV6]] : memref<?x?xf32, #[[MAP1]]>)
//      CHECK:       }
//      CHECK:     }
//      CHECK:   }
//      CHECK:   linalg.matmul
// CHECK-SAME:     __internal_linalg_transform__ = "after_rhs_fusion_original"


// -----

module {
  func @two_operand_fusion(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
                              %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>) {
    %cst = constant 0.000000e+00 : f32
    linalg.copy(%arg0, %arg1) : memref<?x?xf32>, memref<?x?xf32>
    linalg.fill(%arg3, %cst) : memref<?x?xf32>, f32
    linalg.matmul {__internal_linalg_transform__ = "two_operand_fusion"}
      ins(%arg1, %arg2 : memref<?x?xf32>, memref<?x?xf32>)
      outs(%arg3 : memref<?x?xf32>)
    return
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0] -> (32, -d0 + s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0)[s0] -> (16, -d0 + s0)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0)[s0] -> (64, -d0 + s0)>
//      CHECK: func @two_operand_fusion
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: memref<?x?xf32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: memref<?x?xf32>
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//  CHECK-DAG:   %[[C32:.+]] = constant 32 : index
//  CHECK-DAG:   %[[C64:.+]] = constant 64 : index
//  CHECK-DAG:   %[[C16:.+]] = constant 16 : index
//  CHECK-DAG:   %[[CST:.+]] = constant 0.0{{.*}} : f32
//      CHECK:   linalg.copy(%[[ARG0]], %[[ARG1]])
// CHECK-SAME:     __internal_linalg_transform__ = "after_two_operand_fusion_original"
//      CHECK:   linalg.fill(%[[ARG3]], %[[CST]])
// CHECK-SAME:     __internal_linalg_transform__ = "after_two_operand_fusion_original"
//  CHECK-DAG:   %[[M:.+]] = dim %[[ARG1]], %[[C0]]
//      CHECK:   scf.parallel (%[[IV0:.+]]) =
// CHECK-SAME:     (%[[C0]]) to (%[[M]]) step (%[[C32]]) {
//      CHECK:     %[[TILE_M:.+]] = affine.min #[[MAP0]](%[[IV0]])[%[[M]]]
//      CHECK:     %[[K:.+]] = dim %[[ARG1]], %[[C1]]
//      CHECK:     %[[SV1:.+]] = subview %[[ARG1]][%[[IV0]], 0]
// CHECK-SAME:       [%[[TILE_M]], %[[K]]]
//      CHECK:     %[[M_2:.+]] = dim %[[ARG3]], %[[C0]]
//      CHECK:     %[[TILE_M_2:.+]] = affine.min #[[MAP0]](%[[IV0]])[%[[M_2]]]
//      CHECK:     %[[N:.+]] = dim %[[ARG3]], %[[C1]]
//      CHECK:     %[[SV2:.+]] = subview %[[ARG3]][%[[IV0]], 0]
// CHECK-SAME:       [%[[TILE_M_2]], %[[N]]]
//      CHECK:     %[[SV2_2:.+]] = subview %[[ARG3]][%[[IV0]], 0]
// CHECK-SAME:       [%[[TILE_M]], %[[N]]]
//      CHECK:     %[[K_2:.+]] = dim %[[ARG0]], %[[C1]]
//      CHECK:     %[[SV3:.+]] = subview %[[ARG0]][%[[IV0]], 0]
// CHECK-SAME:       [%[[TILE_M]], %[[K_2]]]
//      CHECK:     %[[SV3_2:.+]] = subview %[[ARG1]][%[[IV0]], 0]
// CHECK-SAME:       [%[[TILE_M]], %[[K_2]]]
//      CHECK:     linalg.copy(%[[SV3]], %[[SV3_2]])
// CHECK-SAME:       __internal_linalg_transform__ = "after_two_operand_fusion_producer"
//      CHECK:     linalg.fill(%[[SV2_2]], %[[CST]])
// CHECK-SAME:       __internal_linalg_transform__ = "after_two_operand_fusion_producer"
//  CHECK-DAG:     %[[N_2:.+]] = dim %[[ARG2]], %[[C1]]
//      CHECK:     scf.parallel (%[[IV1:.+]]) =
// CHECK-SAME:       (%[[C0]]) to (%[[N_2]]) step (%[[C64]]) {
// CHECK-NEXT:       scf.for %[[IV2:.+]] = %[[C0]] to %[[K]] step %[[C16]] {
//      CHECK:         %[[TILE_K:.+]] = affine.min #[[MAP2]](%[[IV2]])[%[[K]]]
//      CHECK:         %[[SV4:.+]] = subview %[[SV1]][0, %[[IV2]]]
// CHECK-SAME:           [%[[TILE_M]], %[[TILE_K]]]
//      CHECK:         %[[K_2:.+]] = dim %[[ARG2]], %[[C0]]
//      CHECK:         %[[TILE_K_2:.+]] = affine.min #[[MAP2]](%[[IV2]])[%[[K_2]]]
//      CHECK:         %[[TILE_N:.+]] = affine.min #[[MAP3]](%[[IV1]])[%[[N_2]]]
//      CHECK:         %[[SV5:.+]] = subview %[[ARG2]][%[[IV2]], %[[IV1]]]
// CHECK-SAME:           [%[[TILE_K_2]], %[[TILE_N]]]
//      CHECK:         %[[TILE_N_2:.+]] = affine.min #[[MAP3]](%[[IV1]])[%[[N]]]
//      CHECK:         %[[SV6:.+]] = subview %[[SV2]][0, %[[IV1]]]
// CHECK-SAME:           [%[[TILE_M_2]], %[[TILE_N_2]]]
//      CHECK:         linalg.matmul
// CHECK-SAME:           __internal_linalg_transform__ = "after_two_operand_fusion"
// CHECK-SAME:           ins(%[[SV4]], %[[SV5]]
// CHECK-SAME:             : memref<?x?xf32, #[[MAP1]]>, memref<?x?xf32, #[[MAP1]]>)
// CHECK-SAME:           outs(%[[SV6]] : memref<?x?xf32, #[[MAP1]]>)
//      CHECK:       }
//      CHECK:     }
//      CHECK:   }
//      CHECK:   linalg.matmul
// CHECK-SAME:     __internal_linalg_transform__ = "after_two_operand_fusion_original"

// -----

module {
  func @matmul_fusion(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
                      %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>,
                      %arg4: memref<?x?xf32>) {
    linalg.matmul ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>)
      outs(%arg2 : memref<?x?xf32>)
    linalg.matmul {__internal_linalg_transform__ = "lhs_fusion"}
      ins(%arg2, %arg3 : memref<?x?xf32>, memref<?x?xf32>)
      outs(%arg4 : memref<?x?xf32>)
    return
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0] -> (32, -d0 + s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0)[s0] -> (16, -d0 + s0)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0)[s0] -> (64, -d0 + s0)>
//      CHECK: func @matmul_fusion
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: memref<?x?xf32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: memref<?x?xf32>
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: memref<?x?xf32>
// CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//  CHECK-DAG:   %[[C32:.+]] = constant 32 : index
//  CHECK-DAG:   %[[C64:.+]] = constant 64 : index
//  CHECK-DAG:   %[[C16:.+]] = constant 16 : index
//      CHECK:   linalg.matmul
// CHECK-SAME:     __internal_linalg_transform__ = "after_lhs_fusion_original"
//  CHECK-DAG:   %[[M:.+]] = dim %[[ARG2]], %[[C0]]
//      CHECK:   scf.parallel (%[[IV0:.+]]) =
// CHECK-SAME:     (%[[C0]]) to (%[[M]]) step (%[[C32]]) {
//      CHECK:     %[[TILE_M:.+]] = affine.min #[[MAP0]](%[[IV0]])[%[[M]]]
//      CHECK:     %[[K2:.+]] = dim %[[ARG2]], %[[C1]]
//      CHECK:     %[[SV1:.+]] = subview %[[ARG2]][%[[IV0]], 0]
// CHECK-SAME:       [%[[TILE_M]], %[[K2]]]
//      CHECK:     %[[M_2:.+]] = dim %[[ARG4]], %[[C0]]
//      CHECK:     %[[TILE_M_2:.+]] = affine.min #[[MAP0]](%[[IV0]])[%[[M_2]]]
//      CHECK:     %[[N:.+]] = dim %[[ARG4]], %[[C1]]
//      CHECK:     %[[SV2:.+]] = subview %[[ARG4]][%[[IV0]], 0]
// CHECK-SAME:       [%[[TILE_M_2]], %[[N]]]
//      CHECK:     %[[K2_2:.+]] = dim %[[ARG1]], %[[C1]]
//      CHECK:     %[[K1:.+]] = dim %[[ARG0]], %[[C1]]
//      CHECK:     %[[SV3:.+]] = subview %[[ARG0]][%[[IV0]], 0]
// CHECK-SAME:       [%[[TILE_M]], %[[K1]]]
//      CHECK:     %[[SV4:.+]] = subview %[[ARG1]][0, 0] [%[[K1]], %[[K2_2]]]
//      CHECK:     %[[SV1_2:.+]] = subview %[[ARG2]][%[[IV0]], 0]
// CHECK-SAME:       [%[[TILE_M]], %[[K2_2]]]
//      CHECK:     linalg.matmul
// CHECK-SAME:         __internal_linalg_transform__ = "after_lhs_fusion_producer"
// CHECK-SAME:         ins(%[[SV3]], %[[SV4]]
// CHECK-SAME:           : memref<?x?xf32, #[[MAP1]]>, memref<?x?xf32, #[[MAP1]]>)
// CHECK-SAME:         outs(%[[SV1_2]] : memref<?x?xf32, #[[MAP1]]>)
//  CHECK-DAG:     %[[N_2:.+]] = dim %[[ARG3]], %[[C1]]
//      CHECK:     scf.parallel (%[[IV1:.+]]) =
// CHECK-SAME:       (%[[C0]]) to (%[[N_2]]) step (%[[C64]]) {
// CHECK-NEXT:       scf.for %[[IV2:.+]] = %[[C0]] to %[[K]] step %[[C16]] {
//      CHECK:         %[[TILE_K:.+]] = affine.min #[[MAP2]](%[[IV2]])[%[[K]]]
//      CHECK:         %[[SV6:.+]] = subview %[[SV1]][0, %[[IV2]]]
// CHECK-SAME:           [%[[TILE_M]], %[[TILE_K]]]
//      CHECK:         %[[K_2:.+]] = dim %[[ARG3]], %[[C0]]
//      CHECK:         %[[TILE_K_2:.+]] = affine.min #[[MAP2]](%[[IV2]])[%[[K_2]]]
//      CHECK:         %[[TILE_N:.+]] = affine.min #[[MAP3]](%[[IV1]])[%[[N_2]]]
//      CHECK:         %[[SV7:.+]] = subview %[[ARG3]][%[[IV2]], %[[IV1]]]
// CHECK-SAME:           [%[[TILE_K_2]], %[[TILE_N]]]
//      CHECK:         %[[TILE_N_2:.+]] = affine.min #[[MAP3]](%[[IV1]])[%[[N]]]
//      CHECK:         %[[SV8:.+]] = subview %[[SV2]][0, %[[IV1]]]
// CHECK-SAME:           [%[[TILE_M_2]], %[[TILE_N_2]]]
//      CHECK:         linalg.matmul
// CHECK-SAME:           __internal_linalg_transform__ = "after_lhs_fusion"
// CHECK-SAME:           ins(%[[SV6]], %[[SV7]]
// CHECK-SAME:             : memref<?x?xf32, #[[MAP1]]>, memref<?x?xf32, #[[MAP1]]>)
// CHECK-SAME:           outs(%[[SV8]] : memref<?x?xf32, #[[MAP1]]>)
//      CHECK:       }
//      CHECK:     }
//      CHECK:   }
//      CHECK:   linalg.matmul
// CHECK-SAME:     __internal_linalg_transform__ = "after_lhs_fusion_original"

// -----

module {
  func @matmul_plus_matmul(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
                           %arg2: memref<?x?xf32>) {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %0 = dim %arg2, %c0 : memref<?x?xf32>
    %1 = dim %arg2, %c1 : memref<?x?xf32>
    %2 = alloc(%0, %1) : memref<?x?xf32>
    linalg.matmul ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>)
      outs(%2 : memref<?x?xf32>)
    linalg.generic
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"],
       __internal_linalg_transform__ = "transpose_fusion"}
      ins(%2, %2 : memref<?x?xf32>, memref<?x?xf32>)
      outs(%arg2 : memref<?x?xf32>) {
      ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32) :
        %3 = addf %arg3, %arg4 : f32
        linalg.yield %3 : f32
      }
    return
  }
}
//       CHECK: func @matmul_plus_matmul
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//       CHECK:   %[[T2:.+]] = alloc(%{{.*}}, %{{.*}}) : memref<?x?xf32>
//       CHECK:   linalg.matmul
//  CHECK-SAME:     after_transpose_fusion_original
//       CHECK:   scf.parallel (%[[ARG3:[a-zA-Z0-9_]+]], %[[ARG4:.[a-zA-Z0-9_]+]])
//       CHECK:     %[[T5:.+]] = subview %[[T2]][%[[ARG3]], %[[ARG4]]]
//       CHECK:     %[[T6:.+]] = subview %[[ARG2]][%[[ARG3]], %[[ARG4]]]
//       CHECK:     %[[T8:.+]] = subview %[[ARG0]][%[[ARG3]], 0]
//       CHECK:     %[[T9:.+]] = subview %[[ARG1]][0, %[[ARG4]]]
//       CHECK:     linalg.matmul
//  CHECK-SAME:       after_transpose_fusion_producer
//  CHECK-SAME:       ins(%[[T8]], %[[T9]]
//  CHECK-SAME:       outs(%[[T5]]
//   CHECK-NOT:     linalg.matmul
//       CHECK:     linalg.generic
//  CHECK-SAME:       ins(%[[T5]], %[[T5]]
//  CHECK-SAME:       outs(%[[T6]]
//  CHECK-SAME:       after_transpose_fusion

// -----

module {
  func @matmul_plus_transpose_matmul(%arg0: memref<?x?xf32>,
                                     %arg1: memref<?x?xf32>,
                                     %arg2: memref<?x?xf32>) {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %0 = dim %arg2, %c0 : memref<?x?xf32>
    %1 = dim %arg2, %c1 : memref<?x?xf32>
    %2 = alloc(%0, %1) : memref<?x?xf32>
    linalg.matmul ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>)
      outs(%2 : memref<?x?xf32>)
    // expected-remark @+1 {{unhandled fusion to the same producer but with different indexing maps}}
    linalg.generic
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d1, d0)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"],
       __internal_linalg_transform__ = "transpose_fusion"}
      ins(%2, %2 : memref<?x?xf32>, memref<?x?xf32>)
      outs(%arg2 : memref<?x?xf32>) {
      ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32) :
        %3 = addf %arg3, %arg4 : f32
        linalg.yield %3 : f32
      }
    return
  }
}

// -----

#map0 = affine_map<(d0)[s0] -> (32, -d0 + s0)>
#map1 = affine_map<(d0)[s0] -> (64, -d0 + s0)>
#map2 = affine_map<(d0)[s0] -> (16, -d0 + s0)>
#map3 = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
module {
  func @basic_no_fusion(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
                        %arg2: memref<?x?xf32>) {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %c32 = constant 32 : index
    %c64 = constant 64 : index
    %c16 = constant 16 : index
    %cst = constant 0.000000e+00 : f32
    linalg.fill(%arg2, %cst) : memref<?x?xf32>, f32
    %0 = dim %arg0, %c0 : memref<?x?xf32>
    %1 = dim %arg1, %c1 : memref<?x?xf32>
    %2 = dim %arg0, %c1 : memref<?x?xf32>
    scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%0, %1) step (%c32, %c64) {
      scf.for %arg5 = %c0 to %2 step %c16 {
        %3 = affine.min #map0(%arg3)[%0]
        %4 = affine.min #map1(%arg4)[%1]
        %5 = affine.min #map2(%arg5)[%2]
        %6 = subview %arg0[%arg3, %arg5] [%3, %5] [1, 1] : memref<?x?xf32> to memref<?x?xf32, #map3>
        %7 = subview %arg1[%arg5, %arg4] [%5, %4] [1, 1] : memref<?x?xf32> to memref<?x?xf32, #map3>
        %8 = subview %arg2[%arg3, %arg4] [%3, %4] [1, 1] : memref<?x?xf32> to memref<?x?xf32, #map3>
	// expected-remark @+1 {{unhandled fusion of ops in different basic blocks}}
        linalg.matmul {__internal_linalg_transform__ = "basic_fusion"}
          ins(%6, %7 : memref<?x?xf32, #map3>, memref<?x?xf32, #map3>)
          outs(%8 : memref<?x?xf32, #map3>)
      }
      scf.yield
    }
    return
  }
}

// -----

module {
  func @basic_conv_fusion(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>,
                          %arg2: memref<?x?x?x?xf32>) {
    %cst = constant 0.000000e+00 : f32
    linalg.fill(%arg2, %cst) : memref<?x?x?x?xf32>, f32
    linalg.conv(%arg0, %arg1, %arg2) {
      dilations = [1, 1], strides = [1, 1],
      __internal_linalg_transform__ = "basic_fusion"} :
      memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
    return
  }
}
//      CHECK: func @basic_conv_fusion
//      CHECK:   linalg.fill
// CHECK-SAME:     __internal_linalg_transform__ = "after_basic_fusion_original"
//      CHECK:  scf.parallel (%{{.+}}, %{{.+}}, %{{.+}})
// CHECK-SAME:  {
//      CHECK:    linalg.fill
// CHECK-SAME:      __internal_linalg_transform__ = "after_basic_fusion_producer"
//      CHECK:    linalg.conv
// CHECK-SAME:      __internal_linalg_transform__ = "after_basic_fusion"
//      CHECK:  }
//      CHECK:  linalg.conv
// CHECK-SAME:    __internal_linalg_transform__ = "after_basic_fusion_original"
