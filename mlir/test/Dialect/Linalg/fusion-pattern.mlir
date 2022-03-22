// RUN: mlir-opt %s -test-linalg-fusion-transform-patterns -canonicalize -cse -split-input-file | FileCheck %s

module {
  func @basic_fusion(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
                     %arg2: memref<?x?xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    linalg.fill ins(%cst : f32) outs(%arg2 : memref<?x?xf32>)
    linalg.matmul {__internal_linalg_transform__ = "basic_fusion"}
      ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>)
      outs(%arg2 : memref<?x?xf32>)
    return
  }
}

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 32)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 64)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 16)>
//  CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0)[s0, s1] -> (-d0 + s1, -d0 + s0, 32)>
//  CHECK-DAG: #[[MAP5:.+]] = affine_map<(d0)[s0, s1] -> (-d0 + s1, -d0 + s0, 64)>
//      CHECK: func @basic_fusion
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: memref<?x?xf32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C32:.+]] = arith.constant 32 : index
//  CHECK-DAG:   %[[C64:.+]] = arith.constant 64 : index
//  CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
//  CHECK-DAG:   %[[CST:.+]] = arith.constant 0.0{{.*}} : f32
//  CHECK-DAG:   linalg.fill
// CHECK-SAME:   __internal_linalg_transform__ = "after_basic_fusion_original"
// CHECK-SAME:   ins(%[[CST]]{{.*}}outs(%[[ARG2]]
//  CHECK-DAG:   %[[M:.+]] = memref.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[N:.+]] = memref.dim %[[ARG1]], %[[C1]]
//      CHECK:   scf.parallel (%[[IV0:.+]], %[[IV1:.+]]) =
// CHECK-SAME:     to (%[[M]], %[[N]])
// CHECK-SAME:     step (%[[C32]], %[[C64]]) {
//      CHECK:     %[[TILE_M:.+]] = affine.min #[[MAP0]](%[[IV0]])[%[[M]]]
//      CHECK:     %[[K:.+]] = memref.dim %[[ARG0]], %[[C1]]
//      CHECK:     %[[SV1:.+]] = memref.subview %[[ARG0]][%[[IV0]], 0]
// CHECK-SAME:       [%[[TILE_M]], %[[K]]]
//      CHECK:     %[[K_2:.+]] = memref.dim %[[ARG1]], %[[C0]]
//      CHECK:     %[[TILE_N:.+]] = affine.min #[[MAP2]](%[[IV1]])[%[[N]]]
//      CHECK:     %[[SV2:.+]] = memref.subview %[[ARG1]][0, %[[IV1]]]
// CHECK-SAME:       %[[K_2]], %[[TILE_N]]
//      CHECK:     %[[SV3:.+]] = memref.subview %[[ARG2]][%[[IV0]], %[[IV1]]]
// CHECK-SAME:       [%[[TILE_M]], %[[TILE_N]]]
//      CHECK:     %[[M_2:.+]] = memref.dim %[[ARG2]], %[[C0]]
//      CHECK:     %[[N_2:.+]] = memref.dim %[[ARG2]], %[[C1]]
//      CHECK:     %[[TILE_M_3:.+]] = affine.min #[[MAP4]](%[[IV0]])[%[[M_2]], %[[M]]]
//      CHECK:     %[[TILE_N_3:.+]] = affine.min #[[MAP5]](%[[IV1]])[%[[N_2]], %[[N]]]
//      CHECK:     %[[SV3_2:.+]] = memref.subview %[[ARG2]][%[[IV0]], %[[IV1]]]
// CHECK-SAME:       [%[[TILE_M_3]], %[[TILE_N_3]]]
//      CHECK:     linalg.fill
// CHECK-SAME:       __internal_linalg_transform__ = "after_basic_fusion_producer"
// CHECK-SAME:       ins(%[[CST]]{{.*}}outs(%[[SV3_2]]
//      CHECK:     scf.for %[[IV2:.+]] = %[[C0]] to %[[K]] step %[[C16]] {
//      CHECK:       %[[TILE_K:.+]] = affine.min #[[MAP3]](%[[IV2]])[%[[K]]]
//      CHECK:       %[[SV4:.+]] = memref.subview %[[SV1]][0, %[[IV2]]]
// CHECK-SAME:         [%[[TILE_M]], %[[TILE_K]]]
//      CHECK:       %[[SV5:.+]] = memref.subview %[[SV2]][%[[IV2]], 0]
// CHECK-SAME:         [%[[TILE_K]], %[[TILE_N]]]
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
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 32)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 16)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 64)>
//  CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0)[s0, s1] -> (-d0 + s1, -d0 + s0, 32)>
//      CHECK: func @matmul_fusion
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: memref<?x?xf32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: memref<?x?xf32>
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: memref<?x?xf32>
// CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C32:.+]] = arith.constant 32 : index
//  CHECK-DAG:   %[[C64:.+]] = arith.constant 64 : index
//  CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
//      CHECK:   linalg.matmul
// CHECK-SAME:     __internal_linalg_transform__ = "after_lhs_fusion_original"
//  CHECK-DAG:   %[[M:.+]] = memref.dim %[[ARG2]], %[[C0]]
//      CHECK:   scf.parallel (%[[IV0:.+]]) =
// CHECK-SAME:     (%[[C0]]) to (%[[M]]) step (%[[C32]]) {
//      CHECK:     %[[TILE_M:.+]] = affine.min #[[MAP0]](%[[IV0]])[%[[M]]]
//      CHECK:     %[[K2:.+]] = memref.dim %[[ARG2]], %[[C1]]
//      CHECK:     %[[SV1:.+]] = memref.subview %[[ARG2]][%[[IV0]], 0]
// CHECK-SAME:       [%[[TILE_M]], %[[K2]]]
//      CHECK:     %[[N:.+]] = memref.dim %[[ARG4]], %[[C1]]
//      CHECK:     %[[SV2:.+]] = memref.subview %[[ARG4]][%[[IV0]], 0]
// CHECK-SAME:       [%[[TILE_M]], %[[N]]]
//      CHECK:     %[[M_3:.+]] = memref.dim %[[ARG0]], %[[C0]]
//      CHECK:     %[[TILE_M_3:.+]] = affine.min #[[MAP4]](%[[IV0]])[%[[M_3]], %[[M]]]
//      CHECK:     %[[K1:.+]] = memref.dim %[[ARG0]], %[[C1]]
//      CHECK:     %[[SV3:.+]] = memref.subview %[[ARG0]][%[[IV0]], 0]
// CHECK-SAME:       [%[[TILE_M_3]], %[[K1]]]
//      CHECK:     %[[SV1_2:.+]] = memref.subview %[[ARG2]][%[[IV0]], 0]
// CHECK-SAME:       [%[[TILE_M_3]], %[[K2]]]
//      CHECK:     linalg.matmul
// CHECK-SAME:         __internal_linalg_transform__ = "after_lhs_fusion_producer"
// CHECK-SAME:         ins(%[[SV3]], %[[ARG1]]
// CHECK-SAME:           : memref<?x?xf32, #[[MAP1]]>, memref<?x?xf32>)
// CHECK-SAME:         outs(%[[SV1_2]] : memref<?x?xf32, #[[MAP1]]>)
//      CHECK:     %[[N_2:.+]] = memref.dim %[[ARG3]], %[[C1]]
//      CHECK:     scf.parallel (%[[IV1:.+]]) =
// CHECK-SAME:       (%[[C0]]) to (%[[N_2]]) step (%[[C64]]) {
// CHECK-NEXT:       scf.for %[[IV2:.+]] = %[[C0]] to %[[K2]] step %[[C16]] {
//      CHECK:         %[[TILE_K:.+]] = affine.min #[[MAP2]](%[[IV2]])[%[[K2]]]
//      CHECK:         %[[SV6:.+]] = memref.subview %[[SV1]][0, %[[IV2]]]
// CHECK-SAME:           [%[[TILE_M]], %[[TILE_K]]]
//      CHECK:         %[[TILE_N:.+]] = affine.min #[[MAP3]](%[[IV1]])[%[[N_2]]]
//      CHECK:         %[[SV7:.+]] = memref.subview %[[ARG3]][%[[IV2]], %[[IV1]]]
// CHECK-SAME:           [%[[TILE_K]], %[[TILE_N]]]
//      CHECK:         %[[SV8:.+]] = memref.subview %[[SV2]][0, %[[IV1]]]
// CHECK-SAME:           [%[[TILE_M]], %[[TILE_N]]]
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
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.dim %arg2, %c0 : memref<?x?xf32>
    %1 = memref.dim %arg2, %c1 : memref<?x?xf32>
    %2 = memref.alloc(%0, %1) : memref<?x?xf32>
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
        %3 = arith.addf %arg3, %arg4 : f32
        linalg.yield %3 : f32
      }
    return
  }
}
//       CHECK: func @matmul_plus_matmul
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//       CHECK:   %[[T2:.+]] = memref.alloc(%{{.*}}, %{{.*}}) : memref<?x?xf32>
//       CHECK:   linalg.matmul
//  CHECK-SAME:     after_transpose_fusion_original
//       CHECK:   scf.parallel (%[[ARG3:[a-zA-Z0-9_]+]], %[[ARG4:.[a-zA-Z0-9_]+]])
//       CHECK:     %[[T5:.+]] = memref.subview %[[T2]][%[[ARG3]], %[[ARG4]]]
//       CHECK:     %[[T6:.+]] = memref.subview %[[ARG2]][%[[ARG3]], %[[ARG4]]]
//       CHECK:     %[[T8:.+]] = memref.subview %[[ARG0]][%[[ARG3]], 0]
//       CHECK:     %[[T9:.+]] = memref.subview %[[ARG1]][0, %[[ARG4]]]
//       CHECK:     %[[T10:.+]] = memref.subview %[[T2]][%[[ARG3]], %[[ARG4]]]
//       CHECK:     linalg.matmul
//  CHECK-SAME:       after_transpose_fusion_producer
//  CHECK-SAME:       ins(%[[T8]], %[[T9]]
//  CHECK-SAME:       outs(%[[T10]]
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
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.dim %arg2, %c0 : memref<?x?xf32>
    %1 = memref.dim %arg2, %c1 : memref<?x?xf32>
    %2 = memref.alloc(%0, %1) : memref<?x?xf32>
    linalg.matmul ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>)
      outs(%2 : memref<?x?xf32>)
    linalg.generic
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d1, d0)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"],
       __internal_linalg_transform__ = "transpose_fusion"}
      ins(%2, %2 : memref<?x?xf32>, memref<?x?xf32>)
      outs(%arg2 : memref<?x?xf32>) {
      ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32) :
        %3 = arith.addf %arg3, %arg4 : f32
        linalg.yield %3 : f32
      }
    return
  }
}
// CHECK-LABEL: func @matmul_plus_transpose_matmul
//   CHECK-NOT:   scf.parallel
//   CHECK-NOT:   scf.for
//       CHECK:   linalg.matmul
//   CHECK-NOT:   scf.parallel
//   CHECK-NOT:   scf.for
//       CHECK:   linalg.generic
//   CHECK-NOT:   scf.parallel
//   CHECK-NOT:   scf.for

// -----

#map0 = affine_map<(d0)[s0] -> (32, -d0 + s0)>
#map1 = affine_map<(d0)[s0] -> (64, -d0 + s0)>
#map2 = affine_map<(d0)[s0] -> (16, -d0 + s0)>
#map3 = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
module {
  func @basic_no_fusion(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
                        %arg2: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 0.000000e+00 : f32
    linalg.fill ins(%cst : f32) outs(%arg2 : memref<?x?xf32>)
    %0 = memref.dim %arg0, %c0 : memref<?x?xf32>
    %1 = memref.dim %arg1, %c1 : memref<?x?xf32>
    %2 = memref.dim %arg0, %c1 : memref<?x?xf32>
    scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%0, %1) step (%c32, %c64) {
      scf.for %arg5 = %c0 to %2 step %c16 {
        %3 = affine.min #map0(%arg3)[%0]
        %4 = affine.min #map1(%arg4)[%1]
        %5 = affine.min #map2(%arg5)[%2]
        %6 = memref.subview %arg0[%arg3, %arg5] [%3, %5] [1, 1] : memref<?x?xf32> to memref<?x?xf32, #map3>
        %7 = memref.subview %arg1[%arg5, %arg4] [%5, %4] [1, 1] : memref<?x?xf32> to memref<?x?xf32, #map3>
        %8 = memref.subview %arg2[%arg3, %arg4] [%3, %4] [1, 1] : memref<?x?xf32> to memref<?x?xf32, #map3>
        linalg.matmul {__internal_linalg_transform__ = "basic_fusion"}
          ins(%6, %7 : memref<?x?xf32, #map3>, memref<?x?xf32, #map3>)
          outs(%8 : memref<?x?xf32, #map3>)
      }
      scf.yield
    }
    return
  }
}
// CHECK-LABEL: func @basic_no_fusion
//   CHECK-NOT:   scf.parallel
//       CHECK:   linalg.fill
//       CHECK:   scf.parallel
//       CHECK:     scf.for
//   CHECK-NOT:     linalg.fill
//       CHECK:     linalg.matmul

// -----

module {
  func @basic_conv_fusion(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
                          %arg2: memref<?x?xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    linalg.fill ins(%cst : f32) outs(%arg2 : memref<?x?xf32>)
    linalg.conv_2d {__internal_linalg_transform__ = "basic_fusion"}
      ins(%arg1, %arg0 : memref<?x?xf32>, memref<?x?xf32>) outs(%arg2 : memref<?x?xf32>)
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
//      CHECK:    linalg.conv_2d
// CHECK-SAME:      __internal_linalg_transform__ = "after_basic_fusion"
//      CHECK:  }
//      CHECK:  linalg.conv_2d
// CHECK-SAME:    __internal_linalg_transform__ = "after_basic_fusion_original"
