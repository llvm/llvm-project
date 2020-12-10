// RUN: mlir-opt -pass-pipeline="func(test-linalg-tile-and-fuse{tile-sizes=16,32,64}),canonicalize,cse" -split-input-file %s | FileCheck %s

module {
  func @three_op_fusion(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
                        %arg2: memref<?xf32>, %arg3 : memref<?x?xf32>) {
    %cst = constant 0.000000e+00 : f32
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %d0 = dim %arg0, %c0 : memref<?x?xf32>
    %d1 = dim %arg1, %c1 : memref<?x?xf32>
    %0 = alloc(%d0, %d1) : memref<?x?xf32>
    linalg.fill(%0, %cst) : memref<?x?xf32>, f32
    linalg.matmul ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>)
      outs(%0 : memref<?x?xf32>)
    linalg.generic
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]}
      ins(%0, %arg2 : memref<?x?xf32>, memref<?xf32>)
      outs(%arg3 : memref<?x?xf32>) {
      ^bb0(%arg4 : f32, %arg5 : f32, %arg6 : f32) :
        %5 = addf %arg4, %arg5 : f32
        linalg.yield %5 : f32
      }
    return
  }
}

//   CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
//   CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
//       CHECK: func @three_op_fusion
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: memref<?xf32>
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//       CHECK:   %[[TEMP:.+]] = alloc(%{{.*}}, %{{.*}}) : memref<?x?xf32>
//       CHECK:   scf.parallel (%[[IV0:.+]], %[[IV1:.+]]) = {{.*}} {
//   CHECK-DAG:     %[[SV_TEMP:.+]] = subview %[[TEMP]][%[[IV0]], %[[IV1]]]
//   CHECK-DAG:     %[[SV_ARG2:.+]] = subview %[[ARG2]][%[[IV1]]]
//   CHECK-DAG:     %[[SV_ARG3:.+]] = subview %[[ARG3]][%[[IV0]], %[[IV1]]]
//   CHECK-DAG:     %[[SV_ARG0:.+]] = subview %[[ARG0]][%[[IV0]], 0]
//   CHECK-DAG:     %[[SV_ARG1:.+]] = subview %[[ARG1]][0, %[[IV1]]]
//       CHECK:     linalg.fill(%[[SV_TEMP]], %{{.+}})
//       CHECK:     linalg.matmul
//  CHECK-SAME:       ins(%[[SV_ARG0]], %[[SV_ARG1]]
//  CHECK-SAME:         : memref<?x?xf32, #[[MAP2]]>, memref<?x?xf32, #[[MAP2]]>)
//  CHECK-SAME:       outs(%[[SV_TEMP]] : memref<?x?xf32, #[[MAP2]]>)
//       CHECK:     linalg.generic
//  CHECK-SAME:       ins(%[[SV_TEMP]], %[[SV_ARG2]]
//  CHECK-SAME:         : memref<?x?xf32, #[[MAP2]]>, memref<?xf32, #[[MAP3]]>)
//  CHECK-SAME:       outs(%[[SV_ARG3]] : memref<?x?xf32, #[[MAP2]]>)
//       CHECK:     scf.yield
//       CHECK:   }

// -----

module {
  func @sequence_of_matmul(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
                           %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>,
			   %arg4: memref<?x?xf32>) {
    %cst = constant 0.000000e+00 : f32
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %m = dim %arg0, %c0 : memref<?x?xf32>
    %n1 = dim %arg1, %c1 : memref<?x?xf32>
    %n2 = dim %arg2, %c1 : memref<?x?xf32>
    %n3 = dim %arg3, %c1 : memref<?x?xf32>
    %0 = alloc(%m, %n1) : memref<?x?xf32>
    %1 = alloc(%m, %n2) : memref<?x?xf32>
    linalg.fill(%0, %cst) : memref<?x?xf32>, f32
    linalg.matmul ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>)
      outs(%0 : memref<?x?xf32>)
    linalg.fill(%1, %cst) : memref<?x?xf32>, f32
    linalg.matmul ins(%0, %arg2 : memref<?x?xf32>, memref<?x?xf32>)
      outs(%1 : memref<?x?xf32>)
    linalg.fill(%arg4, %cst) : memref<?x?xf32>, f32
    linalg.matmul ins(%1, %arg3 : memref<?x?xf32>, memref<?x?xf32>)
      outs(%arg4 : memref<?x?xf32>)
    return
  }
}

//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0] -> (16, -d0 + s0)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
//       CHECK: func @sequence_of_matmul
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[C16:.+]] = constant 16 : index
//   CHECK-DAG:   %[[M:.+]] = dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[N1:.+]] = dim %[[ARG1]], %[[C1]]
//   CHECK-DAG:   %[[N2:.+]] = dim %[[ARG2]], %[[C1]]
//       CHECK:   %[[ALLOC1:.+]] = alloc(%[[M]], %[[N1]])
//       CHECK:   %[[ALLOC2:.+]] = alloc(%[[M]], %[[N2]])
//       CHECK:   scf.parallel (%[[IV0:.+]]) = (%[[C0]]) to (%[[M]])
//  CHECK-SAME:     step (%[[C16]]) {
//       CHECK:     %[[TILE_M:.+]] = affine.min #[[MAP0]](%[[IV0]])[%[[M]]]
//       CHECK:     %[[SV_ALLOC2:.+]] = subview %[[ALLOC2]][%[[IV0]], 0]
//  CHECK-SAME:       [%[[TILE_M]], %[[N2]]]
//       CHECK:     %[[M_2:.+]] = dim %[[ARG4]], %[[C0]]
//       CHECK:     %[[TILE_M_2:.+]] = affine.min #[[MAP0]](%[[IV0]])[%[[M_2]]]
//       CHECK:     %[[N3:.+]] = dim %[[ARG4]], %[[C1]]
//       CHECK:     %[[SV_ARG4:.+]] = subview %[[ARG4]][%[[IV0]], 0]
//  CHECK-SAME:       [%[[TILE_M_2]], %[[N3]]]
//       CHECK:     %[[SV_ARG4_2:.+]] = subview %[[ARG4]][%[[IV0]], 0]
//  CHECK-SAME:       [%[[TILE_M]], %[[N3]]]
//       CHECK:     %[[SV_ALLOC1:.+]] = subview %[[ALLOC1]][%[[IV0]], 0]
//  CHECK-SAME:       [%[[TILE_M]], %[[N1]]]
//       CHECK:     %[[SV_ARG2:.+]] = subview %[[ARG2]][0, 0] [%[[N1]], %[[N2]]]
//       CHECK:     %[[N0:.+]] = dim %[[ARG0]], %[[C1]]
//       CHECK:     %[[SV_ARG0:.+]] = subview %[[ARG0]][%[[IV0]], 0]
//  CHECK-SAME:       [%[[TILE_M:.+]], %[[N0]]]
//       CHECK:     %[[SV_ARG1:.+]] = subview %[[ARG1]][0, 0] [%[[N0]], %[[N1]]]
//       CHECK:     linalg.fill(%[[SV_ALLOC1]], %{{.+}})
//       CHECK:     linalg.matmul ins(%[[SV_ARG0]], %[[SV_ARG1]]
//  CHECK-SAME:        : memref<?x?xf32, #[[MAP1]]>, memref<?x?xf32, #[[MAP1]]>)
//  CHECK-SAME:        outs(%[[SV_ALLOC1]] : memref<?x?xf32, #[[MAP1]]>)
//       CHECK:     linalg.fill(%[[SV_ALLOC2]], %{{.+}})
//       CHECK:     linalg.matmul ins(%[[SV_ALLOC1]], %[[SV_ARG2]]
//  CHECK-SAME:        : memref<?x?xf32, #[[MAP1]]>, memref<?x?xf32, #[[MAP1]]>)
//  CHECK-SAME:        outs(%[[SV_ALLOC2]] : memref<?x?xf32, #[[MAP1]]>)
//       CHECK:     linalg.fill(%[[SV_ARG4_2]], %{{.+}})
//       CHECK:     linalg.matmul ins(%[[SV_ALLOC2]], %[[ARG3]]
//  CHECK-SAME:        : memref<?x?xf32, #[[MAP1]]>, memref<?x?xf32>)
//  CHECK-SAME:        outs(%[[SV_ARG4]] : memref<?x?xf32, #[[MAP1]]>)
//       CHECK:     scf.yield
//       CHECK:   }

