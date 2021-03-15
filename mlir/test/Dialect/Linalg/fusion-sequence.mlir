// RUN: mlir-opt -pass-pipeline="func(test-linalg-tile-and-fuse{tile-sizes=16,32,64}),canonicalize,cse" -split-input-file %s | FileCheck %s

module {
  func @three_op_fusion(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
                        %arg2: memref<?xf32>, %arg3 : memref<?x?xf32>) {
    %cst = constant 0.000000e+00 : f32
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %d0 = memref.dim %arg0, %c0 : memref<?x?xf32>
    %d1 = memref.dim %arg1, %c1 : memref<?x?xf32>
    %0 = memref.alloc(%d0, %d1) : memref<?x?xf32>
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
//       CHECK:   %[[TEMP:.+]] = memref.alloc(%{{.*}}, %{{.*}}) : memref<?x?xf32>
//       CHECK:   scf.parallel (%[[IV0:.+]], %[[IV1:.+]]) = {{.*}} {
//   CHECK-DAG:     %[[SV_TEMP:.+]] = memref.subview %[[TEMP]][%[[IV0]], %[[IV1]]]
//   CHECK-DAG:     %[[SV_ARG2:.+]] = memref.subview %[[ARG2]][%[[IV1]]]
//   CHECK-DAG:     %[[SV_ARG3:.+]] = memref.subview %[[ARG3]][%[[IV0]], %[[IV1]]]
//   CHECK-DAG:     %[[SV_ARG0:.+]] = memref.subview %[[ARG0]][%[[IV0]], 0]
//   CHECK-DAG:     %[[SV_ARG1:.+]] = memref.subview %[[ARG1]][0, %[[IV1]]]
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
    %m = memref.dim %arg0, %c0 : memref<?x?xf32>
    %n1 = memref.dim %arg1, %c1 : memref<?x?xf32>
    %n2 = memref.dim %arg2, %c1 : memref<?x?xf32>
    %n3 = memref.dim %arg3, %c1 : memref<?x?xf32>
    %0 = memref.alloc(%m, %n1) : memref<?x?xf32>
    %1 = memref.alloc(%m, %n2) : memref<?x?xf32>
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
//   CHECK-DAG:   %[[M:.+]] = memref.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[N1:.+]] = memref.dim %[[ARG1]], %[[C1]]
//   CHECK-DAG:   %[[N2:.+]] = memref.dim %[[ARG2]], %[[C1]]
//       CHECK:   %[[ALLOC1:.+]] = memref.alloc(%[[M]], %[[N1]])
//       CHECK:   %[[ALLOC2:.+]] = memref.alloc(%[[M]], %[[N2]])
//       CHECK:   scf.parallel (%[[IV0:.+]]) = (%[[C0]]) to (%[[M]])
//  CHECK-SAME:     step (%[[C16]]) {
//       CHECK:     %[[TILE_M:.+]] = affine.min #[[MAP0]](%[[IV0]])[%[[M]]]
//       CHECK:     %[[SV_ALLOC2:.+]] = memref.subview %[[ALLOC2]][%[[IV0]], 0]
//  CHECK-SAME:       [%[[TILE_M]], %[[N2]]]
//       CHECK:     %[[M_2:.+]] = memref.dim %[[ARG4]], %[[C0]]
//       CHECK:     %[[TILE_M_2:.+]] = affine.min #[[MAP0]](%[[IV0]])[%[[M_2]]]
//       CHECK:     %[[N3:.+]] = memref.dim %[[ARG4]], %[[C1]]
//       CHECK:     %[[SV_ARG4:.+]] = memref.subview %[[ARG4]][%[[IV0]], 0]
//  CHECK-SAME:       [%[[TILE_M_2]], %[[N3]]]
//       CHECK:     %[[SV_ARG4_2:.+]] = memref.subview %[[ARG4]][%[[IV0]], 0]
//  CHECK-SAME:       [%[[TILE_M]], %[[N3]]]
//       CHECK:     %[[SV_ALLOC1:.+]] = memref.subview %[[ALLOC1]][%[[IV0]], 0]
//  CHECK-SAME:       [%[[TILE_M]], %[[N1]]]
//       CHECK:     %[[SV_ARG2:.+]] = memref.subview %[[ARG2]][0, 0] [%[[N1]], %[[N2]]]
//       CHECK:     %[[N0:.+]] = memref.dim %[[ARG0]], %[[C1]]
//       CHECK:     %[[SV_ARG0:.+]] = memref.subview %[[ARG0]][%[[IV0]], 0]
//  CHECK-SAME:       [%[[TILE_M:.+]], %[[N0]]]
//       CHECK:     %[[SV_ARG1:.+]] = memref.subview %[[ARG1]][0, 0] [%[[N0]], %[[N1]]]
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

// -----

module {
  func @tensor_op_fusion(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
                         %arg2: tensor<?x?xf32>, %arg3: tensor<?xf32>)
    -> tensor<?x?xf32> {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %1 = memref.dim %0, %c0 : tensor<?x?xf32>
    %2 = memref.dim %0, %c1 : tensor<?x?xf32>
    %3 = linalg.init_tensor [%1, %2] : tensor<?x?xf32>
    %4 = linalg.generic
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]}
      ins(%0, %arg3 : tensor<?x?xf32>, tensor<?xf32>)
      outs(%3 : tensor<?x?xf32>) {
      ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
        %5 = addf %arg4, %arg5 : f32
        linalg.yield %5 : f32
      } -> tensor<?x?xf32>
    return %4 : tensor<?x?xf32>
  }
}
// CHECK-LABEL: func @tensor_op_fusion
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: tensor<?xf32>
//       CHECK:   %[[INIT:.+]] = linalg.init_tensor
//       CHECK:   %[[R0:.+]] = scf.for %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[ARG5:.+]] = %[[INIT]]) -> (tensor<?x?xf32>) {
//       CHECK:     %[[R1:.+]] = scf.for %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[ARG7:.+]] = %[[ARG5]]) -> (tensor<?x?xf32>) {
//   CHECK-DAG:       %[[STARG3:.+]] = subtensor %[[ARG3]]
//   CHECK-DAG:       %[[STARG7:.+]] = subtensor %[[ARG7]]
//   CHECK-DAG:       %[[STARG0:.+]] = subtensor %[[ARG0]]
//   CHECK-DAG:       %[[STARG1:.+]] = subtensor %[[ARG1]]
//   CHECK-DAG:       %[[STARG2:.+]] = subtensor %[[ARG2]]
//       CHECK:       %[[T0:.+]] = linalg.matmul
//  CHECK-SAME:         ins(%[[STARG0]], %[[STARG1]] : tensor<?x?xf32>, tensor<?x?xf32>)
//  CHECK-SAME:         outs(%[[STARG2]] : tensor<?x?xf32>) -> tensor<?x?xf32>
//       CHECK:       %[[T1:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[T0:.+]], %[[STARG3]] : tensor<?x?xf32>, tensor<?xf32>)
//  CHECK-SAME:         outs(%[[STARG7]] : tensor<?x?xf32>)
//       CHECK:       %[[RESULT:.+]] = subtensor_insert %[[T1]] into %[[ARG7]]
//       CHECK:       scf.yield %[[RESULT]]
//       CHECK:     }
//       CHECK:     scf.yield %[[R1]]
//       CHECK:   }
//       CHECK:   return %[[R0]]

// -----

module {
  func @tensor_matmul_fusion(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
                             %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>,
			     %arg4: tensor<?x?xf32>, %arg5: tensor<?x?xf32>,
			     %arg6: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> // [M, N0] * [N0, N1]
    %1 = linalg.matmul ins(%0, %arg3 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg4 : tensor<?x?xf32>) -> tensor<?x?xf32> // [M, N1] * [N1, N2]
    %2 = linalg.matmul ins(%1, %arg5 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg6 : tensor<?x?xf32>) -> tensor<?x?xf32> // [M, N2] * [N2, N3]
    return %2 : tensor<?x?xf32>
  }
}
// CHECK-LABEL: func @tensor_matmul_fusion(
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG5:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG6:[a-zA-Z0-9_]+]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//       CHECK:   %[[R0:.+]] = scf.for %[[IV0:[a-zA-Z0-9_]+]] =
//  CHECK-SAME:     iter_args(%[[ARG8:.+]] = %[[ARG6]]) -> (tensor<?x?xf32>) {
//       CHECK:       %[[N3:.+]] = memref.dim %[[ARG8]], %[[C1]]
//       CHECK:       %[[STARG6:.+]] = subtensor %[[ARG8]][%[[IV0]], 0]
//  CHECK-SAME:         [%{{[a-zA-Z0-9_]+}}, %[[N3]]]
//       CHECK:       %[[N2:.+]] = memref.dim %[[ARG3]], %[[C1]]
//       CHECK:       %[[N1:.+]] = memref.dim %[[ARG1]], %[[C1]]
//       CHECK:       %[[STARG3:.+]] = subtensor %[[ARG3]][0, 0]
//  CHECK-SAME:         [%[[N1]], %[[N2]]]
//       CHECK:       %[[STARG4:.+]] = subtensor %[[ARG4]][%[[IV0]], 0]
//  CHECK-SAME:         [%{{[a-zA-Z0-9_]+}}, %[[N2]]]
//       CHECK:       %[[N0:.+]] = memref.dim %[[ARG0]], %[[C1]]
//       CHECK:       %[[STARG0:.+]] = subtensor %[[ARG0]][%[[IV0]], 0]
//  CHECK-SAME:         [%{{[a-zA-Z0-9_]+}}, %[[N0]]]
//       CHECK:       %[[STARG1:.+]] = subtensor %[[ARG1]][0, 0]
//  CHECK-SAME:         [%[[N0]], %[[N1]]]
//       CHECK:       %[[STARG2:.+]] = subtensor %[[ARG2]][%[[IV0]], 0]
//  CHECK-SAME:         [%{{[a-zA-Z0-9_]+}}, %[[N1]]]
//       CHECK:       %[[T0:.+]] = linalg.matmul
//  CHECK-SAME:         ins(%[[STARG0]], %[[STARG1]]
//  CHECK-SAME:         ) outs(%[[STARG2]] : tensor<?x?xf32>)
//       CHECK:       %[[T1:.+]] = linalg.matmul
//  CHECK-SAME:         ins(%[[T0]], %[[STARG3]]
//  CHECK-SAME:         ) outs(%[[STARG4]] : tensor<?x?xf32>)
//       CHECK:       %[[T2:.+]] = linalg.matmul
//  CHECK-SAME:         ins(%[[T1]], %[[ARG5]]
//  CHECK-SAME:         ) outs(%[[STARG6]] : tensor<?x?xf32>)
//       CHECK:       %[[R1:.+]] = subtensor_insert %[[T2]]
//  CHECK-SAME:         into %[[ARG8]][%[[IV0]], 0]
//       CHECK:       scf.yield %[[R1]]
//       CHECK:     }
//       CHECK:     return %[[R0]]
//       CHECK:   }
