// RUN: mlir-opt -pass-pipeline="func.func(test-linalg-tile-and-fuse{tile-sizes=16,32,64}),resolve-shaped-type-result-dims,canonicalize,cse" -split-input-file %s | FileCheck %s

module {
  func @three_op_fusion(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
                        %arg2: memref<?xf32>, %arg3 : memref<?x?xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = memref.dim %arg0, %c0 : memref<?x?xf32>
    %d1 = memref.dim %arg1, %c1 : memref<?x?xf32>
    %0 = memref.alloc(%d0, %d1) : memref<?x?xf32>
    linalg.fill ins(%cst : f32) outs(%0 : memref<?x?xf32>)
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
        %5 = arith.addf %arg4, %arg5 : f32
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
//       CHECK:     %[[SV_TEMP_1:.+]] = memref.subview %[[TEMP]][%[[IV0]], %[[IV1]]]
//   CHECK-DAG:     %[[SV_ARG2:.+]] = memref.subview %[[ARG2]][%[[IV1]]]
//   CHECK-DAG:     %[[SV_ARG3:.+]] = memref.subview %[[ARG3]][%[[IV0]], %[[IV1]]]
//   CHECK-DAG:     %[[SV_ARG0:.+]] = memref.subview %[[ARG0]][%[[IV0]], 0]
//   CHECK-DAG:     %[[SV_ARG1:.+]] = memref.subview %[[ARG1]][0, %[[IV1]]]
//       CHECK:     %[[SV_TEMP_2:.+]] = memref.subview %[[TEMP]][%[[IV0]], %[[IV1]]]
//       CHECK:     linalg.fill ins(%{{.+}}{{.*}}outs(%[[SV_TEMP_1]]
//       CHECK:     linalg.matmul
//  CHECK-SAME:       ins(%[[SV_ARG0]], %[[SV_ARG1]]
//  CHECK-SAME:         : memref<?x?xf32, #[[MAP2]]>, memref<?x?xf32, #[[MAP2]]>)
//  CHECK-SAME:       outs(%[[SV_TEMP_2]] : memref<?x?xf32, #[[MAP2]]>)
//       CHECK:     linalg.generic
//  CHECK-SAME:       ins(%[[SV_TEMP_1]], %[[SV_ARG2]]
//  CHECK-SAME:         : memref<?x?xf32, #[[MAP2]]>, memref<?xf32, #[[MAP3]]>)
//  CHECK-SAME:       outs(%[[SV_ARG3]] : memref<?x?xf32, #[[MAP2]]>)
//       CHECK:     scf.yield
//       CHECK:   }

// -----

module {
  func @sequence_of_matmul(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
                           %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>,
                           %arg4: memref<?x?xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = memref.dim %arg0, %c0 : memref<?x?xf32>
    %n1 = memref.dim %arg1, %c1 : memref<?x?xf32>
    %n2 = memref.dim %arg2, %c1 : memref<?x?xf32>
    %n3 = memref.dim %arg3, %c1 : memref<?x?xf32>
    %0 = memref.alloc(%m, %n1) : memref<?x?xf32>
    %1 = memref.alloc(%m, %n2) : memref<?x?xf32>
    linalg.fill ins(%cst : f32) outs(%0 : memref<?x?xf32>)
    linalg.matmul ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>)
      outs(%0 : memref<?x?xf32>)
    linalg.fill ins(%cst : f32) outs(%1 : memref<?x?xf32>)
    linalg.matmul ins(%0, %arg2 : memref<?x?xf32>, memref<?x?xf32>)
      outs(%1 : memref<?x?xf32>)
    linalg.fill ins(%cst : f32) outs(%arg4 : memref<?x?xf32>)
    linalg.matmul ins(%1, %arg3 : memref<?x?xf32>, memref<?x?xf32>)
      outs(%arg4 : memref<?x?xf32>)
    return
  }
}

//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 16)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0)[s0, s1] -> (-d0 + s1, -d0 + s0, 16)>


//       CHECK: func @sequence_of_matmul
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
//   CHECK-DAG:   %[[M:.+]] = memref.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[N1:.+]] = memref.dim %[[ARG1]], %[[C1]]
//   CHECK-DAG:   %[[N2:.+]] = memref.dim %[[ARG2]], %[[C1]]
//       CHECK:   %[[ALLOC1:.+]] = memref.alloc(%[[M]], %[[N1]])
//       CHECK:   %[[ALLOC2:.+]] = memref.alloc(%[[M]], %[[N2]])
//       CHECK:   scf.parallel (%[[IV0:.+]]) = (%[[C0]]) to (%[[M]])
//  CHECK-SAME:     step (%[[C16]]) {
//       CHECK:     %[[TILE_M:.+]] = affine.min #[[MAP0]](%[[IV0]])[%[[M]]]
//       CHECK:     %[[SV_ALLOC3:.+]] = memref.subview %[[ALLOC2]][%[[IV0]], 0]
//  CHECK-SAME:       [%[[TILE_M]], %[[N2]]]
//       CHECK:     %[[N3:.+]] = memref.dim %[[ARG4]], %[[C1]]
//       CHECK:     %[[SV_ARG4:.+]] = memref.subview %[[ARG4]][%[[IV0]], 0]
//  CHECK-SAME:       [%[[TILE_M]], %[[N3]]]
//       CHECK:     %[[M_2:.+]] = memref.dim %[[ARG4]], %[[C0]]
//       CHECK:     %[[TILE_M_3:.+]] = affine.min #[[MAP2]](%[[IV0]])[%[[M_2]], %[[M]]]
//       CHECK:     %[[SV_ARG4_2:.+]] = memref.subview %[[ARG4]][%[[IV0]], 0]
//  CHECK-SAME:       [%[[TILE_M_3]], %[[N3]]]
//       CHECK:     %[[SV_ALLOC1:.+]] = memref.subview %[[ALLOC1]][%[[IV0]], 0]
//  CHECK-SAME:       [%[[TILE_M]], %[[N1]]]
//       CHECK:     %[[TILE_M_5:.+]] = affine.min #[[MAP2]](%[[IV0]])[%[[M]], %[[M]]]
//       CHECK:     %[[N0:.+]] = memref.dim %[[ARG0]], %[[C1]]
//       CHECK:     %[[SV_ARG0:.+]] = memref.subview %[[ARG0]][%[[IV0]], 0]
//  CHECK-SAME:       [%[[TILE_M_5]], %[[N0]]]
//       CHECK:     %[[SV_ALLOC4:.+]] = memref.subview %[[ALLOC1]][%[[IV0]], 0]
//  CHECK-SAME:       [%[[TILE_M_5]], %[[N1]]]
//       CHECK:     linalg.fill ins(%{{.+}}{{.*}}outs(%[[SV_ALLOC1]]
//       CHECK:     linalg.matmul ins(%[[SV_ARG0]], %[[ARG1]]
//  CHECK-SAME:        : memref<?x?xf32, #[[MAP1]]>, memref<?x?xf32>)
//  CHECK-SAME:        outs(%[[SV_ALLOC4]] : memref<?x?xf32, #[[MAP1]]>)
//       CHECK:     linalg.fill ins(%{{.+}}{{.*}}outs(%[[SV_ALLOC3]]
//       CHECK:     linalg.matmul ins(%[[SV_ALLOC1]], %[[ARG2]]
//  CHECK-SAME:        : memref<?x?xf32, #[[MAP1]]>, memref<?x?xf32>)
//  CHECK-SAME:        outs(%[[SV_ALLOC3]] : memref<?x?xf32, #[[MAP1]]>)
//       CHECK:     linalg.fill ins(%{{.+}}{{.*}}outs(%[[SV_ARG4_2]]
//       CHECK:     linalg.matmul ins(%[[SV_ALLOC3]], %[[ARG3]]
//  CHECK-SAME:        : memref<?x?xf32, #[[MAP1]]>, memref<?x?xf32>)
//  CHECK-SAME:        outs(%[[SV_ARG4]] : memref<?x?xf32, #[[MAP1]]>)
//       CHECK:     scf.yield
//       CHECK:   }


// -----

module {
  func @tensor_op_fusion(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
                         %arg2: tensor<?x?xf32>, %arg3: tensor<?xf32>)
    -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %1 = tensor.dim %0, %c0 : tensor<?x?xf32>
    %2 = tensor.dim %0, %c1 : tensor<?x?xf32>
    %3 = linalg.init_tensor [%1, %2] : tensor<?x?xf32>
    %4 = linalg.generic
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]}
      ins(%0, %arg3 : tensor<?x?xf32>, tensor<?xf32>)
      outs(%3 : tensor<?x?xf32>) {
      ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
        %5 = arith.addf %arg4, %arg5 : f32
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
//   CHECK-DAG:       %[[STARG3:.+]] = tensor.extract_slice %[[ARG3]]
//   CHECK-DAG:       %[[STARG7:.+]] = tensor.extract_slice %[[ARG7]]
//   CHECK-DAG:       %[[STARG0:.+]] = tensor.extract_slice %[[ARG0]]
//   CHECK-DAG:       %[[STARG1:.+]] = tensor.extract_slice %[[ARG1]]
//   CHECK-DAG:       %[[STARG2:.+]] = tensor.extract_slice %[[ARG2]]
//       CHECK:       %[[T0:.+]] = linalg.matmul
//  CHECK-SAME:         ins(%[[STARG0]], %[[STARG1]] : tensor<?x?xf32>, tensor<?x?xf32>)
//  CHECK-SAME:         outs(%[[STARG2]] : tensor<?x?xf32>) -> tensor<?x?xf32>
//       CHECK:       %[[T1:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[T0:.+]], %[[STARG3]] : tensor<?x?xf32>, tensor<?xf32>)
//  CHECK-SAME:         outs(%[[STARG7]] : tensor<?x?xf32>)
//       CHECK:       %[[RESULT:.+]] = tensor.insert_slice %[[T1]] into %[[ARG7]]
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

//       CHECK: #[[MAP0:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 16)>
//       CHECK: #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (-d0 + s1, -d0 + s0, 16)>

//       CHECK: func @tensor_matmul_fusion(
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG5:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG6:[a-zA-Z0-9_]+]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//       CHECK:   %[[M:.+]] = tensor.dim %[[ARG0]], %c0 : tensor<?x?xf32>
//       CHECK:   %[[R0:.+]] = scf.for %[[IV0:[a-zA-Z0-9_]+]] =
//  CHECK-SAME:     iter_args(%[[ARG8:.+]] = %[[ARG6]]) -> (tensor<?x?xf32>) {
//       CHECK:     %[[TILE_M_1:.+]] = affine.min #[[MAP0]](%[[IV0]])[%[[M]]]
//       CHECK:     %[[N3:.+]] = tensor.dim %[[ARG8]], %[[C1]]
//       CHECK:     %[[STARG6:.+]] = tensor.extract_slice %[[ARG8]][%[[IV0]], 0]
//  CHECK-SAME:       [%[[TILE_M_1]], %[[N3]]]
//       CHECK:     %[[TILE_M_2:.+]] = affine.min #[[MAP1]](%[[IV0]])[%[[M]], %[[M]]]
//       CHECK:     %[[N2:.+]] = tensor.dim %[[ARG4]], %[[C1]]
//       CHECK:     %[[STARG4:.+]] = tensor.extract_slice %[[ARG4]][%[[IV0]], 0]
//  CHECK-SAME:       [%[[TILE_M_2]], %[[N2]]]
//       CHECK:     %[[N0:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//       CHECK:     %[[STARG0:.+]] = tensor.extract_slice %[[ARG0]][%[[IV0]], 0]
//  CHECK-SAME:       [%[[TILE_M_2]], %[[N0]]]
//       CHECK:     %[[N1:.+]] = tensor.dim %[[ARG2]], %[[C1]]
//       CHECK:     %[[STARG2:.+]] = tensor.extract_slice %[[ARG2]][%[[IV0]], 0]
//  CHECK-SAME:       [%[[TILE_M_2]], %[[N1]]]
//       CHECK:     %[[T0:.+]] = linalg.matmul
//  CHECK-SAME:       ins(%[[STARG0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32>
//  CHECK-SAME:       ) outs(%[[STARG2]] : tensor<?x?xf32>)
//       CHECK:     %[[T1:.+]] = linalg.matmul
//  CHECK-SAME:       ins(%[[T0]], %arg3 : tensor<?x?xf32>, tensor<?x?xf32>
//  CHECK-SAME:       ) outs(%[[STARG4]] : tensor<?x?xf32>)
//       CHECK:     %[[T2:.+]] = linalg.matmul
//  CHECK-SAME:       ins(%[[T1]], %arg5 : tensor<?x?xf32>, tensor<?x?xf32>
//  CHECK-SAME:       ) outs(%[[STARG6]] : tensor<?x?xf32>)
//       CHECK:     %[[R1:.+]] = tensor.insert_slice %[[T2]]
//  CHECK-SAME:       into %[[ARG8]][%[[IV0]], 0] [%[[TILE_M_1]], %[[N3]]]
//       CHECK:     scf.yield %[[R1]] : tensor<?x?xf32>
//       CHECK:   }
